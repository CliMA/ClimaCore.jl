"""
    JETConfig

The single source of the JET reports that the tests ignore. Every `@test_opt`
check in the tests is written with [`JETConfig.@test_with_jet`](@ref), which
passes the lists defined here to JET; add to these lists rather than at a call
site.

```julia
using JET
@isdefined(JETConfig) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "JETConfig.jl"),
)

JETConfig.@test_with_jet f(x)
JETConfig.@test_with_jet cuda_ext = true f(x)
JETConfig.@test_with_jet function_filter = true f(x)
```

The checks that call JET in another way (e.g. `JET.@report_opt`) pass
`ignored_modules = JETConfig.ignored_modules()` themselves.
"""
module JETConfig

import JET
import ClimaCore

# CUDA and the CUDA extension of ClimaCore are only loaded on a CUDA device
# (through `ClimaComms.@import_required_backends`), which may happen after this
# module is loaded, so their frames are looked up when a check runs. Their
# runtime dispatch is in the kernel-launch path, which is not ours to fix.
# ClimaCoreCUDAExt itself is ClimaCore code, and since `AnyFrameModule` matches
# any frame in the stack, ignoring it hides every report beneath the extension
# (which is nearly every call on the GPU), so it is only ignored by the checks
# that opt in with `cuda_ext = true`.
# `Field == Field` bottoms out in GPUArrays' generic `==` for device arrays,
# whose `mapreduce` accumulator inference cannot resolve; the dispatches are
# reported against `Base` frames reached through GPUArrays, so only
# `AnyFrameModule`, which matches any frame in the stack rather than the
# innermost, excludes them. CUDA 6 splits CUDA.jl into subpackages, with
# GPUArrays and most of the device code reached through CUDACore, and renames
# CUBLAS (used by `norm`) to cuBLAS.
#
# CUDA's `@memoize` (used by `CUDA.context()`, and so by every query of the
# device) used to expand inline into CUDA's own functions, where
# `AnyFrameModule(CUDA)` caught its reports. As of GPUToolbox 3, it calls
# GPUToolbox.memoize_grow_slow!, whose slot type is not inferred, so the
# runtime dispatch in it is reported against GPUToolbox frames.
toolbox_frames(core) =
    isdefined(core, :GPUToolbox) ? (JET.AnyFrameModule(core.GPUToolbox),) : ()

function cuda_frames(; cuda_ext::Bool)
    ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
    isnothing(ext) && return ()
    CUDA = ext.CUDA
    core = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA
    blas = isdefined(CUDA, :cuBLAS) ? CUDA.cuBLAS : CUDA.CUBLAS
    return (
        JET.AnyFrameModule(CUDA),
        JET.AnyFrameModule(core),
        JET.AnyFrameModule(core.GPUArrays),
        JET.AnyFrameModule(blas),
        toolbox_frames(core)...,
        (cuda_ext ? (JET.AnyFrameModule(ext),) : ())...,
    )
end

# As of Julia 1.12, the message of a failed `@assert` is converted to a string
# behind a compiler barrier (Base._assert_tostring), so its conversion and the
# AssertionError constructed from it are reported as runtime dispatch. The
# first is reported in Base._assert_tostring, and the second in whichever
# method contains the `@assert` (e.g. Threads.@spawn in launch_pool_threads,
# ClimaComms.@assert, and IdSet in Base's printing of types), so it is matched
# by its call instead. Matching a report by method or by a custom matcher needs
# JET 0.12, which requires at least Julia 1.12.
@static if isdefined(JET, :LastFrameMethod) && isdefined(Base, :_assert_tostring)
    struct AssertionErrorDispatch <: JET.ReportMatcher end
    function JET.match_report(
        ::AssertionErrorDispatch,
        @nospecialize(report::JET.InferenceErrorReport),
    )
        tt = report.sig.tt
        return report isa JET.RuntimeDispatchReport &&
               tt isa Type &&
               tt <: Tuple{Type{AssertionError}, Vararg}
    end
    const assert_frames =
        (JET.LastFrameMethod(Base._assert_tostring), AssertionErrorDispatch())
else
    const assert_frames = ()
end

# The loop body that launch_pool_threads spawns on each thread is a
# @nospecialize'd function barrier, so calling it is a runtime dispatch by
# design. JET analyzes the closures passed to Task only when their type is
# concrete; before Julia 1.13 the closure was parametric in the captured `f`
# (a UnionAll when `f::Any`), but as of 1.13 `f` is stored in a field of type
# Any, so the dispatch is now reported.
@static if isdefined(JET, :LastFrameMethod)
    struct PoolThreadsDispatch <: JET.ReportMatcher end
    function JET.match_report(
        ::PoolThreadsDispatch,
        @nospecialize(report::JET.InferenceErrorReport),
    )
        report isa JET.RuntimeDispatchReport || return false
        def = report.vst[end].linfo.def
        return def isa Method &&
               def.module === ClimaCore.DataLayouts &&
               occursin("launch_pool_threads", String(def.name))
    end
    const pool_frames = (PoolThreadsDispatch(),)
else
    const pool_frames = ()
end

# As of Julia 1.13, wait() reaches the scheduler task through a OncePerThread
# whose initializer constructs a Task, and JET's analysis of that Task's body
# recurses back into wait(), so any wait(::Task) (e.g. in launch_pool_threads)
# is reported as failing to optimize due to recursion in OncePerThread.
@static if isdefined(JET, :LastFrameMethod) && isdefined(Base, :get_sched_task)
    struct OncePerThreadRecursion <: JET.ReportMatcher end
    function JET.match_report(
        ::OncePerThreadRecursion,
        @nospecialize(report::JET.InferenceErrorReport),
    )
        report isa JET.OptimizationFailureReport || return false
        sig = report.vst[end].linfo.specTypes
        return sig <: Tuple{Base.OncePerThread, Vararg} ||
               sig <: Tuple{typeof(getindex), Base.OncePerThread, Vararg}
    end
    const scheduler_frames = (OncePerThreadRecursion(),)
else
    const scheduler_frames = ()
end

# Besides CUDA, `@assert`, and the threading launcher, ignore the type
# instabilities from KrylovKit (eigsolve) and CoreLogging (@debug). Pass
# `cuda_ext = true` to also ignore every report beneath ClimaCoreCUDAExt.
ignored_modules(; cuda_ext::Bool = false) = (
    cuda_frames(; cuda_ext)...,
    assert_frames...,
    pool_frames...,
    scheduler_frames...,
    JET.AnyFrameModule(ClimaCore.MatrixFields.KrylovKit),
    JET.AnyFrameModule(Base.CoreLogging),
)

# Calls that are dispatched at runtime outside of ClimaCore:
#  - Base.mapreduce_empty, the error path of reductions over empty collections
#  - Core.UnionAll, constructed in typejoin as of Julia 1.10, which is reached
#    through the error message machinery of sprint in Threads.threading_run,
#    used by parallelize_over
# JET keys its analysis cache on the filter, so the checks that do not need it
# leave it as the default.
function_filter(@nospecialize f) =
    f !== Base.mapreduce_empty && f !== Core.UnionAll

"""
    @test_with_jet [cuda_ext = false] [function_filter = false] [jetconfigs...] f(args...)

Call `JET.@test_opt` on `f(args...)`, ignoring the reports that are listed in
`JETConfig`. The options are literal `Bool`s:

  - `cuda_ext = true` also ignores every report beneath ClimaCoreCUDAExt (see
    `ignored_modules`), which only matters on the GPU
  - `function_filter = true` also skips the calls that are dispatched at runtime
    outside of ClimaCore (see `function_filter`)

Any other option is passed on to `JET.@test_opt` (e.g. `broken = true`).
"""
macro test_with_jet(args...)
    isempty(args) && error("@test_with_jet requires a function call")
    cuda_ext = false
    filter_functions = false
    jetconfigs = Any[]
    for arg in args[1:(end - 1)]
        Meta.isexpr(arg, :(=)) ||
            error("@test_with_jet options must be of the form `name = value`")
        (name, value) = arg.args
        if name === :cuda_ext || name === :function_filter
            value isa Bool || error("@test_with_jet: $name must be true or false")
            name === :cuda_ext ? (cuda_ext = value) : (filter_functions = value)
        else
            push!(jetconfigs, arg)
        end
    end
    configs = Any[:(ignored_modules = $ignored_modules(; cuda_ext = $cuda_ext))]
    filter_functions && push!(configs, :(function_filter = $function_filter))
    return esc(
        Expr(
            :macrocall,
            GlobalRef(JET, Symbol("@test_opt")),
            __source__,
            configs...,
            jetconfigs...,
            args[end],
        ),
    )
end

end # module JETConfig
