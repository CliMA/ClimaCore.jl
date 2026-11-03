"""
    JETConfig

The single source of the JET reports that the tests ignore. Every JET check in
the tests passes `ignored_modules = JETConfig.ignored_modules()`, and the
checks that skip calls by function also pass
`function_filter = JETConfig.function_filter`; add to the lists here rather
than at a call site.

```julia
using JET
@isdefined(JETConfig) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "JETConfig.jl"),
)

@test_opt ignored_modules = JETConfig.ignored_modules() f(x)
```
"""
module JETConfig

import JET
import ClimaCore

# CUDA and the CUDA extension of ClimaCore are only loaded on a CUDA device
# (through `ClimaComms.@import_required_backends`), which may happen after this
# module is loaded, so their frames are looked up when a check runs. Their
# runtime dispatch is in the kernel-launch path, which is not ours to fix.
# `Field == Field` bottoms out in GPUArrays' generic `==` for device arrays,
# whose `mapreduce` accumulator inference cannot resolve; the dispatches are
# reported against `Base` frames reached through GPUArrays, so only
# `AnyFrameModule`, which matches any frame in the stack rather than the
# innermost, excludes them. CUDA 6 splits CUDA.jl into subpackages, with
# GPUArrays and most of the device code reached through CUDACore, and renames
# CUBLAS (used by `norm`) to cuBLAS.
function cuda_frames()
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
        JET.AnyFrameModule(ext),
    )
end

# On Julia 1.12, the message of a failed `@assert` is converted to a string
# behind a compiler barrier (Base._assert_tostring), so its conversion and the
# AssertionError constructed from it are reported as runtime dispatch. The
# first is reported in Base._assert_tostring, and the second in whichever
# method contains the `@assert` (e.g. Threads.@spawn in launch_pool_threads,
# ClimaComms.@assert, and IdSet in Base's printing of types), so it is matched
# by its call instead. Matching a report by method or by a custom matcher needs
# JET 0.12, which Julia 1.12 uses; earlier versions of Julia have no such
# reports.
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

# Besides CUDA and `@assert`, ignore the type instabilities from KrylovKit
# (eigsolve) and CoreLogging (@debug).
ignored_modules() = (
    cuda_frames()...,
    assert_frames...,
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

end # module JETConfig
