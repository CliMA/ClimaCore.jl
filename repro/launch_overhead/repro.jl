# Host time of the Adapt conversion of a ClimaCore data layout for a kernel launch, and the
# inferred code behind it, after one of several orders of first use.
# Run as `julia --project=<env> repro.jl <mode>`; prints one line of results.
#
# Modes:
#   toplevel     a CuArray is allocated at top level before the first conversion
#   infer_first  the scope query of the Adapt rule is inferred before any CUDA call
#   in_function  allocation and first conversion inside one function, called once
#   spawn        the first conversion in a task started with Threads.@spawn
#   distributed  as in_function, right after `import Distributed`
#   ready_task   as in_function, right after `@async nothing`
# Run it with standard output to a file, not to a pipe (see README.md).
import CUDA, ClimaCore, Adapt
import ClimaCore: DataLayouts
const ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
const DL = DataLayouts
mode = ARGS[1]

host_query() = DL.is_subscope(ext.ThisHost(), ext.ThisKernel())
make_data() = DL.VIJFH{Float32, 74, 4, 4, nothing}(CUDA.CuArray{Float32}, 100)
first_use() = Adapt.adapt(CUDA.KernelAdaptor(), make_data())

if mode == "infer_first"
    code_typed(host_query, ())
elseif mode == "in_function"
    first_use()
elseif mode == "distributed"
    import Distributed # its __init__ schedules a task that is ready to run
    first_use()
elseif mode == "ready_task"
    @async nothing
    first_use()
elseif mode == "spawn"
    fetch(Threads.@spawn first_use())
elseif mode != "toplevel"
    error("unknown mode $mode")
end

data = make_data()
to = CUDA.KernelAdaptor()
Adapt.adapt(to, data)
ci_q, _ = only(code_typed(DL.is_subscope, (typeof(ext.ThisHost()), ext.ThisKernel)))
ci_r, _ = only(code_typed(DL.rebuild, typeof.((data, Adapt.adapt(to, parent(data))))))
n_apply(ci) = count(s -> occursin("apply_type", string(s)), ci.code)
n = 10^5
t = minimum(1:3) do _
    @elapsed for _ in 1:n
        Adapt.adapt(to, data)
    end
end
println(
    rpad(mode, 12),
    " is_subscope(ThisHost, ThisKernel): ", length(ci_q.code), " statements, ",
    n_apply(ci_q), " apply_type; rebuild: ", length(ci_r.code), " statements, ",
    n_apply(ci_r), " apply_type; adapt ", round(t / n * 1e9; digits = 1), " ns",
)
