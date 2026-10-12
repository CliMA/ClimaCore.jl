# Temporary diagnostic: profile the device-side activity of flagship AMIP
# coupling steps. Run with the same arguments as experiments/AMIP/run_simulation.jl.
import CUDA
include(abspath(joinpath(ENV["COUPLER_PATH"], "experiments", "AMIP", "code_loading.jl")))
climacore = only(m for (k, m) in Base.loaded_modules if k.name == "ClimaCore")
@info "ClimaCore (getidx is only defined before the merge of this PR)" pathof(climacore) isdefined(climacore.Operators, :getidx)
config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)
step! = getfield(parentmodule(run!), :step!)
@info "Compiling step!..."
step!(cs)
step!(cs)
n_steps = 20
@info "Profiling $n_steps steps..."
res = CUDA.@profile external = false begin
    for _ in 1:n_steps
        step!(cs)
    end
end
show(IOContext(stdout, :limit => false), res)
println()
