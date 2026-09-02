#=
    julia --project=test test/precompile_workload.jl
=#
using Test

# `src/precompile_workload.jl` runs inside ClimaCore's own precompilation, where
# CUDA.jl (a weak dependency) is never loaded. Resolving its device through
# `ClimaComms.device()` would therefore abort precompilation with "Loading
# CUDA.jl is required to use CUDADevice" for any run script that exports
# `CLIMACOMMS_DEVICE=CUDA` before `using ClimaCore`. The workload is guarded on
# `jl_generating_output`, so precompiling in a subprocess with that variable set
# is the only way to exercise it.
@testset "precompile workload ignores CLIMACOMMS_DEVICE" begin
    code = "Base.compilecache(Base.identify_package(\"ClimaCore\"))"
    project = Base.active_project()
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $code`
    env = copy(ENV)
    env["CLIMACOMMS_DEVICE"] = "CUDA"
    mktemp() do path, io
        succeeded = success(pipeline(setenv(cmd, env); stdout = io, stderr = io))
        close(io)
        succeeded ||
            @info "ClimaCore precompilation log" log = read(path, String)
        @test succeeded
    end
end
