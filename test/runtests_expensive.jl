using SafeTestsets
using Base: operator_associativity

#! format: off
# Order of tests is intended to reflect dependency order of functionality

#= TODO: add windows test back in. Currently getting
ReadOnlyMemoryError()
ERROR: Package ClimaCore errored during testing (exit code: 541541187)
Stacktrace:
 [1] pkgerror(msg::String)
=#
if !Sys.iswindows()

@safetestset "Sphere spaces" begin @time include("Spaces/sphere.jl") end
@safetestset "Distributed spaces" begin @time include("Spaces/distributed.jl") end
@safetestset "Fields" begin @time include("Fields/field.jl") end
@safetestset "Limiter" begin @time include("Limiters/limiter.jl") end
@safetestset "Distributed limiters" begin @time include("Limiters/distributed.jl") end
@safetestset "Distributed topology" begin @time include("Topologies/distributed.jl") end
@safetestset "Terrain warp" begin @time include("Spaces/terrain_warp.jl") end

end

#! format: on
