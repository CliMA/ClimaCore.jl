#=
julia --project
using Revise; include(joinpath("test", "Meshes", "opt_meshes.jl"))
=#
using Test
using SparseArrays
using JET

import ClimaCore: ClimaCore, Domains, Meshes, Geometry

@testset "monotonic_check" begin
    faces = range(Geometry.ZPoint(0), Geometry.ZPoint(10); length = 11)
    cfaces = collect(faces)
    @test_opt Meshes.monotonic_check(faces)
    @test_opt Meshes.monotonic_check(cfaces)
    @test 0 == @allocated Meshes.monotonic_check(cfaces)
end
