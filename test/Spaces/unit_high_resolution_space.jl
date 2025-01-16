#=
julia --project=.buildkite
using Revise; include("test/Spaces/unit_high_resolution_space.jl")
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore.CommonSpaces
using ClimaCore: Spaces

# TODO: we should probably move these to datalayouts and just test high resolution kernels.
@testset "Construct high resolution space" begin
    space = ExtrudedCubedSphereSpace(
        Float32;
        radius = 1.0,
        h_elem = 105,
        z_elem = 10,
        z_min = 1.0,
        z_max = 2.0,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    @test space isa Spaces.CenterExtrudedFiniteDifferenceSpace

    space =
        CubedSphereSpace(Float32; radius = 1.0, n_quad_points = 4, h_elem = 105)
    @test space isa Spaces.SpectralElementSpace2D

    space = ColumnSpace(
        Float32;
        z_elem = 500,
        z_min = 0,
        z_max = 10,
        staggering = CellCenter(),
    )
    @test space isa Spaces.FiniteDifferenceSpace
end
