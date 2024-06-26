#=
julia --project
using Revise; include(joinpath("test", "Grids", "convenience_constructors.jl"))
=#
import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore: Grids, Topologies, Meshes
using Test

@testset "Convenience constructors" begin
    grid = Grids.ExtrudedCubedSphereGrid()
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid.horizontal_grid).mesh isa
          Meshes.EquiangularCubedSphere

    grid = Grids.CubedSphereGrid()
    @test grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid).mesh isa Meshes.EquiangularCubedSphere

    grid = Grids.ColumnGrid()
    @test grid isa Grids.FiniteDifferenceGrid

    grid = Grids.Box3DGrid()
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid.horizontal_grid).mesh isa Meshes.RectilinearMesh

    grid = Grids.SliceXZGrid()
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid1D
    @test Grids.topology(grid.horizontal_grid).mesh isa Meshes.IntervalMesh

    grid = Grids.RectangleXYGrid()
    @test grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid).mesh isa Meshes.RectilinearMesh
end
