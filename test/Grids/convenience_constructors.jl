#=
julia --project
using Revise; include(joinpath("test", "Grids", "convenience_constructors.jl"))
=#
import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore: Grids, Topologies, Meshes
using Test

@testset "Convenience constructors" begin
    grid = Grids.ExtrudedCubedSphereGrid(;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        Nq = 4,
    )
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid.horizontal_grid).mesh isa
          Meshes.EquiangularCubedSphere

    grid = Grids.CubedSphereGrid(; radius = 10, Nq = 4, h_elem = 10)
    @test grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid).mesh isa Meshes.EquiangularCubedSphere

    grid = Grids.ColumnGrid(; z_elem = 10, z_min = 0, z_max = 1)
    @test grid isa Grids.FiniteDifferenceGrid

    grid = Grids.Box3DGrid(;
        z_elem = 10,
        x_min = 0,
        x_max = 1,
        y_min = 0,
        y_max = 1,
        z_min = 0,
        z_max = 10,
        x1periodic = false,
        x2periodic = false,
        Nq = 4,
        n1 = 3,
        n2 = 4,
    )
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid.horizontal_grid).mesh isa Meshes.RectilinearMesh

    grid = Grids.SliceXZGrid(;
        z_elem = 10,
        x_min = 0,
        x_max = 1,
        z_min = 0,
        z_max = 1,
        x1periodic = false,
        Nq = 4,
        n1 = 4,
    )
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid1D
    @test Grids.topology(grid.horizontal_grid).mesh isa Meshes.IntervalMesh

    grid = Grids.RectangleXYGrid(;
        x_min = 0,
        x_max = 1,
        y_min = 0,
        y_max = 1,
        x1periodic = false,
        x2periodic = false,
        Nq = 4,
        n1 = 3,
        n2 = 4,
    )
    @test grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid).mesh isa Meshes.RectilinearMesh
end
