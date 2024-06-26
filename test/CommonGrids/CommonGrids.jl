#=
julia --project
using Revise; include(joinpath("test", "CommonGrids", "CommonGrids.jl"))
=#
import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore.CommonGrids
using ClimaCore:
    Geometry,
    Hypsography,
    Fields,
    Spaces,
    Grids,
    Topologies,
    Meshes,
    DataLayouts
using Test

@testset "Convenience constructors" begin
    function warp_surface(coord)
        # sin²(x) form ground elevation
        x = Geometry.component(coord, 1)
        FT = eltype(x)
        hc = FT(500.0)
        h = hc * FT(sin(π * x / 25000)^2)
        return h
    end

    grid = ExtrudedCubedSphereGrid(;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        horizontal_layout_type = DataLayouts.IJHF,
    )
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid.horizontal_grid).mesh isa
          Meshes.EquiangularCubedSphere

    function hypsography_fun(h_grid, z_grid)
        h_space = Spaces.SpectralElementSpace2D(h_grid)
        cf = Fields.coordinate_field(h_space)
        warp_fn = warp_surface # closure
        z_surface = map(cf) do coord
            Geometry.ZPoint(warp_fn(coord))
        end
        Hypsography.LinearAdaption(z_surface)
    end

    grid = ExtrudedCubedSphereGrid(;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        hypsography_fun,
    )
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid.horizontal_grid).mesh isa
          Meshes.EquiangularCubedSphere
    @test Grids.topology(grid.horizontal_grid).mesh isa
          Meshes.EquiangularCubedSphere

    grid = CubedSphereGrid(; radius = 10, n_quad_points = 4, h_elem = 10)
    @test grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid).mesh isa Meshes.EquiangularCubedSphere

    grid = ColumnGrid(; z_elem = 10, z_min = 0, z_max = 1)
    @test grid isa Grids.FiniteDifferenceGrid

    grid = Box3DGrid(;
        z_elem = 10,
        x_min = 0,
        x_max = 1,
        y_min = 0,
        y_max = 1,
        z_min = 0,
        z_max = 10,
        periodic_x = false,
        periodic_y = false,
        n_quad_points = 4,
        x_elem = 3,
        y_elem = 4,
    )
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid.horizontal_grid).mesh isa Meshes.RectilinearMesh

    grid = SliceXZGrid(;
        z_elem = 10,
        x_min = 0,
        x_max = 1,
        z_min = 0,
        z_max = 1,
        periodic_x = false,
        n_quad_points = 4,
        x_elem = 4,
    )
    @test grid isa Grids.ExtrudedFiniteDifferenceGrid
    @test grid.horizontal_grid isa Grids.SpectralElementGrid1D
    @test Grids.topology(grid.horizontal_grid).mesh isa Meshes.IntervalMesh

    grid = RectangleXYGrid(;
        x_min = 0,
        x_max = 1,
        y_min = 0,
        y_max = 1,
        periodic_x = false,
        periodic_y = false,
        n_quad_points = 4,
        x_elem = 3,
        y_elem = 4,
    )
    @test grid isa Grids.SpectralElementGrid2D
    @test Grids.topology(grid).mesh isa Meshes.RectilinearMesh
end
