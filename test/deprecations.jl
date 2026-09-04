using Test
import ClimaCore
using ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces, Fields, Grids
using ClimaCore.CommonSpaces
import ClimaCore.Spaces:
    CenterFiniteDifferenceSpace,
    FaceFiniteDifferenceSpace,
    FiniteDifferenceSpace
import ClimaCore.Grids: FiniteDifferenceGrid
import ClimaCore.DataLayouts
import ClimaComms
ClimaComms.@import_required_backends

@testset "Deprecations" begin
    FT = Float64
    z_max = FT(30e3)
    z_elem = 64
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)

    # Deprecated methods:
    device = ClimaComms.device()
    grid = Grids.FiniteDifferenceGrid(Topologies.IntervalTopology(device, z_mesh))

    @test_deprecated Topologies.IntervalTopology(z_mesh)
    @test Topologies.IntervalTopology(z_mesh) ==
          Topologies.IntervalTopology(device, z_mesh)

    @test_deprecated FaceFiniteDifferenceSpace(z_mesh)
    @test FaceFiniteDifferenceSpace(z_mesh) ==
          FiniteDifferenceSpace(grid, Grids.CellFace())

    @test_deprecated CenterFiniteDifferenceSpace(z_mesh)
    @test CenterFiniteDifferenceSpace(z_mesh) ==
          FiniteDifferenceSpace(grid, Grids.CellCenter())

    @test_deprecated FiniteDifferenceGrid(z_mesh)
    @test FiniteDifferenceGrid(z_mesh) == grid
end

# The old "universal index" convention from ClimaCore's DataLayouts, which is
# still used by ClimaCoreTempestRemap (its netcdf.jl indexes layouts with
# `slab(data, e)[slab_index(i, j)]` and `data[vindex(v)]`).
@inline old_slab_index(i, j) = CartesianIndex(i, j, 1, 1, 1)
@inline old_vindex(v) = CartesianIndex(1, 1, 1, v, 1)

@testset "Deprecated universal CartesianIndex{5} indexing" begin
    ClimaComms.allowscalar(ClimaComms.device()) do
        FT = Float32

        for VIJH in (ClimaCore.DataLayouts.VIJFH, ClimaCore.DataLayouts.VIJHF)
            h_space = CubedSphereSpace(FT; radius = 1, h_elem = 2, n_quad_points = 4, VIJH)

            # Reading coordinates with 2 components (the ClimaCoreTempestRemap
            # crash site in def_space_coord, where the F range must be inserted
            # into the translated index).
            coords = Spaces.coordinates_data(h_space)
            @test eltype(coords) <: Geometry.LatLongPoint
            Nh = ClimaCore.DataLayouts.nelems(coords)
            for e in (1, Nh ÷ 2, Nh), (i, j) in ((1, 1), (2, 3), (4, 4))
                coord = ClimaCore.slab(coords, e)[old_slab_index(i, j)]
                @test coord === coords[1, i, j, e]
                @test coord === coords[CartesianIndex(i, j, 1, 1, e)]
            end

            # Reading and writing scalar field values (TempestRemap's
            # setindex!(::CFVariable, ::SpectralElementField2D, ::Colon)).
            data = Fields.field_values(Fields.coordinate_field(h_space).lat)
            for e in (1, Nh), (i, j) in ((1, 1), (2, 3))
                @test ClimaCore.slab(data, e)[old_slab_index(i, j)] ===
                      data[1, i, j, e]
            end
            ClimaCore.slab(data, Nh)[old_slab_index(2, 3)] = 100
            @test data[1, 2, 3, Nh] === FT(100)
        end

        z_elem = 10
        c_space =
            ColumnSpace(FT; z_elem, z_min = 0, z_max = 1000, staggering = CellCenter())

        # Reading column data at a vertical index (TempestRemap's
        # setindex!(::CFVariable, ::FiniteDifferenceField, ::Colon)).
        c_coords = Spaces.coordinates_data(c_space)
        for v in (1, z_elem ÷ 2, z_elem)
            @test c_coords[old_vindex(v)] === c_coords[v, 1, 1, 1]
        end

        # Broadcasting parent arrays of column data into vectors, and linearly
        # indexing them in vertical order (TempestRemap's def_space_coord for
        # finite difference spaces).
        z = zeros(FT, z_elem)
        z .= Array(parent(c_coords))
        @test z == [c_coords[v, 1, 1, 1].z for v in 1:z_elem]
        f_coords = Spaces.coordinates_data(Spaces.FaceFiniteDifferenceSpace(c_space))
        z_bnds = zeros(FT, 2, z_elem)
        f_coords_cpu = Array(parent(f_coords))
        z_bnds[1, :] .= f_coords_cpu[1:(end - 1)]
        z_bnds[2, :] .= f_coords_cpu[2:end]
        @test z_bnds[1, :] == [f_coords[v, 1, 1, 1].z for v in 1:z_elem]
        @test z_bnds[2, :] == [f_coords[v + 1, 1, 1, 1].z for v in 1:z_elem]

        # Copying the parent array of a column slice into a matrix row
        # (TempestRemap's setindex!(::CFVariable, ::ExtrudedFiniteDifferenceField,
        # ::Colon), which assigns parent(column(data, i, j, h)) to var[col, :]).
        space_3d = ExtrudedCubedSphereSpace(
            FT;
            z_elem,
            z_min = 0,
            z_max = 1000,
            radius = 1,
            h_elem = 2,
            n_quad_points = 4,
            staggering = CellCenter(),
        )
        data_3d = Fields.field_values(Fields.coordinate_field(space_3d).z)
        column_parent = parent(ClimaCore.column(data_3d, 2, 3, 1))
        @test length(column_parent) == z_elem
        var = zeros(FT, 3, z_elem)
        var[1, :] = column_parent
        @test var[1, :] == [data_3d[v, 2, 3, 1] for v in 1:z_elem]
    end
end

nothing
