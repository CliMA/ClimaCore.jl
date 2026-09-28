# Slices of a multi-column space follow the convention of the space it generalizes:
# columns without horizontal coordinates behave like a `FiniteDifferenceSpace`, columns
# at (lat, long) points like an extruded spectral-element space on the sphere.
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Domains, Fields, Geometry, Grids, Meshes, Spaces, Topologies
import ClimaCore.CommonSpaces: MultiColumnSpace, ExtrudedCubedSphereSpace
import ClimaCore.Utilities: half

values(field) = vec(Array(parent(field)))
coordinate_type(space) = eltype(Fields.coordinate_field(space))
# Centers are indexed by integers, faces by half-integers
level_indices(space) =
    Spaces.staggering(space) isa Grids.CellCenter ? (1:Spaces.nlevels(space)) :
    [(v - 1) + half for v in 1:Spaces.nlevels(space)]

@testset "Columns without horizontal coordinates behave like a single column [$FT]" for FT in
                                                                                        (
    Float32,
    Float64,
)
    device = ClimaComms.device()
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(FT(4));
        boundary_names = (:bottom, :top),
    )
    # Stretched so that the vertical metric terms are not all equal
    z_mesh = Meshes.IntervalMesh(z_domain, Meshes.ExponentialStretching(FT(2)); nelems = 8)
    ncols = 3
    for staggering in (Grids.CellCenter(), Grids.CellFace())
        multi = MultiColumnSpace(
            FT;
            ncolumns = ncols, z_elem = 8, z_min = zero(FT), z_max = FT(4), z_mesh,
            staggering, device,
        )
        topology =
            Topologies.IntervalTopology(ClimaComms.SingletonCommsContext(device), z_mesh)
        single = Spaces.FiniteDifferenceSpace(topology, staggering)

        @test multi isa Spaces.AbstractFiniteDifferenceSpace
        @test Spaces.ncolumns(multi) == ncols
        @test Spaces.nlevels(multi) == Spaces.nlevels(single)
        @test Spaces.topology(multi) == Spaces.topology(single)
        @test Spaces.vertical_topology(multi) == Spaces.vertical_topology(single)
        @test Meshes.domain(Spaces.grid(multi)) == Meshes.domain(Spaces.grid(single))
        @test Spaces.global_geometry(multi) == Spaces.global_geometry(single)
        @test (Spaces.z_min(multi), Spaces.z_max(multi)) ==
              (Spaces.z_min(single), Spaces.z_max(single))
        @test coordinate_type(multi) == Geometry.ZPoint{FT}
        @test eltype(Spaces.local_geometry_data(multi)) ==
              eltype(Spaces.local_geometry_data(single))

        # The horizontal space is the first level, with `ZPoint` coordinates
        hspace = Spaces.horizontal_space(multi)
        @test hspace isa Spaces.MultiPointSpace
        @test hspace == Spaces.level(multi, 1)
        @test coordinate_type(hspace) == Geometry.ZPoint{FT}
        @test Spaces.ncolumns(hspace) == ncols
        @test Spaces.quadrature_style(hspace) === nothing
        @test Spaces.node_horizontal_length_scale(hspace) == 1

        # Every level is `ncols` copies of the single column's level
        for v in level_indices(multi)
            level_multi = Spaces.level(multi, v)
            level_single = Spaces.level(single, v)
            @test level_multi isa Spaces.MultiPointSpace
            @test coordinate_type(level_multi) == Geometry.ZPoint{FT}
            @test values(Fields.coordinate_field(level_multi).z) ==
                  repeat(values(Fields.coordinate_field(level_single).z), ncols)
            for h in 1:ncols
                point = Spaces.slab(level_multi, h)
                @test point isa Spaces.PointSpace
                @test values(Fields.local_geometry_field(point)) ==
                      values(Fields.local_geometry_field(level_single))
            end
        end

        # Every column is the single column
        for h in 1:ncols
            column = Spaces.column(multi, 1, 1, h)
            @test column isa Spaces.FiniteDifferenceSpace
            @test coordinate_type(column) == Geometry.ZPoint{FT}
            @test values(Fields.local_geometry_field(column)) ==
                  values(Fields.local_geometry_field(single))
            @test Spaces.horizontal_space(column) == Spaces.level(column, 1)
        end

        # Level and column views of a field agree with the single column's
        f_multi = Fields.coordinate_field(multi).z .^ 2
        f_single = Fields.coordinate_field(single).z .^ 2
        @test Fields.field2array(f_multi) == repeat(Fields.field2array(f_single), 1, ncols)
        for v in level_indices(multi), h in 1:ncols
            @test values(Fields.column(Fields.level(f_multi, v), 1, 1, h)) ==
                  values(Fields.level(f_single, v))
        end
    end
end

@testset "Columns at (lat, long) points behave like an extruded sphere space [$FT]" for FT in
                                                                                        (
    Float32,
    Float64,
)
    device = ClimaComms.device()
    points = [
        Geometry.LatLongPoint(FT(0), FT(0)),
        Geometry.LatLongPoint(FT(30), FT(90)),
        Geometry.LatLongPoint(FT(-45), FT(-120)),
    ]
    kwargs = (; z_elem = 8, z_min = zero(FT), z_max = FT(4), radius = FT(6e6))
    for staggering in (Grids.CellCenter(), Grids.CellFace())
        multi = MultiColumnSpace(FT; points, staggering, device, kwargs...)
        sphere = ExtrudedCubedSphereSpace(
            FT;
            h_elem = 2,
            n_quad_points = 2,
            staggering,
            kwargs...,
        )

        @test Spaces.global_geometry(multi) isa Geometry.ShallowSphericalGlobalGeometry
        @test typeof(Spaces.global_geometry(multi)) ==
              typeof(Spaces.global_geometry(sphere))
        @test coordinate_type(multi) == coordinate_type(sphere) ==
              Geometry.LatLongZPoint{FT}
        @test Spaces.vertical_topology(multi) == Spaces.vertical_topology(sphere)
        # No horizontal topology, so none of the horizontal grid's mesh either
        @test_throws ErrorException Spaces.topology(multi)

        # Levels carry 3D points, the horizontal space the 2D points of the columns
        for v in level_indices(multi)
            @test coordinate_type(Spaces.level(multi, v)) ==
                  coordinate_type(Spaces.level(sphere, v)) ==
                  Geometry.LatLongZPoint{FT}
        end
        hspace = Spaces.horizontal_space(multi)
        @test coordinate_type(hspace) ==
              coordinate_type(Spaces.horizontal_space(sphere)) ==
              Geometry.LatLongPoint{FT}
        @test hspace != Spaces.level(multi, 1)
        coords = Fields.coordinate_field(hspace)
        @test values(coords.lat) == getfield.(points, :lat)
        @test values(coords.long) == getfield.(points, :long)

        # Every column sits at its point, on the vertical mesh
        z = values(Fields.coordinate_field(Spaces.column(sphere, 1, 1, 1)).z)
        for h in 1:length(points)
            column_coords = Fields.coordinate_field(Spaces.column(multi, 1, 1, h))
            @test all(==(points[h].lat), values(column_coords.lat))
            @test all(==(points[h].long), values(column_coords.long))
            @test values(column_coords.z) == z
        end
    end
end
