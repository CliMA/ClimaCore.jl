using Test
import ClimaCore.CommonSpaces: ExtrudedCubedSphereSpace, ColumnSpace
import ClimaCore.Remapping
import ClimaCore.Remapping:
    PressureInterpolator, interpolate_pressure!, interpolate_pressure!!, update!
import ClimaCore: Fields, Geometry, Grids, Meshes, Spaces
import ClimaInterpolations


@testset "Construction of pressure spaces" begin
    for FT in (Float32, Float64)
        extruded_space = ExtrudedCubedSphereSpace(FT;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            radius = 10,
            h_elem = 10,
            n_quad_points = 4,
            staggering = Grids.CellCenter(),
        )

        pfull_extruded_space =
            Remapping.construct_pressure_space(FT, extruded_space, [0.0, 1.0, 10.0, 100.0])
        @test pfull_extruded_space.grid isa Grids.ExtrudedFiniteDifferenceGrid
        @test pfull_extruded_space.grid.horizontal_grid isa Grids.SpectralElementGrid2D
        @test Grids.topology(pfull_extruded_space.grid.horizontal_grid).mesh isa
              Meshes.EquiangularCubedSphere
        @test pfull_extruded_space.staggering == Grids.CellFace()
        @test pfull_extruded_space.grid.vertical_grid.topology.mesh.faces ==
              Geometry.PPoint.([0.0, 1.0, 10.0, 100.0])

        col_space = ColumnSpace(FT;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            staggering = Grids.CellCenter(),
        )

        pfull_col_space =
            Remapping.construct_pressure_space(FT, col_space, [0.0, 1.0, 10.0, 100.0])
        @test pfull_col_space.grid isa Grids.FiniteDifferenceGrid
        @test pfull_col_space.staggering == Grids.CellFace()
        @test pfull_col_space.grid.topology.mesh.faces ==
              Geometry.PPoint.([0.0, 1.0, 10.0, 100.0])
    end
end

@testset "Vertical interpolation to pressure coordinates for ExtrudedFiniteDifferenceSpace" begin
    for FT in (Float32, Float64)
        extruded_space = ExtrudedCubedSphereSpace(FT;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            radius = 10,
            h_elem = 10,
            n_quad_points = 4,
            staggering = Grids.CellCenter(),
        )
        col_space = ColumnSpace(FT;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            staggering = Grids.CellCenter(),
        )
        for space in (extruded_space, col_space)
            pfull_field = fill(FT(1.0), space)
            pfull_field_array = Fields.field2array(pfull_field)
            pfull_field_array .= collect(10:-1:1)
            pfull_levels = [0.0, 1.0, 5.5, 10.0, 12.0]

            # Pressures cannot be in reversed order
            @test_throws ErrorException PressureInterpolator(pfull_field, [10.0, 5.5, 1.0])

            pfull_intp = PressureInterpolator(pfull_field, pfull_levels)
            pfull_field_array = Fields.field2array(Remapping.pfull_field(pfull_intp))

            # Check the pressures along each column is sorted
            @test pfull_field_array == repeat(FT.(1:10), 1, size(pfull_field_array, 2))

            dummy_field = fill(FT(1.0), space)
            function preprocess!(field)
                field_array = Fields.field2array(field)
                field_array[end, :] .= -10000.0
                field_array[begin, :] .= 10000.0
                field_array[5, :] .= 100.0
                field_array[6, :] .= 200.0
            end
            preprocess!(dummy_field)

            dest = interpolate_pressure!(dummy_field, pfull_intp)
            dest_field_array = Fields.field2array(dest)
            @test dest_field_array == repeat(
                [-10000.0, -10000.0, 150.0, 10000.0, 10000.0],
                1,
                size(pfull_field_array, 2),
            )

            # Test if in-place and non in-place behave the same
            # This also test that we do not need to call update! on pfull_intp if the
            # pressure field does not change
            dest2 = deepcopy(dest)
            dest2 .= 0.0
            dummy_field .= 1
            preprocess!(dummy_field)

            interpolate_pressure!!(dest2, dummy_field, pfull_intp)
            @test dest2 == dest
        end
    end
end

@testset "Face spaces" begin
    for FT in (Float32, Float64)
        extruded_space = ExtrudedCubedSphereSpace(FT;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            radius = 10,
            h_elem = 10,
            n_quad_points = 4,
            staggering = Grids.CellCenter(),
        )
        col_space = ColumnSpace(FT;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            staggering = Grids.CellCenter(),
        )
        for space in (extruded_space, col_space)
            space = ExtrudedCubedSphereSpace(FT;
                z_elem = 10,
                z_min = 0,
                z_max = 1,
                radius = 10,
                h_elem = 10,
                n_quad_points = 4,
                staggering = Grids.CellCenter(),
            )
            pfull_field = fill(FT(1.0), space)
            pfull_field_view = Fields.field2array(pfull_field)
            pfull_field_view .= collect(10:-1:1)
            pfull_levels = [0.0, 1.5, 6.0, 9.5, 12.0]

            pfull_intp = PressureInterpolator(pfull_field, pfull_levels)

            face_pfull_field_array =
                Fields.field2array(pfull_intp.scratch_face_pressure_field)
            # Test if pfull field defined on face space is extrapolated correctly
            # from pfull_field
            @test face_pfull_field_array == repeat(
                [1.0, range(1.5, 9.5, step = 1.0)..., 10.0],
                1,
                size(face_pfull_field_array, 2),
            )

            face_space = Spaces.face_space(space)
            dummy_field = fill(FT(1.0), face_space)
            function preprocess!(field)
                field_array = Fields.field2array(field)
                field_array[1, :] .= 100.0 # corresponds to pressure level 10.0
                field_array[2, :] .= 8.0 # corresponds to pressure level 9.5
                field_array[5, :] .= 5.0 # corresponds to pressure level 6.5
                field_array[6, :] .= 4.0 # corresponds to pressure level 5.5
                field_array[10, :] .= 2.0 # corresponds to pressure level 1.5
                field_array[11, :] .= -100.0 # corresponds to pressure level 1.0
            end
            preprocess!(dummy_field)

            dest = interpolate_pressure!(dummy_field, pfull_intp)
            dest_field_array = Fields.field2array(dest)
            @test dest_field_array == repeat(
                [-100.0, 2.0, 4.5, 8.0, 100.0],
                1,
                size(pfull_field_view, 2),
            )

            dest2 = deepcopy(dest)
            dest2 .= 0.0
            dummy_field .= 1.0
            preprocess!(dummy_field)

            interpolate_pressure!!(dest2, dummy_field, pfull_intp)
            @test dest2 == dest
        end
    end
end

@testset "Non monotonic pressure and z relationship" begin
    for FT in (Float32, Float64)
        extruded_space = ExtrudedCubedSphereSpace(FT;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            radius = 10,
            h_elem = 10,
            n_quad_points = 4,
            staggering = Grids.CellCenter(),
        )
        col_space = ColumnSpace(FT;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            staggering = Grids.CellCenter(),
        )
        for space in (extruded_space, col_space)
            pfull_field = fill(FT(1.0), space)
            pfull_field_view = Fields.field2array(pfull_field)
            pfull_field_view .= collect(10:-1:1)
            pfull_levels = [6.0, 7.0, 9.0]

            pfull_intp = PressureInterpolator(pfull_field, pfull_levels)

            # Test implementation detail of interpolation
            # Since aggregate with max is applied to the pressure field, we expect
            # repeated values for a column when the pressures are not monotonic
            pfull_array = Fields.field2array(Remapping.pfull_field(pfull_intp))
            reverse!(pfull_array, dims = 1)
            pfull_array[3, :] .= -100.0
            update!(pfull_intp)

            dummy_field = fill(FT(1.0), space)
            dummy_field_view = Fields.field2array(dummy_field)
            dummy_field_view[3, :] .= -1000.0
            dummy_field_view[4, :] .= 1000.0

            dest = interpolate_pressure!(dummy_field, pfull_intp)
            dest_view = Fields.field2array(dest)
            @test all(isone, dest_view[[1, 3], :])
            @test all(x -> x == 1000.0, dest_view[2, :])
        end
    end
end
