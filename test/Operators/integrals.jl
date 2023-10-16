using Test
using JET
import CUDA
import ClimaComms
import ClimaCore
import ClimaCore: Spaces, Fields, Operators
import ClimaCore.RecursiveApply: rmax
import ClimaCore.Operators:
    column_integral_definite!, column_integral_indefinite!, column_mapreduce!

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

center_to_face_space(center_space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(center_space)
center_to_face_space(center_space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

function test_column_definite_integral!(center_space)
    face_space = center_to_face_space(center_space)
    ᶜz = Fields.coordinate_field(center_space).z
    ᶠz = Fields.coordinate_field(face_space).z
    z_top = Fields.level(ᶠz, Operators.right_idx(face_space))
    ᶜu = map(z -> (; one = one(z), powers = (z, z^2, z^3)), ᶜz)
    ∫u_ref = map(z -> (; one = z, powers = (z^2 / 2, z^3 / 3, z^4 / 4)), z_top)
    ∫u_test = similar(∫u_ref)

    column_integral_definite!(∫u_test, ᶜu)
    ref_array = parent(∫u_ref)
    test_array = parent(∫u_test)
    max_relative_error = maximum(@. abs((ref_array - test_array) / ref_array))
    @test max_relative_error <= 0.006 # Less than 0.6% error.

    cuda = (AnyFrameModule(CUDA),)
    # @test_opt ignored_modules = cuda column_integral_definite!(∫u_test, ᶜu)

    ClimaComms.device() isa ClimaComms.CUDADevice ||
        @test (@allocated column_integral_definite!(∫u_test, ᶜu)) == 0
end

function test_column_integral_indefinite!(center_space)
    face_space = center_to_face_space(center_space)
    ᶜz = Fields.coordinate_field(center_space).z
    ᶠz = Fields.coordinate_field(face_space).z
    ᶜu = map(z -> (; one = one(z), powers = (z, z^2, z^3)), ᶜz)
    ᶠ∫u_ref = map(z -> (; one = z, powers = (z^2 / 2, z^3 / 3, z^4 / 4)), ᶠz)
    ᶠ∫u_test = similar(ᶠ∫u_ref)

    column_integral_indefinite!(ᶠ∫u_test, ᶜu)
    ref_array = parent(Fields.level(ᶠ∫u_ref, Operators.right_idx(face_space)))
    test_array = parent(Fields.level(ᶠ∫u_test, Operators.right_idx(face_space)))
    max_relative_error = maximum(@. abs((ref_array - test_array) / ref_array))
    @test max_relative_error <= 0.006 # Less than 0.6% error at the top level.

    cuda = (AnyFrameModule(CUDA),)
    # @test_opt ignored_modules = cuda column_integral_indefinite!(ᶠ∫u_test, ᶜu)

    ClimaComms.device() isa ClimaComms.CUDADevice ||
        @test (@allocated column_integral_indefinite!(ᶠ∫u_test, ᶜu)) == 0
end

function test_column_mapreduce!(space, alloc_lim)
    z_field = Fields.coordinate_field(space).z
    z_top_field = Fields.level(z_field, Operators.right_idx(space))
    sin_field = @. sin(pi * z_field / z_top_field)
    square_and_sin(z, sin_value) = (; square = z^2, sin = sin_value)
    reduced_field_ref = map(z -> (; square = z^2, sin = one(z)), z_top_field)
    reduced_field_test = similar(reduced_field_ref)
    args = (square_and_sin, rmax, reduced_field_test, z_field, sin_field)

    column_mapreduce!(args...)
    ref_array = parent(reduced_field_ref)
    test_array = parent(reduced_field_test)
    max_relative_error = maximum(@. abs((ref_array - test_array) / ref_array))
    @test max_relative_error <= 0.004 # Less than 0.4% error.

    cuda = (AnyFrameModule(CUDA),)
    # @test_opt ignored_modules = cuda column_mapreduce!(args...)

    # TODO: column_mapreduce! currently allocates memory
    ClimaComms.device() isa ClimaComms.CUDADevice ||
        @test (@allocated column_mapreduce!(args...)) ≤ alloc_lim
    ClimaComms.device() isa ClimaComms.CUDADevice ||
        @test_broken (@allocated column_mapreduce!(args...)) == 0
end

@testset "Integral operations unit tests" begin
    lim = Dict()
    lim[(1, Float32)] = 1808
    lim[(2, Float32)] = 1920
    lim[(3, Float32)] = 4399104
    lim[(4, Float32)] = 4571136

    lim[(1, Float64)] = 2512
    lim[(2, Float64)] = 2688
    lim[(3, Float64)] = 5455872
    lim[(4, Float64)] = 5726208
    for FT in (Float32, Float64)
        for center_space in (
            TU.ColumnCenterFiniteDifferenceSpace(FT),
            TU.CenterExtrudedFiniteDifferenceSpace(FT),
        )
            test_column_definite_integral!(center_space)
            test_column_integral_indefinite!(center_space)
        end

        for (i, space) in enumerate((
            TU.ColumnCenterFiniteDifferenceSpace(FT),
            TU.ColumnFaceFiniteDifferenceSpace(FT),
            TU.CenterExtrudedFiniteDifferenceSpace(FT),
            TU.FaceExtrudedFiniteDifferenceSpace(FT),
        ))
            test_column_mapreduce!(space, lim[(i, FT)])
        end
    end
end
