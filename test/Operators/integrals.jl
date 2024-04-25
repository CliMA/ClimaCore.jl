using Test
using JET
import CUDA
CUDA.allowscalar(false)
import ClimaComms
if pkgversion(ClimaComms) >= v"0.6"
    ClimaComms.@import_required_backends
end
import ClimaCore
import ClimaCore: Spaces, Fields, Operators
import ClimaCore.RecursiveApply: rmax
import ClimaCore.Operators:
    column_integral_definite!, column_integral_indefinite!, column_mapreduce!

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

are_boundschecks_forced = Base.JLOptions().check_bounds == 1
center_to_face_space(center_space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(center_space)
center_to_face_space(center_space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

test_allocs(lim::Nothing, allocs, caller) = nothing
function test_allocs(lim::Int, allocs, caller)
    @assert allocs ≥ 0
    if lim == 0
        @test allocs == 0
    else
        @test allocs ≤ lim
        allocs < lim && @show allocs, lim, caller
        @test_broken allocs == 0 # notify when things improve
    end
end

function test_column_integral_definite!(center_space, alloc_lim)
    face_space = center_to_face_space(center_space)
    ᶜz = Fields.coordinate_field(center_space).z
    ᶠz = Fields.coordinate_field(face_space).z
    z_top = Fields.level(ᶠz, Operators.right_idx(face_space))
    ᶜu = map(z -> (; one = one(z), powers = (z, z^2, z^3)), ᶜz)
    CUDA.@allowscalar ∫u_ref =
        map(z -> (; one = z, powers = (z^2 / 2, z^3 / 3, z^4 / 4)), z_top)
    ∫u_test = similar(∫u_ref)

    column_integral_definite!(∫u_test, ᶜu)
    ref_array = parent(∫u_ref)
    test_array = parent(∫u_test)
    max_relative_error = maximum(@. abs((ref_array - test_array) / ref_array))
    @test max_relative_error <= 0.006 # Less than 0.6% error.

    cuda = (AnyFrameModule(CUDA),)
    @test_opt ignored_modules = cuda column_integral_definite!(∫u_test, ᶜu)

    allocs = @allocated column_integral_definite!(∫u_test, ᶜu)
    test_allocs(alloc_lim, allocs, "test_column_integral_definite")
end

function test_column_integral_indefinite!(center_space, alloc_lim)
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
    @test_opt ignored_modules = cuda column_integral_indefinite!(ᶠ∫u_test, ᶜu)

    allocs = @allocated column_integral_indefinite!(ᶠ∫u_test, ᶜu)
    test_allocs(alloc_lim, allocs, "test_column_integral_indefinite!")
end

function test_column_integral_indefinite_fn!(center_space, alloc_lim)
    face_space = center_to_face_space(center_space)
    ᶜz = Fields.coordinate_field(center_space).z
    ᶠz = Fields.coordinate_field(face_space).z
    FT = Spaces.undertype(center_space)

    ᶜu = Dict()
    ᶠ∫u_ref = Dict()
    ᶠ∫u_test = Dict()


    # ᶜu = map(z -> (; one = one(z), powers = (z, z^2, z^3)), ᶜz)
    # ᶠ∫u_ref = map(z -> (; one = z, powers = (z^2 / 2, z^3 / 3, z^4 / 4)), ᶠz)
    # ᶠ∫u_test = similar(ᶠ∫u_ref)

    for (i, fn) in enumerate(((ϕ, z) -> z, (ϕ, z) -> z^2, (ϕ, z) -> z^3))
        ᶜu = ᶜz .^ i
        ᶠ∫u_ref = ᶠz .^ (i + 1) ./ (i + 1)
        ᶠ∫u_test = similar(ᶠ∫u_ref)

        column_integral_indefinite!(fn, ᶠ∫u_test)
        ref_array =
            parent(Fields.level(ᶠ∫u_ref, Operators.right_idx(face_space)))
        test_array =
            parent(Fields.level(ᶠ∫u_test, Operators.right_idx(face_space)))
        max_relative_error =
            maximum(@. abs((ref_array - test_array) / ref_array))
        @test max_relative_error <= 0.006 # Less than 0.6% error at the top level.

        # @test_opt column_integral_indefinite!(fn, ᶠ∫u_test)

        allocs = @allocated column_integral_indefinite!(fn, ᶠ∫u_test)
        test_allocs(alloc_lim, allocs, "test_column_integral_indefinite_fn!")
    end

end

function test_column_mapreduce!(space, alloc_lim)
    z_field = Fields.coordinate_field(space).z
    z_top_field = Fields.level(z_field, Operators.right_idx(space))
    sin_field = @. sin(pi * z_field / z_top_field)
    square_and_sin(z, sin_value) = (; square = z^2, sin = sin_value)
    CUDA.@allowscalar reduced_field_ref =
        map(z -> (; square = z^2, sin = one(z)), z_top_field)
    reduced_field_test = similar(reduced_field_ref)
    args = (square_and_sin, rmax, reduced_field_test, z_field, sin_field)

    column_mapreduce!(args...)
    ref_array = parent(reduced_field_ref)
    test_array = parent(reduced_field_test)
    max_relative_error = maximum(@. abs((ref_array - test_array) / ref_array))
    @test max_relative_error <= 0.004 # Less than 0.4% error.

    cuda = (AnyFrameModule(CUDA),)
    if alloc_lim == 0
        @test_opt ignored_modules = cuda column_mapreduce!(args...)
    end
    allocs = @allocated column_mapreduce!(args...)
    test_allocs(alloc_lim, allocs, "test_column_mapreduce!")
end

@testset "Integral operations unit tests" begin
    # device = ClimaComms.CPUSingleThreaded();
    device = ClimaComms.device()
    context = ClimaComms.SingletonCommsContext(device)
    broken = device isa ClimaComms.CUDADevice
    if device isa ClimaComms.CPUSingleThreaded
        i_lim = Dict()
        i_lim[(1, Float32)] = 0
        i_lim[(2, Float32)] = 0
        i_lim[(3, Float32)] = 0
        i_lim[(4, Float32)] = 0

        i_lim[(1, Float64)] = 0
        i_lim[(2, Float64)] = 0
        i_lim[(3, Float64)] = 0
        i_lim[(4, Float64)] = 0

        lim = Dict()
        lim[(1, Float32)] = 2768
        lim[(2, Float32)] = 2960
        lim[(3, Float32)] = 8183808
        lim[(4, Float32)] = 8650752

        lim[(1, Float64)] = 3648
        lim[(2, Float64)] = 3920
        lim[(3, Float64)] = 9510912
        lim[(4, Float64)] = 10100736
    else
        i_lim = Dict()
        i_lim[(1, Float32)] = 1728
        i_lim[(2, Float32)] = 4720
        i_lim[(3, Float32)] = 2304
        i_lim[(4, Float32)] = 8176

        i_lim[(1, Float64)] = 1936
        i_lim[(2, Float64)] = 4720
        i_lim[(3, Float64)] = 2544
        i_lim[(4, Float64)] = 8176

        lim = Dict()
        lim[(1, Float32)] = 34144
        lim[(2, Float32)] = 37600
        lim[(3, Float32)] = 4399104
        lim[(4, Float32)] = 4571136

        lim[(1, Float64)] = 35024
        lim[(2, Float64)] = 38560
        lim[(3, Float64)] = 5455872
        lim[(4, Float64)] = 5726208
    end
    if are_boundschecks_forced
        lim = Dict(k => nothing for k in keys(lim))
        i_lim = Dict(k => nothing for k in keys(lim))
    end

    for FT in (Float32, Float64)
        test_column_integral_definite!(
            TU.ColumnCenterFiniteDifferenceSpace(FT; context),
            i_lim[(1, FT)],
        )
        test_column_integral_definite!(
            TU.CenterExtrudedFiniteDifferenceSpace(FT; context),
            i_lim[(2, FT)],
        )
        test_column_integral_indefinite!(
            TU.ColumnCenterFiniteDifferenceSpace(FT; context),
            i_lim[(3, FT)],
        )
        test_column_integral_indefinite!(
            TU.CenterExtrudedFiniteDifferenceSpace(FT; context),
            i_lim[(4, FT)],
        )
        broken || test_column_integral_indefinite_fn!(
            TU.ColumnCenterFiniteDifferenceSpace(FT; context),
            i_lim[(3, FT)],
        )
        broken || test_column_integral_indefinite_fn!(
            TU.CenterExtrudedFiniteDifferenceSpace(FT; context),
            i_lim[(4, FT)],
        )

        broken || test_column_mapreduce!(
            TU.ColumnCenterFiniteDifferenceSpace(FT; context),
            lim[(1, FT)],
        )
        broken || test_column_mapreduce!(
            TU.ColumnFaceFiniteDifferenceSpace(FT; context),
            lim[(2, FT)],
        )
        broken || test_column_mapreduce!(
            TU.CenterExtrudedFiniteDifferenceSpace(FT; context),
            lim[(3, FT)],
        )
        broken || test_column_mapreduce!(
            TU.FaceExtrudedFiniteDifferenceSpace(FT; context),
            lim[(4, FT)],
        )
    end
end
