using Test
using JET
import Random
import CUDA # explicitly required due to JET
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Spaces, Fields, Operators
import ClimaCore.Operators:
    column_integral_definite!,
    column_integral_indefinite!,
    column_reduce!,
    column_accumulate!

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

const CLIMACORE_CUDA_MOD = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
const CUDA_FRAMES =
    (
        AnyFrameModule(CUDA),
        AnyFrameModule(CLIMACORE_CUDA_MOD),
    )

are_boundschecks_forced = Base.JLOptions().check_bounds == 1
center_to_face_space(center_space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(center_space)
center_to_face_space(center_space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

test_allocs(allocs) =
    if ClimaComms.device() isa ClimaComms.AbstractCPUDevice
        @test allocs == 0
    else
        @test allocs ≤ 39656 # GPU always has ~2 kB of non-deterministic allocs.
    end

function test_column_integral_definite!(center_space)
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


    @test_opt ignored_modules = CUDA_FRAMES column_integral_definite!(∫u_test, ᶜu)

    test_allocs(@allocated column_integral_definite!(∫u_test, ᶜu))
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

    @test_opt ignored_modules = CUDA_FRAMES column_integral_indefinite!(ᶠ∫u_test, ᶜu)

    test_allocs(@allocated column_integral_indefinite!(ᶠ∫u_test, ᶜu))
end

function test_column_integral_indefinite_fn!(center_space)
    face_space = center_to_face_space(center_space)
    ᶜz = Fields.coordinate_field(center_space).z
    ᶠz = Fields.coordinate_field(face_space).z
    for (i, fn) in enumerate(((ϕ, z) -> z, (ϕ, z) -> z^2, (ϕ, z) -> z^3))
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

        @test_opt ignored_modules = CUDA_FRAMES column_integral_indefinite!(
            fn,
            ᶠ∫u_test,
        )

        test_allocs(@allocated column_integral_indefinite!(fn, ᶠ∫u_test))
    end
end

function test_column_reduce_and_accumulate!(center_space)
    face_space = center_to_face_space(center_space)
    ᶜwhole_number = ones(center_space)
    ᶜwhole_number_reverse = ones(center_space)
    column_accumulate!(+, ᶜwhole_number, ᶜwhole_number) # 1:Nv per column
    column_accumulate!(+, ᶜwhole_number_reverse, ᶜwhole_number_reverse; reverse = true)
    ᶠwhole_number = ones(face_space)
    column_accumulate!(+, ᶠwhole_number, ᶠwhole_number) # 1:(Nv + 1) per column
    ᶠwhole_number_reverse = ones(face_space)
    column_accumulate!(+, ᶠwhole_number_reverse, ᶠwhole_number_reverse; reverse = true) # 1:(Nv + 1) per column

    safe_binomial(n, k) = binomial(Int32(n), Int32(k)) # GPU-compatible binomial

    # https://en.wikipedia.org/wiki/Motzkin_number#Properties
    motzkin_number(n) =
        sum(0:(n / 2); init = zero(n)) do k
            safe_binomial(n, 2 * k) * safe_binomial(2 * k, k) / (k + 1)
        end
    recursive_motzkin_number((mₙ₋₁, mₙ₋₂), n) =
        ((2 * n + 1) * mₙ₋₁ + (3 * n - 3) * mₙ₋₂) / (n + 2)

    # On step n of the reduction/accumulation, update (mₙ₋₁, mₙ₋₂) to (mₙ, mₙ₋₁).
    f((mₙ₋₁, mₙ₋₂), n) = (recursive_motzkin_number((mₙ₋₁, mₙ₋₂), n), mₙ₋₁)
    init = (1, 0) # m₀ = 1, m₋₁ = 0 (m₋₁ can be set to any finite value)
    transform = first # Get mₙ from each (mₙ, mₙ₋₁) pair before saving to output.

    for input in (ᶜwhole_number, ᶠwhole_number)
        last_input_level = Fields.level(input, Operators.right_idx(axes(input)))
        output = similar(last_input_level)
        reference_output = motzkin_number.(last_input_level)

        set_output! = () -> column_reduce!(f, output, input; init, transform)
        set_output!()
        @test output == reference_output
        @test_opt ignored_modules = CUDA_FRAMES set_output!()
        test_allocs(@allocated set_output!())
    end

    ᶜoutput = similar(ᶜwhole_number)
    ᶠoutput = similar(ᶠwhole_number)
    for (input, output, reference_output, reverse) in (
        (ᶜwhole_number, ᶜoutput, motzkin_number.(ᶜwhole_number), false),
        (ᶠwhole_number, ᶠoutput, motzkin_number.(ᶠwhole_number), false),
        (ᶠwhole_number, ᶜoutput, motzkin_number.(ᶜwhole_number .+ 1), false),
        (ᶜwhole_number, ᶠoutput, motzkin_number.(ᶠwhole_number .- 1), false),
        (ᶜwhole_number_reverse, ᶜoutput, motzkin_number.(ᶜwhole_number_reverse), true),
        (ᶠwhole_number_reverse, ᶠoutput, motzkin_number.(ᶠwhole_number_reverse), true),
        (ᶠwhole_number_reverse, ᶜoutput, motzkin_number.(ᶜwhole_number_reverse .+ 1), true),
        (ᶜwhole_number_reverse, ᶠoutput, motzkin_number.(ᶠwhole_number_reverse .- 1), true),
    )

        set_output! =
            () -> column_accumulate!(f, output, input; init, transform, reverse)
        set_output!()
        @test output == reference_output
        @test_opt ignored_modules = CUDA_FRAMES set_output!()
        test_allocs(@allocated set_output!())
    end


end

function test_fubinis_theorem(space)
    FT = Spaces.undertype(space)
    center_field = zeros(space)
    face_field = zeros(center_to_face_space(space))
    surface_field = Fields.level(face_field, Fields.half)
    surface_Δz = Fields.Δz_field(surface_field) ./ 2

    Random.seed!(1) # ensures reproducibility
    parent(center_field) .= rand.(FT)
    volume_sum = sum(center_field)

    column_integral_definite!(surface_field, center_field)
    horizontal_sum_of_vertical_sum = sum(surface_field ./ surface_Δz)
    @test horizontal_sum_of_vertical_sum ≈ volume_sum rtol = 10 * eps(FT)
end

@testset "Integral operations unit tests" begin
    device = ClimaComms.device()
    context = ClimaComms.SingletonCommsContext(device)
    for FT in (Float32, Float64)
        for space in (
            TU.ColumnCenterFiniteDifferenceSpace(FT; context),
            TU.CenterExtrudedFiniteDifferenceSpace(FT; context),
        )
            test_column_integral_definite!(space)
            test_column_integral_indefinite!(space)
            test_column_integral_indefinite_fn!(space)
            test_column_reduce_and_accumulate!(space)
        end
    end
end

@testset "Volume integral consistency" begin
    device = ClimaComms.device()
    context = ClimaComms.SingletonCommsContext(device)
    for FT in (Float32, Float64)
        bools = (false, true)
        for topography in bools, deep in bools, autodiff_metric in bools
            space_kwargs = (; context, topography, deep, autodiff_metric)
            space = TU.CenterExtrudedFiniteDifferenceSpace(FT; space_kwargs...)
            test_fubinis_theorem(space)
        end
    end
end
