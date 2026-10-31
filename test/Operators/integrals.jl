using Test
using JET
import Random
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: DataLayouts, Spaces, Fields, Operators
import ClimaCore.Operators:
    column_integral_definite!,
    column_integral_indefinite!,
    column_reduce!,
    column_accumulate!

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

# CUDA is only needed to prune JET reports raised from CUDA frames, and it is
# absent from the CPU-only test environment that `Pkg.test` builds. Reach it
# through the ClimaComms extension (which `@import_required_backends` loads
# only on a CUDA device) rather than importing it outright, and prune nothing
# when there is no GPU. This mirrors MatrixFields/matrix_field_test_utils.jl.
const USING_CUDA = ClimaComms.device() isa ClimaComms.CUDADevice
cuda_module(ext) = USING_CUDA ? ext.CUDA : ext
const CUDA_MOD = cuda_module(Base.get_extension(ClimaComms, :ClimaCommsCUDAExt))
const CLIMACORE_CUDA_MOD = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
const CUDA_FRAMES =
    USING_CUDA ?
    (AnyFrameModule(CUDA_MOD), AnyFrameModule(CLIMACORE_CUDA_MOD)) : ()

are_boundschecks_forced = Base.JLOptions().check_bounds == 1
center_to_face_space(center_space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(center_space)
center_to_face_space(center_space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

test_allocs(allocs) =
    if ClimaComms.device() isa ClimaComms.AbstractCPUDevice
        # TODO: Some of these operations have a few unelided views from
        # getproperty (48 bytes each); whether the compiler elides them
        # depends on how much of its inference budget is used up.
        TU.allocation_checks_meaningful() && @test allocs ≤ 128
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

    # Each component of a NamedTuple or Tuple is integrated like a scalar on CPUs, up to
    # the reassociation of @simd loops (GPU kernels of different structure can also fuse
    # multiplications into additions differently).
    if ClimaComms.device(center_space) isa ClimaComms.AbstractCPUDevice
        ∫z = similar(z_top)
        column_integral_definite!(∫z, ᶜz)
        rtol = 8 * eps(Spaces.undertype(center_space)) # 3x the worst difference for 10 values
        @test all(isapprox.(parent(∫z), parent(∫u_test.powers.:1); rtol))
    end

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

    # The same integral inside foreach_column, into a face temporary of each column.
    ᶠ∫u_column_test = similar(ᶠ∫u_test)
    DataLayouts.foreach_column(ᶜu, ᶠ∫u_column_test) do ᶜu_column, ᶠ∫u_column
        ᶠ∫u_temp = similar(ᶜu_column, axes(ᶠ∫u_column))
        column_integral_indefinite!(ᶠ∫u_temp, ᶜu_column)
        @. ᶠ∫u_column = ᶠ∫u_temp
    end
    @test parent(ᶠ∫u_column_test) ≈ parent(ᶠ∫u_test)
end

function test_column_integral_indefinite_fn!(center_space)
    face_space = center_to_face_space(center_space)
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

        # The same integral inside foreach_column (for one function, to limit compilation).
        i == 1 || continue
        ᶠ∫u_column_test = similar(ᶠ∫u_test)
        DataLayouts.foreach_column(ᶠ∫u_column_test) do ᶠ∫u_column
            column_integral_indefinite!(fn, ᶠ∫u_column)
        end
        @test parent(ᶠ∫u_column_test) ≈ parent(ᶠ∫u_test)
    end
end

# Levels of temporaries inside foreach_column, which GPUs can store in the registers of the
# threads that own their points, are read by every thread of a column.
function test_levels_of_column_temporaries!(center_space)
    ᶜz = Fields.coordinate_field(center_space).z
    ᶠinterp = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜoutput = similar(ᶜz)
    DataLayouts.foreach_column(ᶜz, ᶜoutput) do ᶜz_column, ᶜoutput_column
        ᶜtemp = @. sin(ᶜz_column) + 1
        ᶠtemp = @. ᶠinterp(ᶜtemp)
        ᶜbottom = Fields.level(ᶜtemp, 1)
        ᶠtop = Fields.level(ᶠtemp, Operators.right_idx(axes(ᶠtemp)))
        # A point temporary derived from levels, which every thread reads.
        ᶜz_bottom = Fields.level(ᶜz_column, 1)
        ᶜscale = @. ᶜbottom + ᶜz_bottom
        @. ᶜoutput_column = ᶜtemp * ᶜscale + ᶠtop
    end
    # The top face of an extrapolated interpolation has the value at the top center. The
    # array is not named ᶜtemp, so the loop body does not capture a boxed variable, which
    # GPU kernels cannot take as an argument.
    temp_array = Array(parent(@. sin(ᶜz) + 1))
    z_bottom_array = selectdim(Array(parent(ᶜz)), 1, 1:1)
    Nv = size(temp_array, 1)
    reference =
        temp_array .* (selectdim(temp_array, 1, 1:1) .+ z_bottom_array) .+
        selectdim(temp_array, 1, Nv:Nv)
    @test Array(parent(ᶜoutput)) ≈ reference
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

    for (input, reverse) in (
        (ᶜwhole_number, false),
        (ᶠwhole_number, false),
        (ᶜwhole_number_reverse, true),
        (ᶠwhole_number_reverse, true),
    )
        final_input_idx =
            reverse ? Operators.left_idx(axes(input)) : Operators.right_idx(axes(input))
        final_input_level = Fields.level(input, final_input_idx)
        output = similar(final_input_level)
        reference_output = motzkin_number.(final_input_level)

        set_output! = () -> column_reduce!(f, output, input; init, transform, reverse)
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

# Lazy inputs with pointwise and finite difference operators (including linear
# operators, which broadcasts apply as band matrix products) give the same
# reductions and accumulations as their materialized values.
function test_column_reduce_and_accumulate_lazy!(center_space)
    FT = Spaces.undertype(center_space)
    face_space = Spaces.face_space(center_space)
    ᶜa = Fields.coordinate_field(center_space).z .+ 1
    ᶜb = sin.(Fields.coordinate_field(center_space).z)
    ᶠc = cos.(Fields.coordinate_field(face_space).z)
    ᶜinterp = Operators.InterpolateF2C()
    ᶠinterp = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜdiv = Operators.DivergenceF2C()
    ᶠgrad = Operators.GradientC2F(;
        bottom = Operators.SetValue(zero(FT)),
        top = Operators.SetValue(zero(FT)),
    )
    lazy(args...) = Base.Broadcast.broadcasted(args...)
    inputs = (
        lazy(*, ᶜa, ᶜb),
        lazy(*, ᶜa, lazy(ᶜinterp, ᶠc)),
        lazy(ᶜdiv, lazy(ClimaCore.Geometry.WVector, lazy(*, ᶠc, lazy(ᶠinterp, ᶜb)))),
        lazy(+, ᶜb, lazy(ᶜdiv, lazy(ClimaCore.Geometry.WVector, lazy(ᶠinterp, ᶜa)))),
        lazy(ᶜdiv, lazy(ᶠgrad, ᶜb)), # reads the metric terms of every column
    )
    init = one(FT) # a concrete value, so that the closures below are inferable
    for input in inputs
        materialized_input = Base.Broadcast.materialize(input)
        ᶜoutput = similar(ᶜa)
        ᶜreference = similar(ᶜa)
        ᶠoutput = similar(ᶠc)
        ᶠreference = similar(ᶠc)
        level_output = Fields.level(similar(ᶜa), 1)
        level_reference = Fields.level(similar(ᶜa), 1)
        for (reduce!, output, reference) in (
            ((out, in) -> column_accumulate!(+, out, in), ᶜoutput, ᶜreference),
            (
                (out, in) -> column_accumulate!(+, out, in; init, reverse = true),
                ᶠoutput,
                ᶠreference,
            ),
            ((out, in) -> column_reduce!(max, out, in), level_output, level_reference),
            (
                (out, in) -> column_integral_definite!(out, in),
                level_output,
                level_reference,
            ),
            ((out, in) -> column_integral_indefinite!(out, in), ᶠoutput, ᶠreference),
        )
            reduce!(reference, materialized_input)
            reduce!(output, input)
            # Equal up to roundoff: GPU kernels of different structure fuse multiplications
            # differently. Roundoff scales with the reduction of the input's magnitude, which
            # matters for reductions that cancel to 0 (e.g., the integral of a divergence).
            scale = similar(output)
            reduce!(scale, Base.Broadcast.broadcasted(abs, materialized_input))
            @test parent(output) ≈ parent(reference) rtol = 100 * eps(FT) atol =
                100 * eps(FT) * maximum(abs, parent(scale))
            @test_opt ignored_modules = CUDA_FRAMES reduce!(output, input)
            test_allocs(@allocated reduce!(output, input))
        end
    end
end

# Lazy inputs of NamedTuples (and Tuples, whose buffers have the same structure) are read
# as AutoBroadcasters, which support math functions like -. GPUs copy such inputs into
# buffers for operators that are not associative, and the buffers must also be read as
# AutoBroadcasters.
function test_column_reduce_and_accumulate_lazy_tuples!(center_space)
    FT = Spaces.undertype(center_space)
    face_space = Spaces.face_space(center_space)
    ᶜinterp = Operators.InterpolateF2C()
    lazy(args...) = Base.Broadcast.broadcasted(args...)
    sub(x, y) = x - y
    reference_sub(x, y) = map(-, x, y) # also defined without AutoBroadcasters
    value(z) = (; a = z, b = sin(z))
    init = (; a = one(FT), b = zero(FT))
    ᶜvalues = value.(Fields.coordinate_field(center_space).z)
    ᶠvalues = value.(Fields.coordinate_field(face_space).z)
    level_output = similar(Fields.level(ᶠvalues, Fields.half))
    for (f!, output, input) in (
        (
            (f, out, in) -> column_reduce!(f, out, in),
            level_output,
            lazy(+, ᶜvalues, ᶜvalues),
        ),
        (
            (f, out, in) -> column_reduce!(f, out, in; init, reverse = true),
            level_output,
            lazy(+, ᶠvalues, ᶠvalues),
        ),
        (
            (f, out, in) -> column_accumulate!(f, out, in; init),
            similar(ᶠvalues),
            lazy(+, ᶜvalues, ᶜvalues),
        ),
        (
            (f, out, in) -> column_accumulate!(f, out, in; reverse = true),
            similar(ᶜvalues),
            lazy(ᶜinterp, ᶠvalues),
        ),
    )
        reference = similar(output)
        f!(reference_sub, reference, Base.Broadcast.materialize(input))
        f!(sub, output, input)
        @test parent(output) ≈ parent(reference)
    end
end

# A column field (e.g., Spaces.column(field, i, j, h)) in an operator expression with fields
# on the whole space is read in every column, like a copy of it on the whole space.
function test_column_reduce_and_accumulate_column_fields!(center_space)
    FT = Spaces.undertype(center_space)
    ᶜa = Fields.coordinate_field(center_space).z .+ 1
    ᶠb = cos.(Fields.coordinate_field(Spaces.face_space(center_space)).z)
    ᶠinterp = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜdiv = Operators.DivergenceF2C()
    ᶠupwind = Operators.UpwindBiasedProductC2F()
    ᶠvanleer = Operators.LinVanLeerC2F(; constraint = Operators.MonotoneLocalExtrema())
    lazy(args...) = Base.Broadcast.broadcasted(args...)
    ᶠw = lazy(ClimaCore.Geometry.WVector, lazy(ᶠinterp, ᶜa))
    inputs(ᶜc, ᶠc) = (
        lazy(ᶜdiv, lazy(ᶠvanleer, lazy(*, ᶠw, ᶠc), ᶜa, FT(0.1))),
        lazy(*, ᶜa, lazy(ᶜdiv, lazy(ᶠupwind, ᶠw, ᶜc))),
    )
    ᶜcolumn = Spaces.column(ᶜa .^ 2, 2, 2, 3)
    ᶠcolumn = Spaces.column(ᶠb, 2, 2, 3)
    (ᶜcopy, ᶠcopy) = (similar(ᶜa), similar(ᶠb))
    @. ᶜcopy = ᶜcolumn
    @. ᶠcopy = ᶠcolumn
    level_output = Fields.level(similar(ᶜa), 1)
    level_reference = Fields.level(similar(ᶜa), 1)
    for (input, reference_input) in zip(inputs(ᶜcolumn, ᶠcolumn), inputs(ᶜcopy, ᶠcopy))
        for (reduce!, output, reference) in (
            ((out, in) -> column_reduce!(+, out, in), level_output, level_reference),
            (
                (out, in) -> column_accumulate!(+, out, in; reverse = true),
                similar(ᶜa),
                similar(ᶜa),
            ),
            (column_integral_definite!, level_output, level_reference),
            (column_integral_indefinite!, similar(ᶠb), similar(ᶠb)),
        )
            reduce!(reference, reference_input)
            reduce!(output, input)
            @test parent(output) ≈ parent(reference) rtol = 100 * eps(FT)
            @test_opt ignored_modules = CUDA_FRAMES reduce!(output, input)
        end
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

# On GPUs, columns with more levels than a warp are divided among several warps, and columns
# with more levels than a block's threads give each thread several levels (in rounds).
function test_column_reduce_and_accumulate_deep!(center_space)
    face_space = center_to_face_space(center_space)
    FT = Spaces.undertype(center_space)
    ᶜinput = ones(center_space)
    ᶠinput = ones(face_space)
    parent(ᶜinput) .= rand.(FT)
    parent(ᶠinput) .= rand.(FT)
    ᶜoutput = similar(ᶜinput)
    ᶠoutput = similar(ᶠinput)
    surface_output = similar(Fields.level(ᶠinput, Fields.half))
    ᶜarray = Array(parent(ᶜinput))
    ᶠarray = Array(parent(ᶠinput))
    reverse_cumsum(array) = reverse(cumsum(reverse(array; dims = 1); dims = 1); dims = 1)
    init = FT(2)
    # Staggered offsets and operators that cannot be reordered are tested on fewer levels.
    for (input, output, reverse, reference) in (
        (ᶜinput, ᶜoutput, false, init .+ cumsum(ᶜarray; dims = 1)),
        (ᶜinput, ᶜoutput, true, init .+ reverse_cumsum(ᶜarray)),
        (
            ᶜinput,
            ᶠoutput,
            false,
            init .+ cat(zero(ᶜarray[1:1, :, :, :, :]), cumsum(ᶜarray; dims = 1); dims = 1),
        ),
    )
        column_accumulate!(+, output, input; init, reverse)
        @test Array(parent(output)) ≈ reference
    end
    column_accumulate!(max, ᶠoutput, ᶠinput; reverse = true)
    @test Array(parent(ᶠoutput)) ==
          reverse(accumulate(max, reverse(ᶠarray; dims = 1); dims = 1); dims = 1)
    # An in-place accumulation with an operator that one thread applies in order, reading
    # every level before overwriting it (DataLayouts tests the same accumulation forward
    # with init).
    ᶠinplace = copy(ᶠinput)
    column_accumulate!((a, b) -> a - b, ᶠinplace, ᶠinplace; reverse = true)
    @test Array(parent(ᶠinplace)) ==
          reverse(accumulate(-, reverse(ᶠarray; dims = 1); dims = 1); dims = 1)
    column_reduce!(+, surface_output, ᶜinput; init)
    @test Array(parent(surface_output)) ≈ init .+ sum(ᶜarray; dims = 1)
end

# Accumulations with operators that are applied in order only share their final value
# between threads, so the values at all levels can exceed a GPU block's shared memory.
function test_column_accumulate_large_values!(center_space)
    FT = Spaces.undertype(center_space)
    ᶜinput = ones(center_space)
    parent(ᶜinput) .= rand.(FT)
    ᶜoutput = similar(ᶜinput)
    init = (zero(FT), ntuple(_ -> FT(NaN), Val(100)))
    lazy_input = Base.Broadcast.broadcasted(+, ᶜinput, ᶜinput)
    column_accumulate!(ᶜoutput, lazy_input; init, transform = first) do (sum, nans), value
        (sum + value, nans)
    end
    @test Array(parent(ᶜoutput)) ≈ cumsum(2 .* Array(parent(ᶜinput)); dims = 1)
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
            test_column_reduce_and_accumulate_lazy!(space)
        end
        # 150 levels take two rounds of several warps on GPUs; one precision is enough.
        # Lazy tuples are also tested on one space, since their buffers ignore its type.
        if FT == Float64
            deep_space =
                TU.CenterExtrudedFiniteDifferenceSpace(FT; context, zelem = 150, helem = 2)
            test_column_reduce_and_accumulate_deep!(deep_space)
            test_column_accumulate_large_values!(deep_space)
            # Columns within a warp and longer than a warp on GPUs.
            for zelem in (10, 63)
                test_levels_of_column_temporaries!(
                    TU.CenterExtrudedFiniteDifferenceSpace(FT; context, zelem, helem = 2),
                )
            end
            test_column_reduce_and_accumulate_lazy_tuples!(
                TU.CenterExtrudedFiniteDifferenceSpace(FT; context),
            )
        end
        test_column_reduce_and_accumulate_column_fields!(
            TU.CenterExtrudedFiniteDifferenceSpace(FT; context, topography = true),
        )
        # Columns of these spaces are indexed by (i, h) rather than (i, j, h).
        for space in (
            TU.MultiColumnSpace(FT; context),
            Spaces.ExtrudedFiniteDifferenceSpace(
                TU.SpectralElementSpace1D(FT; context),
                TU.ColumnCenterFiniteDifferenceSpace(FT; context),
            ),
        )
            test_column_reduce_and_accumulate_lazy!(space)
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
