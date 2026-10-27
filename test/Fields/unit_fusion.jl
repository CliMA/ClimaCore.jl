# Fused slice-loop broadcasts, in their own slow-tier file: compiling the
# fused spectral machinery for every space is the largest native-code and
# coverage-instrumentation load in the unit tests, which the one-process
# GitHub Actions jobs cannot afford (they exclude slow tests); the Buildkite
# fields job runs this file uninstrumented.
using Test
using JET
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore.DataLayouts
import ClimaCore.DataLayouts:
    foreach_point, foreach_level, foreach_slab, foreach_column
import ClimaCore: Fields, Operators, Spaces, Geometry

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

# In-body temporaries route through DataLayouts.register_similar; slices whose
# sizes are not static (e.g. level slices of spaces with dynamic element counts)
# exercise its fallback to a plain similar allocation.
function test_fused_loop(foreach_slice, space, subspace)
    FT = Spaces.undertype(space)
    dest = Fields.Field(FT, space)
    src1 = Fields.Field(Tuple{FT, FT}, space)
    src2 = Fields.Field(Tuple{FT, FT}, subspace)
    parent(src1) .= rand.(FT)
    parent(src2) .= rand.(FT)

    function fused_loop!(dest, src1, src2)
        @. dest = 0
        temp1 = @. sin(src1.:1) + cos(src2.:1)
        @. dest += temp1 * temp1
        temp2 = @. sin(src1.:2) + cos(src2.:2)
        @. dest += temp2 * temp2
        @. dest /= 2
    end

    foreach_slice(fused_loop!, dest, src1, src2)
    @test dest ≈ @. sum((sin(src1) + cos(src2))^2) / 2

    CUDA_FRAMES = @isdefined(CUDA) ? (AnyFrameModule(CUDA),) : ()
    @test_opt ignored_modules = CUDA_FRAMES foreach_slice(fused_loop!, dest, src1, src2)
end

# As in test_fused_loop, but with spectral element operators in place of the
# pointwise functions. Every operator reads all of the points of a slab, so a
# fused slab loop is the smallest loop that can evaluate one. Later statements
# read earlier results both pointwise and through operators, and the in-body
# temporaries temp1 and temp2 live in per-thread registers (see
# register_similar), so temp1 stays live across temp2's spectral statement
# even though that statement's operator buffers have the same byte size as
# temp1's slab.
function test_fused_slab_loop(space, subspace)
    FT = Spaces.undertype(space)
    (div, wdiv) = (Operators.Divergence(), Operators.Divergence{Operators.WeakForm}())
    (grad, wgrad) = (Operators.Gradient(), Operators.Gradient{Operators.WeakForm}())
    curl = Operators.Curl()
    dest = Fields.Field(FT, space)
    src1 = Fields.Field(Tuple{FT, FT}, space)
    src2 = Fields.Field(Tuple{FT, FT}, subspace)
    parent(src1) .= rand.(FT)
    parent(src2) .= rand.(FT)

    function fused_loop!(dest, src1, src2)
        @. dest = 0
        temp1 = @. wdiv(grad(src1.:1)) + src2.:1
        @. dest += temp1 * temp1
        temp2 = @. div(wgrad(src1.:2)) - div(curl(Geometry.Covariant3Vector(src1.:2)))
        @. dest += (wdiv(grad(temp1)) + wdiv(grad(temp2)) + src2.:2)^2
        @. dest /= 2
    end

    foreach_slab(fused_loop!, dest, src1, src2)
    temp1_ref = @. wdiv(grad(src1.:1)) + src2.:1
    temp2_ref = @. div(wgrad(src1.:2)) - div(curl(Geometry.Covariant3Vector(src1.:2)))
    @test dest ≈ @. (
        temp1_ref^2 + (wdiv(grad(temp1_ref)) + wdiv(grad(temp2_ref)) + src2.:2)^2
    ) / 2

    # No @test_opt here: JET's OptAnalyzer crashes on any spectral broadcast
    # under Julia 1.11; re-enable once JET supports 1.11's interpreter.
end

# As in test_fused_slab_loop, but with finite difference operators in a fused
# column loop: linear stencils, boundary conditions, and nonlinear advection.
function test_fused_column_loop(space, subspace)
    FT = Spaces.undertype(space)
    extrapolate = Operators.Extrapolate()
    ᶠinterp = Operators.InterpolateC2F(; bottom = extrapolate, top = extrapolate)
    ᶜinterp = Operators.InterpolateF2C()
    ᶠgrad = Operators.GradientC2F(;
        bottom = Operators.SetValue(FT(0)),
        top = Operators.SetGradient(Geometry.WVector(FT(1))),
    )
    ᶜdiv = Operators.DivergenceF2C()
    ᶠupwind = Operators.Upwind3rdOrderBiasedProductC2F()
    ᶠvanleer = Operators.LinVanLeerC2F(; constraint = Operators.MonotoneLocalExtrema())
    dest = Fields.Field(FT, space)
    src1 = Fields.Field(Tuple{FT, FT}, space)
    src2 = Fields.Field(Tuple{FT, FT}, subspace)
    parent(src1) .= rand.(FT)
    parent(src2) .= rand.(FT)

    function fused_loop!(dest, src1, src2)
        @. dest = 0
        temp1 = @. ᶜdiv(Geometry.WVector(ᶠgrad(src1.:1))) + src2.:1
        @. dest += temp1 * temp1
        temp2 = @. ᶜinterp(ᶠinterp(src1.:2)) -
           ᶜdiv(ᶠupwind(Geometry.WVector(ᶠinterp(temp1)), src1.:2))
        @. dest +=
            (
                ᶜdiv(Geometry.WVector(ᶠgrad(temp1))) +
                ᶜdiv(ᶠvanleer(Geometry.WVector(ᶠinterp(temp2)), temp2, 1)) + src2.:2
            )^2
        @. dest /= 2
    end

    foreach_column(fused_loop!, dest, src1, src2)
    temp1_ref = @. ᶜdiv(Geometry.WVector(ᶠgrad(src1.:1))) + src2.:1
    temp2_ref =
        @. ᶜinterp(ᶠinterp(src1.:2)) -
           ᶜdiv(ᶠupwind(Geometry.WVector(ᶠinterp(temp1_ref)), src1.:2))
    @test dest ≈ @. (
        temp1_ref^2 +
        (
            ᶜdiv(Geometry.WVector(ᶠgrad(temp1_ref))) +
            ᶜdiv(ᶠvanleer(Geometry.WVector(ᶠinterp(temp2_ref)), temp2_ref, 1)) +
            src2.:2
        )^2
    ) / 2

    CUDA_FRAMES = @isdefined(CUDA) ? (AnyFrameModule(CUDA),) : ()
    @test_opt ignored_modules = CUDA_FRAMES foreach_column(fused_loop!, dest, src1, src2)
end

has_center_columns(space) =
    Spaces.has_vertical(space) && Spaces.staggering(space) == Spaces.CellCenter()

function test_all_fused_loops(space1)
    test_fused_loop(foreach_point, space1, space1)
    test_fused_loop(foreach_level, space1, space1)
    test_fused_loop(foreach_slab, space1, space1)
    test_fused_loop(foreach_column, space1, space1)
    Spaces.has_horizontal(space1) && test_fused_slab_loop(space1, space1)
    has_center_columns(space1) && test_fused_column_loop(space1, space1)

    space2 = Spaces.level(space1, 1)
    if space1 !== space2
        @test_throws DimensionMismatch test_fused_loop(foreach_point, space1, space2)
        @test_throws DimensionMismatch test_fused_loop(foreach_level, space1, space2)
        @test_throws DimensionMismatch test_fused_loop(foreach_slab, space1, space2)
        test_fused_loop(foreach_column, space1, space2)
        has_center_columns(space1) && test_fused_column_loop(space1, space2)
    end

    space3 = Spaces.column(space1, 1, 1, 1)
    if space1 !== space3
        @test_throws DimensionMismatch test_fused_loop(foreach_point, space1, space3)
        @test_throws DimensionMismatch test_fused_loop(foreach_column, space1, space3)
        test_fused_loop(foreach_level, space1, space3)
        # Mismatched-subspace args make foreach_slab throw before the loop body is
        # considered; only the single-slab case runs a body with a column argument.
        if DataLayouts.nelems(Spaces.local_geometry_data(space1)) == 1
            test_fused_loop(foreach_slab, space1, space3)
            Spaces.has_horizontal(space1) && test_fused_slab_loop(space1, space3)
        else
            @test_throws DimensionMismatch test_fused_loop(foreach_slab, space1, space3)
        end
    end
end

@testset "fused slice-loop broadcasts" begin
    context = ClimaComms.context(ClimaComms.device())
    foreach(test_all_fused_loops, TU.all_spaces(Float64; context))
    # Columns with more points than the threads of a GPU sub-block
    tall_space = TU.CenterExtrudedFiniteDifferenceSpace(
        Float64;
        zelem = 200,
        helem = 1,
        Nq = 2,
        context,
    )
    test_fused_column_loop(tall_space, tall_space)
end

# Every statement of a fused loop body is compiled into the same GPU kernel, and statements
# share the shared memory of their slice, since the buffers of different statements are
# never live at the same time.
@testset "fused loops with many shared-memory buffers" begin
    FT = Float64
    context = ClimaComms.context(ClimaComms.device())
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 63, helem = 2, context)
    ᶜinterp = Operators.InterpolateF2C()
    ᶠinterp = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    tuple_field(n) = fill(ntuple(FT, n), space)
    reference(arg) =
        Base.materialize(Base.broadcasted(ᶜinterp, Base.broadcasted(ᶠinterp, arg)))

    # 17 statements that each cache a column of 64 faces with 6 values (3 KB) when it fits in
    # shared memory, 51 KB if they did not share their buffers.
    x = tuple_field(6)
    outputs = ntuple(_ -> similar(x), 17)
    foreach_column(outputs..., x) do args...
        ntuple(Val(17)) do n
            (output, input) = (args[n], args[18])
            @. output = ᶜinterp(ᶠinterp(n * input))
            return nothing
        end
        return nothing
    end
    @test all(n -> parent(outputs[n]) == parent(reference(Base.broadcasted(*, n, x))), 1:17)

    # One statement that sums the same 17 operators.
    terms = ntuple(
        n -> Base.broadcasted(
            ᶜinterp,
            Base.broadcasted(ᶠinterp, Base.broadcasted(*, n, x)),
        ),
        17,
    )
    sum_output = Base.materialize(reduce((a, b) -> Base.broadcasted(+, a, b), terms))
    @test parent(sum_output) == sum(n -> parent(outputs[n]), 1:17)

    # The 17 statements on a temporary of the loop body, which is in registers on GPUs and
    # has to be cached in shared memory (48 bytes per thread of a column, which the :medium
    # level of some devices does not hold).
    foreach_column(outputs..., x; shmem = Val(:max)) do args...
        temp = @. 2 * args[18]
        ntuple(Val(17)) do n
            output = args[n]
            @. output = ᶜinterp(ᶠinterp(n * temp))
            return nothing
        end
        return nothing
    end
    @test all(
        n -> parent(outputs[n]) == parent(reference(Base.broadcasted(*, 2n, x))),
        1:17,
    )

    # 14 statements whose cached columns have 1 to 14 values (0.5 KB to 7 KB): 52.5 KB of
    # shared memory if they did not share it; those that do not fit are recomputed.
    inputs = ntuple(tuple_field, 14)
    outputs = map(similar, inputs)
    fused_loop!() = foreach_column(outputs..., inputs...) do args...
        ntuple(Val(14)) do n
            (output, input) = (args[n], args[14 + n])
            @. output = ᶜinterp(ᶠinterp(input))
            return nothing
        end
        return nothing
    end
    correct() = all(n -> parent(outputs[n]) == parent(reference(inputs[n])), 1:14)

    # A column of 1101 faces with 6 values (52 KB) is recomputed instead of cached.
    tall_space =
        TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 1100, helem = 1, Nq = 2, context)
    tall_x = fill(ntuple(FT, 6), tall_space)
    @test parent(@. ᶜinterp(ᶠinterp(2 * tall_x))) ==
          parent(fill(ntuple(n -> 2 * FT(n), 6), tall_space))
    fused_loop!()
    @test correct()
end

# Temporaries of a fused loop body are in registers on GPUs (see register_similar), stored
# in MArrays that GPU kernels would allocate on the heap if they were passed to a function
# that is not inlined, like a separately compiled statement or operator.
@testset "fused loops with register temporaries" begin
    FT = Float64
    context = ClimaComms.context(ClimaComms.device())
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 63, helem = 2, context)
    ᶜinterp = Operators.InterpolateF2C()
    ᶠinterp = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    # Property views of a temporary, read and written by pointwise and operator statements.
    x = fill((FT(1), FT(2)), space)
    output = Fields.Field(FT, space)
    foreach_column(output, x) do output, x
        temp = @. 2 * x
        @. temp.:2 = ᶜinterp(ᶠinterp(temp.:1)) + temp.:2
        @. output = temp.:1 * temp.:2
        return nothing
    end
    @test all(==(FT(12)), Array(parent(output)))

    # Temporaries that other threads read at neighboring points are cached in shared memory,
    # since they are in registers, so a large one needs a level of shared memory that fits it:
    # 25 KB for 63 centers with 50 values, or 26 KB for 1100 centers with 3 values (with 128
    # threads per column, which takes a block over the default limit of 48 KB).
    for (zelem, n) in ((63, Val(50)), (1100, Val(3)))
        tall_space =
            TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem, helem = 1, Nq = 2, context)
        z = Fields.coordinate_field(tall_space).z
        x = map(z -> ntuple(i -> i * z^2, n), z)
        output = Fields.Field(eltype(x), Spaces.face_space(tall_space))
        foreach_column(output, x; shmem = Val(:max)) do output, x
            temp = @. 2 * x
            @. output = ᶠinterp(temp)
            return nothing
        end
        @test parent(output) ≈ parent(@. ᶠinterp(2 * x))
    end

    # Operators on 6 values of a temporary (48 bytes per thread of a column, which the
    # :medium level of some devices does not hold).
    lazy_interp(arg) = Base.broadcasted(ᶜinterp, Base.broadcasted(ᶠinterp, arg))
    x = fill(ntuple(FT, 6), space)
    output = similar(x)
    foreach_column(output, x; shmem = Val(:max)) do output, x
        temp = @. 2 * x
        deep_interp = lazy_interp(lazy_interp(lazy_interp(lazy_interp(lazy_interp(temp)))))
        Base.materialize!(output, deep_interp)
        return nothing
    end
    @test parent(output) == parent(@. 2 * x)
    (grad, wdiv) = (Operators.Gradient(), Operators.Divergence{Operators.WeakForm}())
    slab_space =
        TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 1, helem = 1, Nq = 8, context)
    y = map(φ -> ntuple(n -> n * sind(φ), 6), Fields.coordinate_field(slab_space).lat)
    output = similar(y)
    foreach_slab(output, y; shmem = Val(:max)) do output, y
        temp = @. 2 * y
        @. output = wdiv(grad(wdiv(grad(wdiv(grad(temp))))))
        return nothing
    end
    @test parent(output) ≈ parent(@. wdiv(grad(wdiv(grad(wdiv(grad(2 * y)))))))

    # A statement with more operators than the inliner inlines, which writes to a temporary.
    x = fill(FT(1), space)
    output = similar(x)
    foreach_column(output, x) do output, x
        temp = @. 2 * x
        sum_temp = @. ᶜinterp(ᶠinterp(temp)) + ᶜinterp(ᶠinterp(2 * temp)) +
           ᶜinterp(ᶠinterp(3 * temp)) + ᶜinterp(ᶠinterp(4 * temp)) +
           ᶜinterp(ᶠinterp(5 * temp)) + ᶜinterp(ᶠinterp(6 * temp)) +
           ᶜinterp(ᶠinterp(7 * temp)) + ᶜinterp(ᶠinterp(8 * temp)) +
           ᶜinterp(ᶠinterp(9 * temp)) + ᶜinterp(ᶠinterp(10 * temp)) +
           ᶜinterp(ᶠinterp(11 * temp)) + ᶜinterp(ᶠinterp(12 * temp))
        @. output = sum_temp
        return nothing
    end
    @test all(==(FT(156)), Array(parent(output)))

    # SetValue boundary faces read the closest centers of a temporary, which on GPUs are in
    # the registers of other threads.
    ᶜz = Fields.coordinate_field(space).z
    x = @. 1 + ᶜz^2
    output = similar(x)
    ᶜdiv = Operators.DivergenceF2C()
    ᶠgrad = Operators.GradientC2F(;
        bottom = Operators.SetValue(FT(0)),
        top = Operators.SetValue(FT(1)),
    )
    foreach_column(output, x) do output, x
        temp = @. 2 * x
        @. output = ᶜdiv(Geometry.WVector(ᶠgrad(temp)))
        return nothing
    end
    @test Array(parent(output)) ≈ Array(parent(@. ᶜdiv(Geometry.WVector(ᶠgrad(2 * x)))))
end
