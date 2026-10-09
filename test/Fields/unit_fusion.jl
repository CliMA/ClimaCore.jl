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

# As in test_fused_slab_loop, but with finite difference operators, which read
# every point of a column, in a fused column loop. The operators cover linear
# stencils (rewritten as operator matrix products), boundary conditions that
# set values or derivatives, and linear and nonlinear advection.
function test_fused_column_loop(space, subspace)
    FT = Spaces.undertype(space)
    ᶠinterp = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
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
    # Columns with more points than the threads of a GPU sub-block, which loop
    # over several points per thread in every statement of the fused loop
    tall_space = TU.CenterExtrudedFiniteDifferenceSpace(
        Float64;
        zelem = 200,
        helem = 1,
        Nq = 2,
        context,
    )
    test_fused_column_loop(tall_space, tall_space)
end
