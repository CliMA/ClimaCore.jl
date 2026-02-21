#=
julia --project=.buildkite
using Revise; include(joinpath("test", "MatrixFields", "operator_matrices.jl"))
=#

import LinearAlgebra: I

import ClimaCore
import ClimaCore.Operators:
    SetValue,
    SetGradient,
    SetDivergence,
    SetCurl,
    Extrapolate,
    FirstOrderOneSided,
    ThirdOrderOneSided,
    InterpolateC2F,
    InterpolateF2C,
    LeftBiasedC2F,
    LeftBiasedF2C,
    RightBiasedC2F,
    RightBiasedF2C,
    WeightedInterpolateC2F,
    WeightedInterpolateF2C,
    UpwindBiasedProductC2F,
    Upwind3rdOrderBiasedProductC2F,
    FCTBorisBook,
    FCTZalesak,
    AdvectionC2C,
    AdvectionF2F,
    FluxCorrectionC2C,
    FluxCorrectionF2F,
    SetBoundaryOperator,
    GradientC2F,
    GradientF2C,
    DivergenceC2F,
    DivergenceF2C,
    CurlC2F

include("matrix_field_test_utils.jl")

apply_op_matrix(::Nothing, op_matrix, arg) = @lazy @. op_matrix() * arg
apply_op_matrix(boundary_op, op_matrix, arg) =
    @lazy @. boundary_op(op_matrix() * arg)
apply_op_matrix(::Nothing, op_matrix, arg1, arg2) =
    @lazy @. op_matrix(arg1) * arg2
apply_op_matrix(boundary_op, op_matrix, arg1, arg2) =
    @lazy @. boundary_op(op_matrix(arg1) * arg2)

apply_op(::Nothing, op, args...) = @lazy @. op(args...)
apply_op(boundary_op, op, args...) = @lazy @. boundary_op(op(args...))

function test_op_matrix(
    ::Type{Op},
    ::Type{BC},
    args,
    requires_boundary_values = false,
) where {Op, BC}
    FT = Spaces.undertype(axes(args[end]))

    # Use zeroed-out boundary conditions to avoid affine operator warnings.
    op_bc = if BC <: SetValue
        BC(nested_zero(eltype(args[end])))
    elseif BC <: SetGradient
        BC(zero(Geometry.Covariant3Vector{FT}))
    elseif BC <: SetDivergence
        BC(zero(FT))
    elseif BC <: SetCurl
        BC(zero(Geometry.Contravariant12Vector{FT}))
    else
        BC()
    end

    op = if BC <: Nothing
        Op()
    elseif Op <: Union{LeftBiasedC2F, LeftBiasedF2C}
        Op(; bottom = op_bc)
    elseif Op <: Union{RightBiasedC2F, RightBiasedF2C}
        Op(; top = op_bc)
    else
        Op(; bottom = op_bc, top = op_bc)
    end
    op_matrix = MatrixFields.operator_matrix(op)

    # This boundary condition doesn't matter, since it's applied after the
    # operator. It is zeroed out for simplicity, but it does not need to be.
    boundary_op = if requires_boundary_values
        boundary_op_bc = SetValue(
            nested_zero(eltype(Base.Broadcast.broadcasted(op, args...))),
        )
        SetBoundaryOperator(; bottom = boundary_op_bc, top = boundary_op_bc)
    else
        nothing
    end

    test_field_broadcast(;
        test_name = "operator matrix of $Op ($(BC <: Nothing ? "no BCs" : BC))",
        get_result = apply_op_matrix(boundary_op, op_matrix, args...),
        set_result = apply_op_matrix(boundary_op, op_matrix, args...),
        ref_set_result = apply_op(boundary_op, op, args...),
        time_ratio_limit = 60, # Extrapolating operator matrices are very slow.
    )
end

@testset "operator_matrix Unit Tests" begin
    FT = Float64
    center_space, face_space = test_spaces(FT)

    seed!(1) # ensures reproducibility
    ᶜscalar = random_field(FT, center_space)
    ᶠscalar = random_field(FT, face_space)
    ᶜnested = random_field(NestedType{FT}, center_space)
    ᶠnested = random_field(NestedType{FT}, face_space)
    ᶜuvw = random_field(Geometry.UVWVector{FT}, center_space)
    ᶠuvw = random_field(Geometry.UVWVector{FT}, face_space)
    ᶜc12 = random_field(Geometry.Covariant12Vector{FT}, center_space)

    # For each operator, test the operator matrix for every possible boundary
    # condition, and use the most generic possible inputs. The nested inputs can
    # be replaced with any nested or scalar type, and the UVW inputs can be
    # replaced with any vector type.
    # Note: Even though the UpwindBiasedProduct and Gradient operators should
    # work with nested inputs, they currently throw errors unless they are given
    # scalar inputs because of bugs in their return_eltype methods.
    # Note: The Curl operator currently only works with C12, C1, or C2 inputs.
    test_op_matrix(InterpolateC2F, Nothing, (ᶜnested,), true)
    test_op_matrix(InterpolateC2F, SetValue, (ᶜnested,))
    test_op_matrix(InterpolateC2F, SetGradient, (ᶜnested,))
    test_op_matrix(InterpolateC2F, Extrapolate, (ᶜnested,))
    test_op_matrix(InterpolateF2C, Nothing, (ᶠnested,))
    test_op_matrix(LeftBiasedC2F, Nothing, (ᶜnested,), true)
    test_op_matrix(LeftBiasedC2F, SetValue, (ᶜnested,))
    test_op_matrix(LeftBiasedF2C, Nothing, (ᶠnested,))
    test_op_matrix(LeftBiasedF2C, SetValue, (ᶠnested,))
    test_op_matrix(RightBiasedC2F, Nothing, (ᶜnested,), true)
    test_op_matrix(RightBiasedC2F, SetValue, (ᶜnested,))
    test_op_matrix(RightBiasedF2C, Nothing, (ᶠnested,))
    test_op_matrix(RightBiasedF2C, SetValue, (ᶠnested,))
    test_op_matrix(WeightedInterpolateC2F, Nothing, (ᶜscalar, ᶜnested), true)
    test_op_matrix(WeightedInterpolateC2F, SetValue, (ᶜscalar, ᶜnested))
    test_op_matrix(WeightedInterpolateC2F, SetGradient, (ᶜscalar, ᶜnested))
    test_op_matrix(WeightedInterpolateC2F, Extrapolate, (ᶜscalar, ᶜnested))
    test_op_matrix(WeightedInterpolateF2C, Nothing, (ᶠscalar, ᶠnested))
    test_op_matrix(UpwindBiasedProductC2F, Nothing, (ᶠuvw, ᶜscalar), true)
    test_op_matrix(UpwindBiasedProductC2F, SetValue, (ᶠuvw, ᶜscalar))
    test_op_matrix(UpwindBiasedProductC2F, Extrapolate, (ᶠuvw, ᶜscalar))
    test_op_matrix(
        Upwind3rdOrderBiasedProductC2F,
        FirstOrderOneSided,
        (ᶠuvw, ᶜscalar),
        true,
    )
    test_op_matrix(
        Upwind3rdOrderBiasedProductC2F,
        ThirdOrderOneSided,
        (ᶠuvw, ᶜscalar),
        true,
    )
    test_op_matrix(AdvectionC2C, SetValue, (ᶠuvw, ᶜnested))
    test_op_matrix(AdvectionC2C, Extrapolate, (ᶠuvw, ᶜnested))
    test_op_matrix(AdvectionF2F, Nothing, (ᶠuvw, ᶠnested), true)
    test_op_matrix(FluxCorrectionC2C, Extrapolate, (ᶠuvw, ᶜnested))
    test_op_matrix(FluxCorrectionF2F, Extrapolate, (ᶜuvw, ᶠnested))
    test_op_matrix(SetBoundaryOperator, SetValue, (ᶠnested,))
    test_op_matrix(GradientC2F, Nothing, (ᶜscalar,), true)
    test_op_matrix(GradientC2F, SetValue, (ᶜscalar,))
    test_op_matrix(GradientC2F, SetGradient, (ᶜscalar,))
    test_op_matrix(GradientF2C, Nothing, (ᶠscalar,))
    test_op_matrix(GradientF2C, SetValue, (ᶠscalar,))
    test_op_matrix(GradientF2C, Extrapolate, (ᶠscalar,))
    test_op_matrix(DivergenceC2F, Nothing, (ᶜuvw,), true)
    test_op_matrix(DivergenceC2F, SetValue, (ᶜuvw,))
    test_op_matrix(DivergenceC2F, SetDivergence, (ᶜuvw,))
    test_op_matrix(DivergenceF2C, Nothing, (ᶠuvw,))
    test_op_matrix(DivergenceF2C, SetValue, (ᶠuvw,))
    test_op_matrix(DivergenceF2C, SetDivergence, (ᶠuvw,))
    test_op_matrix(DivergenceF2C, Extrapolate, (ᶠuvw,))
    test_op_matrix(CurlC2F, Nothing, (ᶜc12,), true)
    test_op_matrix(CurlC2F, SetValue, (ᶜc12,))
    test_op_matrix(CurlC2F, SetCurl, (ᶜc12,))

    @test_throws "nonlinear" MatrixFields.operator_matrix(FCTBorisBook())
    @test_throws "nonlinear" MatrixFields.operator_matrix(FCTZalesak())
end

@testset "Operator Matrix Broadcasting" begin
    FT = Float64
    center_space, face_space = test_spaces(FT)

    seed!(1) # ensures reproducibility
    ᶜscalar = random_field(FT, center_space)
    ᶠscalar = random_field(FT, face_space)
    ᶜnested = random_field(NestedType{FT}, center_space)
    ᶠuvw = random_field(Geometry.UVWVector{FT}, face_space)
    c12_a = rand(Geometry.Covariant12Vector{FT})
    c12_b = rand(Geometry.Covariant12Vector{FT})

    set_scalar_values =
        (; bottom = SetValue(zero(FT)), top = SetValue(zero(FT)))
    set_nested_values = (;
        bottom = SetValue(nested_zero(NestedType{FT})),
        top = SetValue(nested_zero(NestedType{FT})),
    )
    c12_zero = zero(Geometry.Covariant12Vector{FT})
    set_c12_values = (; bottom = SetValue(c12_zero), top = SetValue(c12_zero))
    extrapolate = (; bottom = Extrapolate(), top = Extrapolate())

    ᶠinterp = InterpolateC2F(; set_nested_values...)
    ᶜlbias = LeftBiasedF2C()
    ᶠrbias = RightBiasedC2F(; set_nested_values.top)
    ᶜwinterp = WeightedInterpolateF2C()
    ᶠupwind = UpwindBiasedProductC2F(; set_scalar_values...)
    ᶜadvect = AdvectionC2C(; extrapolate...)
    ᶜflux_correct = FluxCorrectionC2C(; extrapolate...)
    ᶠgrad = GradientC2F(; set_scalar_values...)
    ᶜdiv = DivergenceF2C()
    ᶠcurl = CurlC2F(; set_c12_values...)
    ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)
    ᶜlbias_matrix = MatrixFields.operator_matrix(ᶜlbias)
    ᶠrbias_matrix = MatrixFields.operator_matrix(ᶠrbias)
    ᶜwinterp_matrix = MatrixFields.operator_matrix(ᶜwinterp)
    ᶠupwind_matrix = MatrixFields.operator_matrix(ᶠupwind)
    ᶜadvect_matrix = MatrixFields.operator_matrix(ᶜadvect)
    ᶜflux_correct_matrix = MatrixFields.operator_matrix(ᶜflux_correct)
    ᶠgrad_matrix = MatrixFields.operator_matrix(ᶠgrad)
    ᶜdiv_matrix = MatrixFields.operator_matrix(ᶜdiv)
    ᶠcurl_matrix = MatrixFields.operator_matrix(ᶠcurl)

    @test_throws "does not contain any Fields" @. ᶜlbias_matrix() *
                                                  ᶠinterp_matrix()

    ᶜ0 = @. zero(ᶜscalar)
    ᶜ1 = @. one(ᶜscalar)
    ᶠ1 = @. one(ᶠscalar)
    for get_result in (
        @lazy(@. ᶜlbias_matrix() * ᶠinterp_matrix() + DiagonalMatrixRow(ᶜ0)),
        @lazy(@. DiagonalMatrixRow(ᶜ0) + ᶜlbias_matrix() * ᶠinterp_matrix()),
        @lazy(@. ᶜlbias_matrix() * ᶠinterp_matrix() * DiagonalMatrixRow(ᶜ1)),
        @lazy(@. ᶜlbias_matrix() * DiagonalMatrixRow(ᶠ1) * ᶠinterp_matrix()),
        @lazy(@. DiagonalMatrixRow(ᶜ1) * ᶜlbias_matrix() * ᶠinterp_matrix()),
    )
        test_field_broadcast(;
            test_name = "product of two lazy operator matrices",
            get_result,
            set_result = @lazy(@. ᶜlbias_matrix() * ᶠinterp_matrix()),
        )
    end

    test_field_broadcast(;
        test_name = "product of six operator matrices",
        get_result = @lazy(
            @. ᶜflux_correct_matrix(ᶠuvw) *
               ᶜadvect_matrix(ᶠuvw) *
               ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix()
        ),
        set_result = @lazy(
            @. ᶜflux_correct_matrix(ᶠuvw) *
               ᶜadvect_matrix(ᶠuvw) *
               ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix()
        ),
    )

    test_field_broadcast(;
        test_name = "applying six operators to a nested field using operator \
                     matrices",
        get_result = @lazy(
            @. ᶜflux_correct_matrix(ᶠuvw) *
               ᶜadvect_matrix(ᶠuvw) *
               ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix() *
               ᶜnested
        ),
        set_result = @lazy(
            @. ᶜflux_correct_matrix(ᶠuvw) *
               ᶜadvect_matrix(ᶠuvw) *
               ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix() *
               ᶜnested
        ),
        ref_set_result = @lazy(
            @. ᶜflux_correct(
                ᶠuvw,
                ᶜadvect(
                    ᶠuvw,
                    ᶜwinterp(ᶠscalar, ᶠrbias(ᶜlbias(ᶠinterp(ᶜnested)))),
                ),
            )
        ),
    )

    test_field_broadcast(;
        test_name = "applying six operators to a nested field using operator \
                     matrices, but with forced right associativity",
        get_result = @lazy(
            @. ᶜflux_correct_matrix(ᶠuvw) * (
                ᶜadvect_matrix(ᶠuvw) * (
                    ᶜwinterp_matrix(ᶠscalar) * (
                        ᶠrbias_matrix() *
                        (ᶜlbias_matrix() * (ᶠinterp_matrix() * ᶜnested))
                    )
                )
            )
        ),
        set_result = @lazy(
            @. ᶜflux_correct_matrix(ᶠuvw) * (
                ᶜadvect_matrix(ᶠuvw) * (
                    ᶜwinterp_matrix(ᶠscalar) * (
                        ᶠrbias_matrix() *
                        (ᶜlbias_matrix() * (ᶠinterp_matrix() * ᶜnested))
                    )
                )
            )
        ),
        ref_set_result = @lazy(
            @. ᶜflux_correct(
                ᶠuvw,
                ᶜadvect(
                    ᶠuvw,
                    ᶜwinterp(ᶠscalar, ᶠrbias(ᶜlbias(ᶠinterp(ᶜnested)))),
                ),
            )
        ),
        time_ratio_limit = 30, # This case's ref function is fast on Buildkite.
        test_broken_with_cuda = true, # TODO: Fix this.
    )

    # TODO: For some reason, we need to compile and run @test_opt on several
    # simpler broadcast expressions before we can run the remaining two test
    # cases. As of Julia 1.8.5, the tests fail if we skip this step. Is this a
    # false positive, a compiler issue, or a sign that the code can be improved?
    for get_result in (
        @lazy(
            @. (c12_b',) *
               ᶜwinterp_matrix(ᶠscalar) *
               ᶠcurl_matrix() *
               (c12_a,) +
               (DiagonalMatrixRow(ᶜdiv(ᶠuvw)) - ᶜadvect_matrix(ᶠuvw)) / 5
        ),
        @lazy(
            @. ᶜdiv_matrix() *
               DiagonalMatrixRow(ᶠscalar) *
               ᶠgrad_matrix() *
               (
                   (c12_b',) *
                   ᶜwinterp_matrix(ᶠscalar) *
                   ᶠcurl_matrix() *
                   (c12_a,) +
                   (DiagonalMatrixRow(ᶜdiv(ᶠuvw)) - ᶜadvect_matrix(ᶠuvw)) / 5
               )
        ),
    )
        materialize(get_result)
        @test_opt ignored_modules = cuda_frames materialize(get_result)
    end

    test_field_broadcast(;
        test_name = "non-trivial combination of operator matrices and other \
                     matrix fields",
        get_result = @lazy(
            @. ᶠupwind_matrix(ᶠuvw) * (
                ᶜdiv_matrix() *
                DiagonalMatrixRow(ᶠscalar) *
                ᶠgrad_matrix() *
                (
                    (c12_b',) *
                    ᶜwinterp_matrix(ᶠscalar) *
                    ᶠcurl_matrix() *
                    (c12_a,) +
                    (DiagonalMatrixRow(ᶜdiv(ᶠuvw)) - ᶜadvect_matrix(ᶠuvw)) / 5
                ) - (2I,)
            )
        ),
        set_result = @lazy(
            @. ᶠupwind_matrix(ᶠuvw) * (
                ᶜdiv_matrix() *
                DiagonalMatrixRow(ᶠscalar) *
                ᶠgrad_matrix() *
                (
                    (c12_b',) *
                    ᶜwinterp_matrix(ᶠscalar) *
                    ᶠcurl_matrix() *
                    (c12_a,) +
                    (DiagonalMatrixRow(ᶜdiv(ᶠuvw)) - ᶜadvect_matrix(ᶠuvw)) / 5
                ) - (2I,)
            )
        ),
    )

    # TODO: This case's reference function takes too long to compile on both
    # CPUs and GPUs (more than half an hour), as of Julia 1.9. This might be
    # happening because of excessive inlining---aside from *, all other finite
    # difference operators use @propagate_inbounds. So, the reference function
    # is currently disabled, although the test does pass when it is enabled.
    test_field_broadcast(;
        test_name = "applying a non-trivial sequence of operations to a scalar \
                     field using operator matrices and other matrix fields",
        get_result = @lazy(
            @. ᶠupwind_matrix(ᶠuvw) *
               (
                   ᶜdiv_matrix() *
                   DiagonalMatrixRow(ᶠscalar) *
                   ᶠgrad_matrix() *
                   (
                       (c12_b',) *
                       ᶜwinterp_matrix(ᶠscalar) *
                       ᶠcurl_matrix() *
                       (c12_a,) +
                       (DiagonalMatrixRow(ᶜdiv(ᶠuvw)) - ᶜadvect_matrix(ᶠuvw)) /
                       5
                   ) - (2I,)
               ) *
               ᶜscalar
        ),
        set_result = @lazy(
            @. ᶠupwind_matrix(ᶠuvw) *
               (
                   ᶜdiv_matrix() *
                   DiagonalMatrixRow(ᶠscalar) *
                   ᶠgrad_matrix() *
                   (
                       (c12_b',) *
                       ᶜwinterp_matrix(ᶠscalar) *
                       ᶠcurl_matrix() *
                       (c12_a,) +
                       (DiagonalMatrixRow(ᶜdiv(ᶠuvw)) - ᶜadvect_matrix(ᶠuvw)) /
                       5
                   ) - (2I,)
               ) *
               ᶜscalar
        ),
        # ref_set_result = @lazy(@. ᶠupwind(
        #     ᶠuvw,
        #     ᶜdiv(
        #         ᶠscalar * ᶠgrad(
        #             (c12_b',) * ᶜwinterp(ᶠscalar, ᶠcurl((c12_a,) * ᶜscalar)) +
        #             (ᶜdiv(ᶠuvw) * ᶜscalar - ᶜadvect(ᶠuvw, ᶜscalar)) / 5,
        #         ),
        #     ) - 2 * ᶜscalar,
        # )),
        # max_eps_error_limit = 20, # This case's roundoff error is large.
    )
end
