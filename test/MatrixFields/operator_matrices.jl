#=
julia --project=.buildkite
using Revise; include(joinpath("test", "MatrixFields", "operator_matrices.jl"))
=#

import LinearAlgebra: I

import ClimaCore.RecursiveApply: rzero
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
        BC(rzero(eltype(args[end])))
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
            rzero(eltype(Base.Broadcast.broadcasted(op, args...))),
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
    test_op_matrix(WeightedInterpolateC2F, Extrapolate, (ᶜscalar, ᶜnested))
    test_op_matrix(WeightedInterpolateF2C, Nothing, (ᶠscalar, ᶠnested))
    test_op_matrix(UpwindBiasedProductC2F, Nothing, (ᶠuvw, ᶜscalar), true)
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
    test_op_matrix(FluxCorrectionF2F, Extrapolate, (ᶜuvw, ᶠnested))
    test_op_matrix(SetBoundaryOperator, SetValue, (ᶠnested,))
    test_op_matrix(GradientC2F, Nothing, (ᶜscalar,), true)
    test_op_matrix(GradientC2F, SetGradient, (ᶜscalar,))
    test_op_matrix(GradientF2C, Nothing, (ᶠscalar,))
    test_op_matrix(GradientF2C, SetValue, (ᶠscalar,))
    test_op_matrix(GradientF2C, Extrapolate, (ᶠscalar,))
    test_op_matrix(DivergenceC2F, Nothing, (ᶜuvw,), true)
    test_op_matrix(DivergenceC2F, SetDivergence, (ᶜuvw,))
    test_op_matrix(DivergenceF2C, Nothing, (ᶠuvw,))
    test_op_matrix(DivergenceF2C, SetValue, (ᶠuvw,))
    test_op_matrix(DivergenceF2C, SetDivergence, (ᶠuvw,))
    test_op_matrix(DivergenceF2C, Extrapolate, (ᶠuvw,))
    test_op_matrix(CurlC2F, Nothing, (ᶜc12,), true)
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
    nested_zero = rzero(NestedType{FT})
    set_nested_values =
        (; bottom = SetValue(nested_zero), top = SetValue(nested_zero))
    c12_zero = zero(Geometry.Covariant12Vector{FT})
    extrapolate = (; bottom = Extrapolate(), top = Extrapolate())

    ᶠinterp = InterpolateC2F(; set_nested_values...)
    ᶜlbias = LeftBiasedF2C()
    ᶠrbias = RightBiasedC2F(; set_nested_values.top)
    ᶜwinterp = WeightedInterpolateF2C()
    ᶜdiv = DivergenceF2C()
    ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)
    ᶜlbias_matrix = MatrixFields.operator_matrix(ᶜlbias)
    ᶠrbias_matrix = MatrixFields.operator_matrix(ᶠrbias)
    ᶜwinterp_matrix = MatrixFields.operator_matrix(ᶜwinterp)
    ᶜdiv_matrix = MatrixFields.operator_matrix(ᶜdiv)

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
            @. ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix()
        ),
        set_result = @lazy(
            @. ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix()
        ),
    )

    test_field_broadcast(;
        test_name = "applying six operators to a nested field using operator \
                     matrices",
        get_result = @lazy(
            @. ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix() *
               ᶜnested
        ),
        set_result = @lazy(
            @. ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix() *
               ᶜnested
        ),
        ref_set_result = @lazy(
            @. ᶜwinterp(ᶠscalar, ᶠrbias(ᶜlbias(ᶠinterp(ᶜnested))))
        ),
    )
    # this test is will fail because of incorrect results, not InvalidIRError
    USING_CUDA || test_field_broadcast(;
        test_name = "applying six operators to a nested field using operator \
                     matrices, but with forced right associativity",
        get_result = @lazy(
            @. (
                (
                ᶜwinterp_matrix(ᶠscalar) * (
                    ᶠrbias_matrix() *
                    (ᶜlbias_matrix() * (ᶠinterp_matrix() * ᶜnested))
                )
            )
            )
        ),
        set_result = @lazy(
            @. (
                (
                ᶜwinterp_matrix(ᶠscalar) * (
                    ᶠrbias_matrix() *
                    (ᶜlbias_matrix() * (ᶠinterp_matrix() * ᶜnested))
                )
            )
            )
        ),
        ref_set_result = @lazy(
            @. ᶜwinterp(ᶠscalar, ᶠrbias(ᶜlbias(ᶠinterp(ᶜnested))))
        ),
        time_ratio_limit = 30, # This case's ref function is fast on Buildkite.
        test_broken_with_cuda = true, # TODO: Fix this.
    )

end
