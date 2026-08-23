import LinearAlgebra: I

import ClimaCore.RecursiveApply: rzero
import ClimaCore
import ClimaCore.Operators:
    SetValue,
    SetGradient,
    SetDivergence,
    SetCurl,
    Extrapolate,
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
    LinVanLeerC2F,
    TVDLimitedFluxC2F,
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
    # The advection operators' boundary faces are computed with the interior
    # stencil, padding ghost points with the Extrapolate boundary condition's
    # extrapolation from the in-range interior points (Extrapolate{0}, the
    # value of the closest interior point, when no boundary condition is
    # given), so no SetBoundaryOperator is needed.
    test_op_matrix(UpwindBiasedProductC2F, Nothing, (ᶠuvw, ᶜscalar))
    test_op_matrix(UpwindBiasedProductC2F, Extrapolate{0}, (ᶠuvw, ᶜscalar))
    test_op_matrix(UpwindBiasedProductC2F, Extrapolate{2}, (ᶠuvw, ᶜscalar))
    test_op_matrix(Upwind3rdOrderBiasedProductC2F, Nothing, (ᶠuvw, ᶜscalar))
    test_op_matrix(
        Upwind3rdOrderBiasedProductC2F,
        Extrapolate{0},
        (ᶠuvw, ᶜscalar),
    )
    test_op_matrix(
        Upwind3rdOrderBiasedProductC2F,
        Extrapolate{1},
        (ᶠuvw, ᶜscalar),
    )
    test_op_matrix(
        Upwind3rdOrderBiasedProductC2F,
        Extrapolate{2},
        (ᶠuvw, ᶜscalar),
    )
    test_op_matrix(SetBoundaryOperator, SetValue, (ᶠnested,))
    test_op_matrix(GradientC2F, Nothing, (ᶜscalar,), true)
    test_op_matrix(GradientC2F, SetGradient, (ᶜscalar,))
    test_op_matrix(GradientF2C, Nothing, (ᶠscalar,))
    test_op_matrix(GradientF2C, SetValue, (ᶠscalar,))
    test_op_matrix(GradientF2C, SetGradient, (ᶠscalar,))
    test_op_matrix(DivergenceC2F, Nothing, (ᶜuvw,), true)
    test_op_matrix(DivergenceC2F, SetDivergence, (ᶜuvw,))
    test_op_matrix(DivergenceF2C, Nothing, (ᶠuvw,))
    test_op_matrix(DivergenceF2C, SetValue, (ᶠuvw,))
    test_op_matrix(DivergenceF2C, SetDivergence, (ᶠuvw,))
    test_op_matrix(CurlC2F, Nothing, (ᶜc12,), true)
    test_op_matrix(CurlC2F, SetCurl, (ᶜc12,))

    @test_throws "nonlinear" MatrixFields.operator_matrix(FCTBorisBook())
    @test_throws "nonlinear" MatrixFields.operator_matrix(FCTZalesak())
    @test_throws "nonlinear" MatrixFields.operator_matrix(
        LinVanLeerC2F(;
            constraint = ClimaCore.Operators.AlgebraicMean(),
        ),
    )
    @test_throws "nonlinear" MatrixFields.operator_matrix(
        TVDLimitedFluxC2F(;
            method = ClimaCore.Operators.MinModLimiter(),
        ),
    )
end

# Test the operator matrices' interior and boundary rows against hand-written
# stencils. The order is reduced to the number of in-range interior points.
@testset "Operator matrix rows against hand-written stencils" begin
    FT = Float64
    n = 5 # enough for 2 interior faces between the two boundary windows
    comms_ctx = ClimaComms.SingletonCommsContext(comms_device)
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(10));
        boundary_names = (:bottom, :top),
    )
    # Stretched mesh, so that the expected metric (J) factors are nontrivial.
    mesh = Meshes.IntervalMesh(
        domain,
        Meshes.ExponentialStretching(FT(5));
        nelems = n,
    )
    topology = Topologies.IntervalTopology(comms_ctx, mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(topology)
    face_space = Spaces.FaceFiniteDifferenceSpace(topology)

    # Sign-changing velocity, to exercise both upwind branches at interior and
    # boundary faces.
    ᶠz = Fields.coordinate_field(face_space).z
    ᶠw = @. Geometry.WVector(2 * cos(3 * ᶠz) + FT(1) / 10)
    ᶠv³ = vec(
        Array(
            parent(
                Geometry.contravariant3.(
                    ᶠw,
                    Fields.local_geometry_field(face_space),
                ),
            ),
        ),
    )
    ᶜJ = vec(Array(parent(Fields.local_geometry_field(center_space).J)))
    ᶠJ = vec(Array(parent(Fields.local_geometry_field(face_space).J)))

    # The single-column parent array has singleton horizontal dimensions;
    # flatten it to (level, band entry).
    level_by_entry(field) =
        reshape(Array(parent(field)), size(parent(field), 1), :)

    # Materializes the matrix of a one-argument operator as a field of rows by
    # multiplying it with the identity matrix on its input space.
    function matrix_rows(op, input_space)
        op_matrix = MatrixFields.operator_matrix(op)
        input_ones = ones(input_space)
        return level_by_entry(
            materialize(@lazy @. op_matrix() * DiagonalMatrixRow(input_ones)),
        )
    end

    # InterpolateC2F: interior faces average the two adjacent centers. With
    # Extrapolate, the boundary face copies the closest center; with SetValue
    # (or no boundary condition), the matrix's boundary rows are zero (the
    # value is imposed outside the matrix, by a SetBoundaryOperator).
    interp_interior = [f in (1, n + 1) ? [0, 0] : [0.5, 0.5] for f in 1:(n + 1)]
    @test matrix_rows(InterpolateC2F(), center_space) ≈
          stack(interp_interior; dims = 1)
    @test matrix_rows(
        InterpolateC2F(; bottom = SetValue(FT(0)), top = SetValue(FT(0))),
        center_space,
    ) ≈ stack(interp_interior; dims = 1)
    @test matrix_rows(
        InterpolateC2F(; bottom = Extrapolate(), top = Extrapolate()),
        center_space,
    ) ≈ stack(
        [f == 1 ? [0, 1] : f == n + 1 ? [1, 0] : [0.5, 0.5] for f in 1:(n + 1)];
        dims = 1,
    )

    # GradientF2C: G(x)[i] = (x[i+1/2] - x[i-1/2]) e³. Without boundary
    # conditions the boundary-face values of the input are used; SetValue
    # zeroes the fixed input's coefficient (the prescribed value's contribution
    # is affine, not linear).
    @test matrix_rows(GradientF2C(), face_space) ≈
          stack([[-1, 1] for i in 1:n]; dims = 1)
    @test matrix_rows(
        GradientF2C(; bottom = SetValue(FT(0)), top = SetValue(FT(0))),
        face_space,
    ) ≈ stack(
        [i == 1 ? [0, 1] : i == n ? [-1, 0] : [-1, 1] for i in 1:n];
        dims = 1,
    )

    # DivergenceF2C: D(v)[i] = (J v³[i+1/2] - J v³[i-1/2]) / J[i], with the
    # fixed input's coefficient zeroed under SetValue.
    div_bcs = (;
        bottom = SetValue(zero(Geometry.Contravariant3Vector{FT})),
        top = SetValue(zero(Geometry.Contravariant3Vector{FT})),
    )
    @test matrix_rows(DivergenceF2C(; div_bcs...), face_space) ≈ stack(
        [
            [
                i == 1 ? zero(FT) : -(ᶠJ[i] / ᶜJ[i]),
                i == n ? zero(FT) : ᶠJ[i + 1] / ᶜJ[i],
            ] for i in 1:n
        ];
        dims = 1,
    )

    # UpwindBiasedProductC2F: interior row ((v³ + |v³|) / 2, (v³ - |v³|) / 2).
    # Its stencil only reaches one ghost point, at each boundary face itself,
    # where a single interior point is in range, so every extrapolation order
    # gives the same boundary row: v³ times the closest center.
    for N in (0, 2)
        upwind1_op_matrix = MatrixFields.operator_matrix(
            UpwindBiasedProductC2F(;
                bottom = Extrapolate(N),
                top = Extrapolate(N),
            ),
        )
        upwind1_matrix = materialize(@lazy @. upwind1_op_matrix(ᶠw))
        @test level_by_entry(upwind1_matrix) ≈ stack(
            [
                let v = ᶠv³[f]
                    f == 1 ? [0, v] :
                    f == n + 1 ? [v, 0] :
                    [(v + abs(v)) / 2, (v - abs(v)) / 2]
                end for f in 1:(n + 1)
            ];
            dims = 1,
        )
    end

    # Upwind3rdOrderBiasedProductC2F: interior row
    # (-v - |v|, 7v + 3|v|, 7v - 3|v|, -v + |v|) / 12 over the 4 centers around
    # the face. The boundary rows below were derived by hand: substitute the
    # shared ghost value g = Σ wₖ xₖ (with the Extrapolate weights over the
    # in-range centers, ordered from the boundary outwards, and the order
    # reduced to the in-range count) into the interior stencil and collect the
    # coefficients of the in-range centers.
    function upwind3_expected_row(v, N, f)
        a = abs(v)
        if f == 1               # boundary face: 2 ghosts, 2 in range, order ≤ 1
            N == 0 ? [0, 0, 13v - a, -v + a] : [0, 0, 19v + a, -7v - a]
        elseif f == 2           # one-in face: 1 ghost, 3 in range
            N == 0 ? [0, 6v + 2a, 7v - 3a, -v + a] :
            N == 1 ? [0, 5v + a, 8v - 2a, -v + a] : [0, 4v, 10v, -2v]
        elseif f == n           # one-in face, top
            N == 0 ? [-v - a, 7v + 3a, 6v - 2a, 0] :
            N == 1 ? [-v - a, 8v + 2a, 5v - a, 0] : [-2v, 10v, 4v, 0]
        elseif f == n + 1       # boundary face, top
            N == 0 ? [-v - a, 13v + a, 0, 0] : [-7v + a, 19v - a, 0, 0]
        else                    # interior
            [-v - a, 7v + 3a, 7v - 3a, -v + a]
        end ./ 12
    end
    for N in 0:2
        upwind3_op_matrix = MatrixFields.operator_matrix(
            Upwind3rdOrderBiasedProductC2F(;
                bottom = Extrapolate(N),
                top = Extrapolate(N),
            ),
        )
        upwind3_matrix = materialize(@lazy @. upwind3_op_matrix(ᶠw))
        @test level_by_entry(upwind3_matrix) ≈ stack(
            [upwind3_expected_row(ᶠv³[f], N, f) for f in 1:(n + 1)];
            dims = 1,
        )
    end
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

    nested_zero = rzero(NestedType{FT})
    set_nested_values =
        (; bottom = SetValue(nested_zero), top = SetValue(nested_zero))
    c3_zero = zero(Geometry.Covariant3Vector{FT})
    set_c3_gradients = (; bottom = SetGradient(c3_zero), top = SetGradient(c3_zero))
    ct12_zero = zero(Geometry.Contravariant12Vector{FT})
    set_ct12_curls = (; bottom = SetCurl(ct12_zero), top = SetCurl(ct12_zero))

    ᶠinterp = InterpolateC2F(; set_nested_values...)
    ᶜlbias = LeftBiasedF2C()
    ᶠrbias = RightBiasedC2F(; set_nested_values.top)
    ᶜwinterp = WeightedInterpolateF2C()
    ᶠwinterp = WeightedInterpolateC2F(; set_nested_values...)
    ᶜrbias = RightBiasedF2C(; set_nested_values.top)
    ᶠupwind = UpwindBiasedProductC2F()
    ᶠgrad = GradientC2F(; set_c3_gradients...)
    ᶜdiv = DivergenceF2C()
    ᶠcurl = CurlC2F(; set_ct12_curls...)
    ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)
    ᶜlbias_matrix = MatrixFields.operator_matrix(ᶜlbias)
    ᶠrbias_matrix = MatrixFields.operator_matrix(ᶠrbias)
    ᶜwinterp_matrix = MatrixFields.operator_matrix(ᶜwinterp)
    ᶠwinterp_matrix = MatrixFields.operator_matrix(ᶠwinterp)
    ᶜrbias_matrix = MatrixFields.operator_matrix(ᶜrbias)
    ᶠupwind_matrix = MatrixFields.operator_matrix(ᶠupwind)
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
            @. ᶜrbias_matrix() *
               ᶠwinterp_matrix(ᶜscalar) *
               ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix()
        ),
        set_result = @lazy(
            @. ᶜrbias_matrix() *
               ᶠwinterp_matrix(ᶜscalar) *
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
            @. ᶜrbias_matrix() *
               ᶠwinterp_matrix(ᶜscalar) *
               ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix() *
               ᶜnested
        ),
        set_result = @lazy(
            @. ᶜrbias_matrix() *
               ᶠwinterp_matrix(ᶜscalar) *
               ᶜwinterp_matrix(ᶠscalar) *
               ᶠrbias_matrix() *
               ᶜlbias_matrix() *
               ᶠinterp_matrix() *
               ᶜnested
        ),
        ref_set_result = @lazy(
            @. ᶜrbias(
                ᶠwinterp(
                    ᶜscalar,
                    ᶜwinterp(ᶠscalar, ᶠrbias(ᶜlbias(ᶠinterp(ᶜnested)))),
                ),
            )
        ),
    )
    # This test will fail because of incorrect results, not InvalidIRError
    USING_CUDA || test_field_broadcast(;
        test_name = "applying six operators to a nested field using operator \
                     matrices, but with forced right associativity",
        get_result = @lazy(
            @. ᶜrbias_matrix() * (
                ᶠwinterp_matrix(ᶜscalar) * (
                    ᶜwinterp_matrix(ᶠscalar) * (
                        ᶠrbias_matrix() *
                        (ᶜlbias_matrix() * (ᶠinterp_matrix() * ᶜnested))
                    )
                )
            )
        ),
        set_result = @lazy(
            @. ᶜrbias_matrix() * (
                ᶠwinterp_matrix(ᶜscalar) * (
                    ᶜwinterp_matrix(ᶠscalar) * (
                        ᶠrbias_matrix() *
                        (ᶜlbias_matrix() * (ᶠinterp_matrix() * ᶜnested))
                    )
                )
            )
        ),
        ref_set_result = @lazy(
            @. ᶜrbias(
                ᶠwinterp(
                    ᶜscalar,
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
               (
                DiagonalMatrixRow(ᶜdiv(ᶠuvw)) -
                ᶜdiv_matrix() * ᶠupwind_matrix(ᶠuvw)
            ) / 5
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
                   (
                       DiagonalMatrixRow(ᶜdiv(ᶠuvw)) -
                       ᶜdiv_matrix() * ᶠupwind_matrix(ᶠuvw)
                   ) / 5
               )
        ),
    )
        materialize(get_result)
        @test_opt ignored_modules = CUDA_FRAMES materialize(get_result)
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
                    (
                        DiagonalMatrixRow(ᶜdiv(ᶠuvw)) -
                        ᶜdiv_matrix() * ᶠupwind_matrix(ᶠuvw)
                    ) / 5
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
                    (
                        DiagonalMatrixRow(ᶜdiv(ᶠuvw)) -
                        ᶜdiv_matrix() * ᶠupwind_matrix(ᶠuvw)
                    ) / 5
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
                       (
                           DiagonalMatrixRow(ᶜdiv(ᶠuvw)) -
                           ᶜdiv_matrix() * ᶠupwind_matrix(ᶠuvw)
                       ) / 5
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
                       (
                           DiagonalMatrixRow(ᶜdiv(ᶠuvw)) -
                           ᶜdiv_matrix() * ᶠupwind_matrix(ᶠuvw)
                       ) / 5
                   ) - (2I,)
               ) *
               ᶜscalar
        ),
        # ref_set_result = @lazy(@. ᶠupwind(
        #     ᶠuvw,
        #     ᶜdiv(
        #         ᶠscalar * ᶠgrad(
        #             (c12_b',) * ᶜwinterp(ᶠscalar, ᶠcurl((c12_a,) * ᶜscalar)) +
        #             (ᶜdiv(ᶠuvw) * ᶜscalar - ᶜdiv(ᶠupwind(ᶠuvw, ᶜscalar))) / 5,
        #         ),
        #     ) - 2 * ᶜscalar,
        # )),
        # max_eps_error_limit = 20, # This case's roundoff error is large.
    )
end
