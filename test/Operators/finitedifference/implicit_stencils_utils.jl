using Test
using ClimaComms
using Random: seed!
seed!(1) # ensures reproducibility

using ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces, Fields
using ClimaCore: Operators
import ClimaCore.Operators as OP

import ArgParse
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--float_type"
        help = "Float type"
        arg_type = String
        default = "Float32"
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return (s, parsed_args)
end

# Let stencil_op1 = Operator2Stencil(op1), stencil_op2 = Operator2Stencil(op2),
# apply = ApplyStencil(), and compose = ComposeStencils().

# op1.(a)[i] = ∑_j C[i, j] * a[j]                                            ==>
# op1.(a1 .* a0)[i] =
#   ∑_j C[i, j] * a1[j] * a0[j] =
#   ∑_j stencil_op1.(a1)[i, j] * a0[j] =
#   apply.(stencil_op1.(a1), a0)[i]                                          ==>
# op1.(a1 .* a0) = apply.(stencil_op1.(a1), a0)

# op1.(a)[i] = ∑_j C1[i, j] * a[j] and op2.(a)[i] = ∑_k C2[i, k] * a[k]      ==>
# op2.(a2 .* op1.(a1 .* a0))[i] =
#   ∑_k C2[i, k] * a2[k] * op1.(a1 .* a0)[k] =
#   ∑_k C2[i, k] * a2[k] * (∑_j C1[k, j] * a1[j] * a0[j]) =
#   ∑_j (∑_k C2[i, k] * a2[k] * C1[k, j] * a1[j]) * a0[j] =
#   ∑_j (∑_k stencil_op2.(a2)[i, k] * stencil_op1.(a1)[k, j]) * a0[j] =
#   ∑_j compose.(stencil_op2.(a2), stencil_op1.(a1))[i, j] * a0[j] =
#   apply.(compose.(stencil_op2.(a2), stencil_op1.(a1)), a0)[i]              ==>
# op2.(a2 .* op1.(a1 .* a0)) =
#   apply.(compose.(stencil_op2.(a2), stencil_op1.(a1)), a0)

struct CurriedTwoArgOperator{O, A}
    op::O
    arg2::A
end

Base.Broadcast.broadcasted(op::CurriedTwoArgOperator, arg1) =
    Base.Broadcast.broadcasted(op.op, arg1, op.arg2)

Operators.Operator2Stencil(op::CurriedTwoArgOperator) =
    CurriedTwoArgOperator(Operators.Operator2Stencil(op.op), op.arg2)

function get_space(::Type{FT}) where {FT}

    radius = FT(1e7)
    zmax = FT(1e4)
    helem = npoly = 2
    velem = 4

    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), hmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)

    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zero(FT)),
        Geometry.ZPoint{FT}(zmax);
        boundary_tags = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
    vspace = Spaces.CenterFiniteDifferenceSpace(vmesh)

    # TODO: Replace this with a space that includes topography.
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    return center_space
end

function get_all_ops(center_space)
    FT = Spaces.undertype(center_space)
    center_coords = Fields.coordinate_field(center_space)
    face_coords = Fields.coordinate_field(
        Spaces.FaceExtrudedFiniteDifferenceSpace(center_space),
    )

    # We can't use non-zero non-extrapolation BCs because Operator2Stencil
    # does not account for them (it only handles linear transformations).
    zero_scalar = OP.SetValue(zero(FT))
    zero_vector = OP.SetValue(Geometry.Covariant3Vector(zero(FT)))
    zero_grad = OP.SetGradient(Geometry.Covariant3Vector(zero(FT)))
    zero_div = OP.SetDivergence(zero(FT))
    zero_curl = OP.SetCurl(Geometry.Contravariant3Vector(zero(FT)))
    extrap = OP.Extrapolate()

    # `C` denotes "center" and `F` denotes "face"
    # `S` denotes "scalar" and `V` denotes "vector"

    rand_scalar(coord) = randn(Geometry.float_type(coord))
    rand_vector(coord) =
        Geometry.Covariant3Vector(randn(Geometry.float_type(coord)))
    a_CS = map(rand_scalar, center_coords)
    a_FS = map(rand_scalar, face_coords)
    a_CV = map(rand_vector, center_coords)
    a_FV = map(rand_vector, face_coords)

    ops_F2C_S2S = (
        OP.InterpolateF2C(),
        OP.LeftBiasedF2C(),
        OP.RightBiasedF2C(),
        OP.LeftBiasedF2C(bottom = zero_scalar),
        OP.RightBiasedF2C(top = zero_scalar),
    )
    ops_C2F_S2S = (
        OP.InterpolateC2F(bottom = extrap, top = extrap),
        OP.InterpolateC2F(bottom = zero_scalar, top = zero_scalar),
        OP.InterpolateC2F(bottom = zero_grad, top = zero_grad),
        OP.LeftBiasedC2F(bottom = zero_scalar),
        OP.RightBiasedC2F(top = zero_scalar),
    )
    ops_F2C_V2V = (
        OP.InterpolateF2C(),
        OP.LeftBiasedF2C(),
        OP.RightBiasedF2C(),
        OP.LeftBiasedF2C(bottom = zero_vector),
        OP.RightBiasedF2C(top = zero_vector),
        CurriedTwoArgOperator(
            OP.AdvectionC2C(bottom = zero_vector, top = zero_vector),
            a_CV,
        ),
        CurriedTwoArgOperator(
            OP.AdvectionC2C(bottom = extrap, top = extrap),
            a_CV,
        ),
        CurriedTwoArgOperator(
            OP.FluxCorrectionC2C(bottom = extrap, top = extrap),
            a_CV,
        ),
    )
    ops_C2F_V2V = (
        OP.InterpolateC2F(bottom = extrap, top = extrap),
        OP.InterpolateC2F(bottom = zero_vector, top = zero_vector),
        OP.LeftBiasedC2F(bottom = zero_vector),
        OP.RightBiasedC2F(top = zero_vector),
        CurriedTwoArgOperator(
            OP.FluxCorrectionF2F(bottom = extrap, top = extrap),
            a_FV,
        ),
        OP.CurlC2F(bottom = zero_vector, top = zero_vector),
        OP.CurlC2F(bottom = zero_curl, top = zero_curl),
    )
    ops_F2C_S2V = (
        OP.GradientF2C(),
        OP.GradientF2C(bottom = zero_scalar, top = zero_scalar),
    )
    ops_C2F_S2V = (
        OP.GradientC2F(bottom = zero_scalar, top = zero_scalar),
        OP.GradientC2F(bottom = zero_grad, top = zero_grad),
    )
    ops_F2C_V2S = (
        CurriedTwoArgOperator(
            OP.AdvectionC2C(bottom = zero_scalar, top = zero_scalar),
            a_CS,
        ),
        CurriedTwoArgOperator(
            OP.AdvectionC2C(bottom = extrap, top = extrap),
            a_CS,
        ),
        CurriedTwoArgOperator(
            OP.FluxCorrectionC2C(bottom = extrap, top = extrap),
            a_CS,
        ),
        OP.DivergenceF2C(),
        OP.DivergenceF2C(bottom = zero_vector, top = zero_vector),
    )
    ops_C2F_V2S = (
        CurriedTwoArgOperator(
            OP.FluxCorrectionF2F(bottom = extrap, top = extrap),
            a_FS,
        ),
        OP.DivergenceC2F(bottom = zero_vector, top = zero_vector),
        OP.DivergenceC2F(bottom = zero_div, top = zero_div),
    )
    return (;
        extrap,
        ops_F2C_S2S,
        ops_C2F_S2S,
        ops_F2C_V2V,
        ops_C2F_V2V,
        ops_F2C_S2V,
        ops_C2F_S2V,
        ops_F2C_V2S,
        ops_C2F_V2S,
        a_FS,
        a_CS,
        a_FV,
        a_CV,
    )
end

function test_pointwise_stencils_throws(all_ops)
    (; extrap, a_FS, a_FV) = all_ops
    # TODO: Make these test cases work.
    for (a, op) in (
        (a_FS, OP.GradientF2C(bottom = extrap, top = extrap)),
        (a_FV, OP.DivergenceF2C(bottom = extrap, top = extrap)),
    )
        @test_throws ArgumentError OP.Operator2Stencil(op).(a)
    end
end

#! format: off
function test_pointwise_stencils_apply(all_ops)
    (;
        ops_F2C_S2S,
        ops_C2F_S2S,
        ops_F2C_V2V,
        ops_C2F_V2V,
        ops_F2C_S2V,
        ops_C2F_S2V,
        ops_F2C_V2S,
        ops_C2F_V2S,
    ) = all_ops
    (; a_FS, a_CS, a_FV, a_CV) = all_ops
    # Manually unroll for better inference:
    @assert length(ops_F2C_S2S) == 5
    @assert length(ops_F2C_S2V) == 2
    @assert length(ops_C2F_S2S) == 5
    @assert length(ops_C2F_S2V) == 2
    @assert length(ops_F2C_V2V) == 8
    @assert length(ops_F2C_V2S) == 5
    @assert length(ops_C2F_V2V) == 7
    @assert length(ops_C2F_V2S) == 3

    Base.Cartesian.@nexprs 5 i -> apply_single(a_FS, a_FS, ops_F2C_S2S[i])
    Base.Cartesian.@nexprs 2 i -> apply_single(a_FS, a_FS, ops_F2C_S2V[i])
    Base.Cartesian.@nexprs 5 i -> apply_single(a_CS, a_CS, ops_C2F_S2S[i])
    Base.Cartesian.@nexprs 2 i -> apply_single(a_CS, a_CS, ops_C2F_S2V[i])
    Base.Cartesian.@nexprs 8 i -> apply_single(a_FS, a_FV, ops_F2C_V2V[i])
    Base.Cartesian.@nexprs 5 i -> apply_single(a_FS, a_FV, ops_F2C_V2S[i])
    Base.Cartesian.@nexprs 7 i -> apply_single(a_CS, a_CV, ops_C2F_V2V[i])
    Base.Cartesian.@nexprs 3 i -> apply_single(a_CS, a_CV, ops_C2F_V2S[i])
end
#! format: on

function apply_single(a0, a1, op1)
    apply = OP.ApplyStencil()
    compose = OP.ComposeStencils()
    stencil_op1 = OP.Operator2Stencil(op1)
    tested_value = apply.(stencil_op1.(a1), a0)
    ref_value = op1.(a1 .* a0)
    # @test tested_value ≈ ref_value atol = 1e-6
    return nothing
end

function get_tested_value(op1, op2, a0, a1, a2)
    apply = OP.ApplyStencil()
    compose = OP.ComposeStencils()
    stencil_op1 = OP.Operator2Stencil(op1)
    stencil_op2 = OP.Operator2Stencil(op2)
    tested_value = apply.(compose.(stencil_op2.(a2), stencil_op1.(a1)), a0)
    return tested_value
end
function get_ref_value(op1, op2, a0, a1, a2)
    apply = OP.ApplyStencil()
    compose = OP.ComposeStencils()
    stencil_op1 = OP.Operator2Stencil(op1)
    stencil_op2 = OP.Operator2Stencil(op2)
    ref_value = op2.(a2 .* op1.(a1 .* a0))
    return ref_value
end

#! format: off
function test_pointwise_stencils_compose(all_ops)
    (;
        ops_F2C_S2S,
        ops_C2F_S2S,
        ops_F2C_V2V,
        ops_C2F_V2V,
        ops_F2C_S2V,
        ops_C2F_S2V,
        ops_F2C_V2S,
        ops_C2F_V2S,
    ) = all_ops
    (; a_FS, a_CS, a_FV, a_CV) = all_ops
    @assert (length(ops_F2C_S2S), length(ops_C2F_S2S)) == (5, 5)
    @assert (length(ops_F2C_S2S), length(ops_C2F_S2V)) == (5, 2)
    @assert (length(ops_C2F_S2S), length(ops_F2C_S2S)) == (5, 5)
    @assert (length(ops_C2F_S2S), length(ops_F2C_S2V)) == (5, 2)
    @assert (length(ops_F2C_S2S), length(ops_C2F_V2V)) == (5, 7)
    @assert (length(ops_F2C_S2S), length(ops_C2F_V2S)) == (5, 3)
    @assert (length(ops_C2F_S2S), length(ops_F2C_V2V)) == (5, 8)
    @assert (length(ops_C2F_S2S), length(ops_F2C_V2S)) == (5, 5)
    @assert (length(ops_F2C_V2S), length(ops_C2F_S2S)) == (5, 5)
    @assert (length(ops_F2C_V2S), length(ops_C2F_S2V)) == (5, 2)
    @assert (length(ops_C2F_V2S), length(ops_F2C_S2S)) == (3, 5)
    @assert (length(ops_C2F_V2S), length(ops_F2C_S2V)) == (3, 2)
    @assert (length(ops_F2C_V2S), length(ops_C2F_V2V)) == (5, 7)
    @assert (length(ops_F2C_V2S), length(ops_C2F_V2S)) == (5, 3)
    @assert (length(ops_C2F_V2S), length(ops_F2C_V2V)) == (3, 8)
    @assert (length(ops_C2F_V2S), length(ops_F2C_V2S)) == (3, 5)

    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 5 j -> compose_single(a_FS, a_FS, a_CS, ops_F2C_S2S[i], ops_C2F_S2S[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 2 j -> compose_single(a_FS, a_FS, a_CS, ops_F2C_S2S[i], ops_C2F_S2V[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 5 j -> compose_single(a_CS, a_CS, a_FS, ops_C2F_S2S[i], ops_F2C_S2S[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 2 j -> compose_single(a_CS, a_CS, a_FS, ops_C2F_S2S[i], ops_F2C_S2V[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 7 j -> compose_single(a_FS, a_FS, a_CV, ops_F2C_S2S[i], ops_C2F_V2V[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 3 j -> compose_single(a_FS, a_FS, a_CV, ops_F2C_S2S[i], ops_C2F_V2S[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 8 j -> compose_single(a_CS, a_CS, a_FV, ops_C2F_S2S[i], ops_F2C_V2V[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 5 j -> compose_single(a_CS, a_CS, a_FV, ops_C2F_S2S[i], ops_F2C_V2S[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 5 j -> compose_single(a_FS, a_FV, a_CS, ops_F2C_V2S[i], ops_C2F_S2S[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 2 j -> compose_single(a_FS, a_FV, a_CS, ops_F2C_V2S[i], ops_C2F_S2V[j])
    Base.Cartesian.@nexprs 3 i -> Base.Cartesian.@nexprs 5 j -> compose_single(a_CS, a_CV, a_FS, ops_C2F_V2S[i], ops_F2C_S2S[j])
    Base.Cartesian.@nexprs 3 i -> Base.Cartesian.@nexprs 2 j -> compose_single(a_CS, a_CV, a_FS, ops_C2F_V2S[i], ops_F2C_S2V[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 7 j -> compose_single(a_FS, a_FV, a_CV, ops_F2C_V2S[i], ops_C2F_V2V[j])
    Base.Cartesian.@nexprs 5 i -> Base.Cartesian.@nexprs 3 j -> compose_single(a_FS, a_FV, a_CV, ops_F2C_V2S[i], ops_C2F_V2S[j])
    Base.Cartesian.@nexprs 3 i -> Base.Cartesian.@nexprs 8 j -> compose_single(a_CS, a_CV, a_FV, ops_C2F_V2S[i], ops_F2C_V2V[j])
    Base.Cartesian.@nexprs 3 i -> Base.Cartesian.@nexprs 5 j -> compose_single(a_CS, a_CV, a_FV, ops_C2F_V2S[i], ops_F2C_V2S[j])
end
#! format: on

function compose_single(a0, a1, a2, op1, op2)
    # stencil_op1 = OP.Operator2Stencil(op1)
    # stencil_op2 = OP.Operator2Stencil(op2)
    # test_op(op1, op2, a0, a1)
    # tested_value =
    #     apply.(compose.(stencil_op2.(a2), stencil_op1.(a1)), a0)
    # ref_value = op2.(a2 .* op1.(a1 .* a0))
    # @test tested_value ≈ ref_value atol = 1e-6
    tv = get_tested_value(op1, op2, a0, a1, a2)
    rv = get_ref_value(op1, op2, a0, a1, a2)
    # @test tv ≈ rv atol = 1e-6
    return nothing
end
