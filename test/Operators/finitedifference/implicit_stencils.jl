using Test
using ClimaComms
using Random: seed!

using ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces, Fields
using ClimaCore: Operators

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

(s, parsed_args) = parse_commandline()

const FT = parsed_args["float_type"] == "Float64" ? Float64 : Float32

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

@testset "Pointwise Stencil Construction/Composition/Application" begin
    seed!(1) # ensures reproducibility

    radius = FT(1e7)
    zmax = FT(1e4)
    helem = npoly = 2
    velem = 4

    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(),
        hmesh,
    )
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
    center_coords = Fields.coordinate_field(center_space)
    face_coords = Fields.coordinate_field(
        Spaces.FaceExtrudedFiniteDifferenceSpace(center_space),
    )

    # We can't use non-zero non-extrapolation BCs because Operator2Stencil
    # does not account for them (it only handles linear transformations).
    zero_scalar = Operators.SetValue(zero(FT))
    zero_vector = Operators.SetValue(Geometry.Covariant3Vector(zero(FT)))
    zero_grad = Operators.SetGradient(Geometry.Covariant3Vector(zero(FT)))
    zero_div = Operators.SetDivergence(zero(FT))
    zero_curl = Operators.SetCurl(Geometry.Contravariant3Vector(zero(FT)))
    extrap = Operators.Extrapolate()

    # `C` denotes "center" and `F` denotes "face"
    # `S` denotes "scalar" and `V` denotes "vector"

    rand_scalar(coord) = randn(Geometry.float_type(coord))
    rand_vector(coord) =
        Geometry.Covariant3Vector(randn(Geometry.float_type(coord)))
    a_CS = map(rand_scalar, center_coords)
    a_FS = map(rand_scalar, face_coords)
    a_CV = map(rand_vector, center_coords)
    a_FV = map(rand_vector, face_coords)

    ops_F2C_⍰2⍰ = (
        Operators.InterpolateF2C(),
        Operators.LeftBiasedF2C(),
        Operators.RightBiasedF2C(),
    )
    ops_C2F_⍰2⍰ = (Operators.InterpolateC2F(bottom = extrap, top = extrap),)
    ops_F2C_S2S = (
        ops_F2C_⍰2⍰...,
        Operators.LeftBiasedF2C(bottom = zero_scalar),
        Operators.RightBiasedF2C(top = zero_scalar),
    )
    ops_C2F_S2S = (
        ops_C2F_⍰2⍰...,
        Operators.InterpolateC2F(bottom = zero_scalar, top = zero_scalar),
        Operators.InterpolateC2F(bottom = zero_grad, top = zero_grad),
        Operators.LeftBiasedC2F(bottom = zero_scalar),
        Operators.RightBiasedC2F(top = zero_scalar),
    )
    ops_F2C_V2V = (
        ops_F2C_⍰2⍰...,
        Operators.LeftBiasedF2C(bottom = zero_vector),
        Operators.RightBiasedF2C(top = zero_vector),
        CurriedTwoArgOperator(
            Operators.AdvectionC2C(bottom = zero_vector, top = zero_vector),
            a_CV,
        ),
        CurriedTwoArgOperator(
            Operators.AdvectionC2C(bottom = extrap, top = extrap),
            a_CV,
        ),
        CurriedTwoArgOperator(
            Operators.FluxCorrectionC2C(bottom = extrap, top = extrap),
            a_CV,
        ),
    )
    ops_C2F_V2V = (
        ops_C2F_⍰2⍰...,
        Operators.InterpolateC2F(bottom = zero_vector, top = zero_vector),
        Operators.LeftBiasedC2F(bottom = zero_vector),
        Operators.RightBiasedC2F(top = zero_vector),
        CurriedTwoArgOperator(
            Operators.FluxCorrectionF2F(bottom = extrap, top = extrap),
            a_FV,
        ),
        Operators.CurlC2F(bottom = zero_vector, top = zero_vector),
        Operators.CurlC2F(bottom = zero_curl, top = zero_curl),
    )
    ops_F2C_S2V = (
        Operators.GradientF2C(),
        Operators.GradientF2C(bottom = zero_scalar, top = zero_scalar),
    )
    ops_C2F_S2V = (
        Operators.GradientC2F(bottom = zero_scalar, top = zero_scalar),
        Operators.GradientC2F(bottom = zero_grad, top = zero_grad),
    )
    ops_F2C_V2S = (
        CurriedTwoArgOperator(
            Operators.AdvectionC2C(bottom = zero_scalar, top = zero_scalar),
            a_CS,
        ),
        CurriedTwoArgOperator(
            Operators.AdvectionC2C(bottom = extrap, top = extrap),
            a_CS,
        ),
        CurriedTwoArgOperator(
            Operators.FluxCorrectionC2C(bottom = extrap, top = extrap),
            a_CS,
        ),
        Operators.DivergenceF2C(),
        Operators.DivergenceF2C(bottom = zero_vector, top = zero_vector),
    )
    ops_C2F_V2S = (
        CurriedTwoArgOperator(
            Operators.FluxCorrectionF2F(bottom = extrap, top = extrap),
            a_FS,
        ),
        Operators.DivergenceC2F(bottom = zero_vector, top = zero_vector),
        Operators.DivergenceC2F(bottom = zero_div, top = zero_div),
    )

    # TODO: Make these test cases work.
    for (a, op) in (
        (a_FS, Operators.GradientF2C(bottom = extrap, top = extrap)),
        (a_FV, Operators.DivergenceF2C(bottom = extrap, top = extrap)),
    )
        @test_throws ArgumentError Operators.Operator2Stencil(op).(a)
    end

    apply = Operators.ApplyStencil()
    compose = Operators.ComposeStencils()
    for (a0, a1, op1s) in (
        (a_FS, a_FS, (ops_F2C_S2S..., ops_F2C_S2V...)),
        (a_CS, a_CS, (ops_C2F_S2S..., ops_C2F_S2V...)),
        (a_FS, a_FV, (ops_F2C_V2V..., ops_F2C_V2S...)),
        (a_CS, a_CV, (ops_C2F_V2V..., ops_C2F_V2S...)),
    )
        for op1 in op1s
            stencil_op1 = Operators.Operator2Stencil(op1)
            tested_value = apply.(stencil_op1.(a1), a0)
            @test tested_value ≈ op1.(a1 .* a0) atol = 1e-6
        end
    end
    for (a0, a1, a2, op1s, op2s) in (
        (a_FS, a_FS, a_CS, ops_F2C_S2S, (ops_C2F_S2S..., ops_C2F_S2V...)),
        (a_CS, a_CS, a_FS, ops_C2F_S2S, (ops_F2C_S2S..., ops_F2C_S2V...)),
        (a_FS, a_FS, a_CV, ops_F2C_S2S, (ops_C2F_V2V..., ops_C2F_V2S...)),
        (a_CS, a_CS, a_FV, ops_C2F_S2S, (ops_F2C_V2V..., ops_F2C_V2S...)),
        (a_FS, a_FV, a_CS, ops_F2C_V2S, (ops_C2F_S2S..., ops_C2F_S2V...)),
        (a_CS, a_CV, a_FS, ops_C2F_V2S, (ops_F2C_S2S..., ops_F2C_S2V...)),
        (a_FS, a_FV, a_CV, ops_F2C_V2S, (ops_C2F_V2V..., ops_C2F_V2S...)),
        (a_CS, a_CV, a_FV, ops_C2F_V2S, (ops_F2C_V2V..., ops_F2C_V2S...)),
    )
        for op1 in op1s
            for op2 in op2s
                stencil_op1 = Operators.Operator2Stencil(op1)
                stencil_op2 = Operators.Operator2Stencil(op2)
                tested_value =
                    apply.(compose.(stencil_op2.(a2), stencil_op1.(a1)), a0)
                @test tested_value ≈ op2.(a2 .* op1.(a1 .* a0)) atol = 1e-6
            end
        end
    end
end
