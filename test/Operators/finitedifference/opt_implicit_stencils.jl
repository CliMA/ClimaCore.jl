using Test
using ClimaComms
using Random: seed!

using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Fields, Quadratures
using ClimaCore: Operators

struct CurriedTwoArgOperator{O, A}
    op::O
    arg2::A
end

Base.Broadcast.broadcasted(op::CurriedTwoArgOperator, arg1) =
    Base.Broadcast.broadcasted(op.op, arg1, op.arg2)

Operators.Operator2Stencil(op::CurriedTwoArgOperator) =
    CurriedTwoArgOperator(Operators.Operator2Stencil(op.op), op.arg2)

#! format: off
@testset "Implicit Stencil optimization" begin
    seed!(1) # ensures reproducibility

    FT = Float64
    radius = FT(1e7)
    zmax = FT(1e4)
    helem = npoly = 2
    velem = 4

    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), hmesh)
    quad = Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)

    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zero(FT)),
        Geometry.ZPoint{FT}(zmax);
        boundary_names = (:bottom, :top),
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


    apply_configs = (
        (a_FS, a_FS, (ops_F2C_S2S..., ops_F2C_S2V...)),
        (a_CS, a_CS, (ops_C2F_S2S..., ops_C2F_S2V...)),
        (a_FS, a_FV, (ops_F2C_V2V..., ops_F2C_V2S...)),
        (a_CS, a_CV, (ops_C2F_V2V..., ops_C2F_V2S...)),
    )

    compressed_keys(op) = nameof(typeof(op))

    function apply_single_stencil(a0, a1, op1)
        apply = Operators.ApplyStencil()
        stencil_op1 = Operators.Operator2Stencil(op1)
        applied_stencil_op = apply.(stencil_op1.(a1), a0)
        p = @allocated @. applied_stencil_op = apply.(stencil_op1.(a1), a0)
        return p
    end

    function apply_all_stencils(apply_configs)
        success_tabs = Dict()
        for (a0, a1, op1s) in apply_configs
            for op1 in op1s
                key = compressed_keys(op1)
                alloc = apply_single_stencil(a0, a1, op1)
                if haskey(success_tabs, key)
                    success_tabs[key] += alloc
                else
                    success_tabs[key] = alloc
                end
            end
        end
        return success_tabs
    end

    success_tabs = apply_all_stencils(apply_configs)

    @test success_tabs[:RightBiasedC2F] == 0
    @test success_tabs[:InterpolateC2F] == 0
    @test success_tabs[:CurriedTwoArgOperator] == 0
    @test success_tabs[:GradientF2C] == 0
    @test success_tabs[:DivergenceF2C] == 0
    @test success_tabs[:LeftBiasedC2F] == 0
    @test success_tabs[:DivergenceC2F] == 0
    @test success_tabs[:GradientC2F] == 0
    @test success_tabs[:CurlC2F] == 0
    @test success_tabs[:InterpolateF2C] == 0
    @test success_tabs[:LeftBiasedF2C] == 0
    @test success_tabs[:RightBiasedF2C] == 0

    # for k in keys(success_tabs) # for debugging
    #     @show k, success_tabs[k]
    # end

    apply_configs = (
        (a_FS, a_FS, a_CS, ops_F2C_S2S, (ops_C2F_S2S..., ops_C2F_S2V...)),
        (a_CS, a_CS, a_FS, ops_C2F_S2S, (ops_F2C_S2S..., ops_F2C_S2V...)),
        (a_FS, a_FS, a_CV, ops_F2C_S2S, (ops_C2F_V2V..., ops_C2F_V2S...)),
        (a_CS, a_CS, a_FV, ops_C2F_S2S, (ops_F2C_V2V..., ops_F2C_V2S...)),
        (a_FS, a_FV, a_CS, ops_F2C_V2S, (ops_C2F_S2S..., ops_C2F_S2V...)),
        (a_CS, a_CV, a_FS, ops_C2F_V2S, (ops_F2C_S2S..., ops_F2C_S2V...)),
        (a_FS, a_FV, a_CV, ops_F2C_V2S, (ops_C2F_V2V..., ops_C2F_V2S...)),
        (a_CS, a_CV, a_FV, ops_C2F_V2S, (ops_F2C_V2V..., ops_F2C_V2S...)),
    )

    compressed_keys(op1, op2) = (nameof(typeof(op1)), nameof(typeof(op2)))

    function apply_single_composed_stencils(a0, a1, a2, op1, op2)
        compose = Operators.ComposeStencils()
        apply = Operators.ApplyStencil()
        stencil_op1 = Operators.Operator2Stencil(op1)
        stencil_op2 = Operators.Operator2Stencil(op2)
        applied_stencil_op = apply.(compose.(stencil_op2.(a2), stencil_op1.(a1)), a0) # compile
        p = @allocated @. applied_stencil_op = apply.(compose.(stencil_op2.(a2), stencil_op1.(a1)), a0)
        return p
    end

    function apply_composed_stencils(apply_configs)
        success_compose_tabs = Dict()
        for (a0, a1, a2, op1s, op2s) in apply_configs
            for op1 in op1s
                for op2 in op2s
                    key = compressed_keys(op1, op2)
                    alloc = apply_single_composed_stencils(a0, a1, a2, op1, op2)
                    if haskey(success_compose_tabs, key)
                        success_compose_tabs[key] += alloc
                    else
                        success_compose_tabs[key] = alloc
                    end
                end
            end
        end
        return success_compose_tabs
    end
    success_compose_tabs = apply_composed_stencils(apply_configs)

    @test success_compose_tabs[(:CurriedTwoArgOperator, :DivergenceF2C)] == 0
    @test success_compose_tabs[(:InterpolateC2F, :GradientF2C)] == 0
    @test success_compose_tabs[(:RightBiasedF2C, :LeftBiasedC2F)] == 0
    @test success_compose_tabs[(:DivergenceF2C, :RightBiasedC2F)] == 0
    @test success_compose_tabs[(:DivergenceF2C, :CurriedTwoArgOperator)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :CurriedTwoArgOperator)] == 0
    @test success_compose_tabs[(:DivergenceC2F, :LeftBiasedF2C)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :RightBiasedF2C)] == 0
    @test success_compose_tabs[(:RightBiasedF2C, :InterpolateC2F)] == 0
    @test success_compose_tabs[(:LeftBiasedF2C, :InterpolateC2F)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :CurlC2F)] == 0
    @test success_compose_tabs[(:DivergenceC2F, :GradientF2C)] == 0
    @test success_compose_tabs[(:DivergenceC2F, :InterpolateF2C)] == 0
    @test success_compose_tabs[(:InterpolateF2C, :InterpolateC2F)] == 0
    @test success_compose_tabs[(:LeftBiasedF2C, :LeftBiasedC2F)] == 0
    @test success_compose_tabs[(:RightBiasedF2C, :CurriedTwoArgOperator)] == 0
    @test success_compose_tabs[(:RightBiasedC2F, :LeftBiasedF2C)] == 0
    @test success_compose_tabs[(:RightBiasedC2F, :CurriedTwoArgOperator)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :InterpolateF2C)] == 0
    @test success_compose_tabs[(:RightBiasedF2C, :GradientC2F)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :GradientC2F)] == 0
    @test success_compose_tabs[(:DivergenceC2F, :RightBiasedF2C)] == 0
    @test success_compose_tabs[(:InterpolateF2C, :LeftBiasedC2F)] == 0
    @test success_compose_tabs[(:InterpolateC2F, :DivergenceF2C)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :RightBiasedC2F)] == 0
    @test success_compose_tabs[(:LeftBiasedC2F, :GradientF2C)] == 0
    @test success_compose_tabs[(:LeftBiasedC2F, :InterpolateF2C)] == 0
    @test success_compose_tabs[(:DivergenceF2C, :GradientC2F)] == 0
    @test success_compose_tabs[(:DivergenceC2F, :CurriedTwoArgOperator)] == 0
    @test success_compose_tabs[(:DivergenceF2C, :DivergenceC2F)] == 0
    @test success_compose_tabs[(:InterpolateF2C, :GradientC2F)] == 0
    @test success_compose_tabs[(:LeftBiasedC2F, :RightBiasedF2C)] == 0
    @test success_compose_tabs[(:DivergenceF2C, :CurlC2F)] == 0
    @test success_compose_tabs[(:RightBiasedC2F, :DivergenceF2C)] == 0
    @test success_compose_tabs[(:LeftBiasedC2F, :LeftBiasedF2C)] == 0
    @test success_compose_tabs[(:InterpolateC2F, :LeftBiasedF2C)] == 0
    @test success_compose_tabs[(:RightBiasedF2C, :CurlC2F)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :LeftBiasedF2C)] == 0
    @test success_compose_tabs[(:LeftBiasedF2C, :GradientC2F)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :LeftBiasedC2F)] == 0
    @test success_compose_tabs[(:InterpolateF2C, :RightBiasedC2F)] == 0
    @test success_compose_tabs[(:InterpolateF2C, :DivergenceC2F)] == 0
    @test success_compose_tabs[(:RightBiasedF2C, :DivergenceC2F)] == 0
    @test success_compose_tabs[(:RightBiasedC2F, :InterpolateF2C)] == 0
    @test success_compose_tabs[(:LeftBiasedF2C, :DivergenceC2F)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :InterpolateC2F)] == 0
    @test success_compose_tabs[(:InterpolateC2F, :InterpolateF2C)] == 0
    @test success_compose_tabs[(:LeftBiasedF2C, :RightBiasedC2F)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :DivergenceC2F)] == 0
    @test success_compose_tabs[(:DivergenceF2C, :LeftBiasedC2F)] == 0
    @test success_compose_tabs[(:RightBiasedC2F, :RightBiasedF2C)] == 0
    @test success_compose_tabs[(:InterpolateF2C, :CurlC2F)] == 0
    @test success_compose_tabs[(:LeftBiasedF2C, :CurlC2F)] == 0
    @test success_compose_tabs[(:InterpolateC2F, :RightBiasedF2C)] == 0
    @test success_compose_tabs[(:LeftBiasedF2C, :CurriedTwoArgOperator)] == 0
    @test success_compose_tabs[(:LeftBiasedC2F, :DivergenceF2C)] == 0
    @test success_compose_tabs[(:DivergenceF2C, :InterpolateC2F)] == 0
    @test success_compose_tabs[(:InterpolateF2C, :CurriedTwoArgOperator)] == 0
    @test success_compose_tabs[(:InterpolateC2F, :CurriedTwoArgOperator)] == 0
    @test success_compose_tabs[(:LeftBiasedC2F, :CurriedTwoArgOperator)] == 0
    @test success_compose_tabs[(:RightBiasedC2F, :GradientF2C)] == 0
    @test success_compose_tabs[(:CurriedTwoArgOperator, :GradientF2C)] == 0
    @test success_compose_tabs[(:RightBiasedF2C, :RightBiasedC2F)] == 0
    @test success_compose_tabs[(:DivergenceC2F, :DivergenceF2C)] == 0

    # for k in keys(success_tabs) # for debugging
    #     @show k, success_tabs[k]
    # end

end
#! format: on
