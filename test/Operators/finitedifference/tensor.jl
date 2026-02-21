using Test

using ClimaComms
using ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Fields,
    Operators,
    Quadratures
using ClimaCore.Geometry
using LinearAlgebra

for FT in (Float32, Float64)
    context = ClimaComms.context()
    hdomain = Domains.SphereDomain{FT}(6.37122e6)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, 30)
    htopology = Topologies.Topology2D(context, hmesh)
    hspace = Spaces.SpectralElementSpace2D(htopology, Quadratures.GLL{4}())

    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(30e3);
        boundary_names = (:bottom, :top),
    )
    stretch = Meshes.GeneralizedExponentialStretching(FT(30), FT(5000))
    vmesh = Meshes.IntervalMesh(vdomain, stretch; nelems = 45)
    vtopology = Topologies.IntervalTopology(context, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

    cspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)

    uvw = map(Fields.coordinate_field(cspace)) do coord
        Geometry.UVWVector(
            Geometry.float_type(coord)(1 * coord.z),
            Geometry.float_type(coord)(2 * coord.z),
            Geometry.float_type(coord)(3 * coord.z),
        )
    end

    ∇ᵥuvw_boundary =
        Geometry.WVector(FT(1)) ⊗ Geometry.UVWVector(FT(1), FT(2), FT(3))

    gradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(∇ᵥuvw_boundary),
        top = Operators.SetGradient(∇ᵥuvw_boundary),
    )
    ∇ᵥuvw = Geometry.project.(Ref(Geometry.UVWAxis()), gradc2f.(uvw))
    ∇ᵥuvw_scalar =
        Geometry.UVWVector(FT(0), FT(0), FT(1)) ⊗
        Geometry.UVWVector(FT(1), FT(2), FT(3))
    ∇ᵥuvw_ref = fill(∇ᵥuvw_scalar, fspace)
    @test ∇ᵥuvw ≈ ∇ᵥuvw_ref

    S = (∇ᵥuvw .+ adjoint.(∇ᵥuvw)) ./ 2
    S_scalar = (∇ᵥuvw_scalar + adjoint(∇ᵥuvw_scalar)) / 2
    S_ref = fill(S_scalar, fspace)
    @test S ≈ S_ref

    @test norm(S_scalar) ≈ norm(Geometry.components(S_scalar))
    @test norm.(S) ≈ fill(norm(S_scalar), fspace)

    divf2c = Operators.DivergenceF2C()
    divS = Geometry.Covariant123Vector.(divf2c.(S))

    divS_ref = fill(Geometry.Covariant123Vector(FT(0), FT(0), FT(0)), cspace)
    @test divS ≈ divS_ref atol = eps(FT)
end
