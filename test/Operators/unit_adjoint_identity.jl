# Adjoint (integration-by-parts) identities between divergence and gradient:
#
#     ⟨f, div(u)⟩ = -⟨grad(f), u⟩
#
# with the L2 inner product ⟨a, b⟩ = ∫ a b = sum(a .* b).
#
# For spectral elements, the weak operators are defined by exactly this
# property: `WeakDivergence` satisfies ⟨φ, wdiv(u)⟩ = -⟨∇φ, u⟩ elementwise
# (the element boundary integrals are dropped), so the identity holds to
# roundoff on any mesh, with no DSS and no periodicity required.
#
# For the staggered finite difference pair, ⟨f, divf2c(u)⟩ on centers
# telescopes against -⟨gradc2f(f), u⟩ on faces whenever u vanishes on the
# boundary faces, so the identity again holds to roundoff.
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Fields,
    Geometry,
    Operators,
    Quadratures
import Random: seed!

@testset "SE weak divergence is minus the gradient's adjoint [$FT]" for FT in
                                                                        (
    Float32,
    Float64,
)
    context = ClimaComms.context()
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(-FT(π)),
            Geometry.XPoint(FT(π)),
            periodic = true,
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(-FT(π)),
            Geometry.YPoint(FT(π)),
            periodic = true,
        ),
    )
    mesh = Meshes.RectilinearMesh(domain, 4, 4)
    topology = Topologies.Topology2D(context, mesh)
    space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())

    seed!(1)
    f = Fields.Field(FT, space)
    u = Fields.Field(Geometry.UVVector{FT}, space)
    parent(f) .= rand.(FT)
    parent(u) .= rand.(FT)

    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()

    # ⟨f, wdiv(u)⟩ = -⟨∇f, u⟩, with the covariant gradient paired with u
    # through the UV basis.
    lhs = sum(f .* wdiv.(u))
    ∇f_uv = Geometry.UVVector.(grad.(f))
    rhs =
        -sum(
            @. ∇f_uv.components.data.:1 * u.components.data.:1 +
               ∇f_uv.components.data.:2 * u.components.data.:2
        )
    scale = sum(abs.(f .* wdiv.(u)))
    @test abs(lhs - rhs) <= 100 * eps(FT) * scale
end

@testset "FD staggered divf2c is minus gradc2f's adjoint [$FT]" for FT in (
    Float32,
    Float64,
)
    context = ClimaComms.context()
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(1));
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 16)
    topology = Topologies.IntervalTopology(context, mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(topology)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

    seed!(1)
    f = Fields.Field(FT, center_space)
    u = Fields.Field(Geometry.WVector{FT}, face_space)
    parent(f) .= rand.(FT)
    parent(u) .= rand.(FT)
    # The boundary terms of the summation by parts vanish iff u is zero on
    # the boundary faces.
    _u = Array(parent(u))
    _u[1] = 0
    _u[end] = 0
    parent(u) .= _u

    divf2c = Operators.DivergenceF2C()
    gradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
        top = Operators.SetGradient(Geometry.WVector(FT(0))),
    )

    lhs = sum(f .* divf2c.(u))
    ∇f = Geometry.WVector.(gradc2f.(f))
    rhs = -sum(@. ∇f.components.data.:1 * u.components.data.:1)
    scale = sum(abs.(f .* divf2c.(u)))
    @test abs(lhs - rhs) <= 100 * eps(FT) * scale
end
