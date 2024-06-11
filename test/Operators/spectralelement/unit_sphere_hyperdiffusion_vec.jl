include("utils_sphere_hyperdiffusion_vec.jl")

@testset "hyperdiffusion on vectors" begin
    FT = Float64

    l = Int(7)
    m = Int(4)

    Ne = 6
    Nq = 7

    radius = FT(1)
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, Ne)
    grid_topology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
        mesh,
    )

    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

    # compute vector spherical harmonics (VSH) as the gradient of scalar spherical harmonics
    # eigenfunction properties of VSH: ∇⁴ VSH(l,m) = l²(l+1)² VSH(l,m)
    VSH_local = map(coords) do coord
        Geometry.UVVector(VSH(l, m, coord.lat, coord.long)...)
    end
    VSH_cov = Geometry.transform.(Ref(Geometry.Covariant12Axis()), VSH_local)

    # hyperdiffusion operator
    ∇⁴VSH_cov = ∇⁴(VSH_cov)
    ∇⁴VSH_local = Geometry.transform.(Ref(Geometry.UVAxis()), ∇⁴VSH_cov)

    # exact solution
    ∇⁴VSH_exact = @. l^2 * (l + 1)^2 * VSH_local

    @test ∇⁴VSH_local ≈ ∇⁴VSH_exact rtol = 2e-2
end
