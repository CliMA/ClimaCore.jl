include("utils_sphere_hyperdiffusion_vec.jl")

convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

@testset "convergence tests for vector hyperdiffusions on the sphere" begin
    FT = Float64

    Nes = [4, 8, 16]
    Nqs = [4, 5, 6, 7, 8, 9, 10]

    l = Int(7)
    m = Int(4)

    radius = FT(1)
    domain = Domains.SphereDomain(radius)
    for (Iq, Nq) in enumerate(Nqs)
        err_∇⁴ = Array{FT}(undef, length(Nes))
        Δh = Array{FT}(undef, length(Nes))

        for (Ie, Ne) in enumerate(Nes)
            mesh = Meshes.EquiangularCubedSphere(domain, Ne)
            grid_topology = Topologies.Topology2D(
                ClimaComms.SingletonCommsContext(
                    ClimaComms.CPUSingleThreaded(),
                ),
                mesh,
            )

            quad = Quadratures.GLL{Nq}()
            space = Spaces.SpectralElementSpace2D(grid_topology, quad)
            coords = Fields.coordinate_field(space)

            VSH_local = map(coords) do coord
                Geometry.UVVector(VSH(l, m, coord.lat, coord.long)...)
            end
            VSH_cov =
                Geometry.transform.(Ref(Geometry.Covariant12Axis()), VSH_local)
            ∇⁴VSH_cov = ∇⁴(VSH_cov)
            ∇⁴VSH_local = Geometry.transform.(Ref(Geometry.UVAxis()), ∇⁴VSH_cov)
            ∇⁴VSH_exact = @. l^2 * (l + 1)^2 * VSH_local

            err_∇⁴[Ie] = norm(∇⁴VSH_local .- ∇⁴VSH_exact)
            Δh[Ie] = 1 / Ne
        end

        convergence_rate_∇⁴ = convergence_rate(err_∇⁴, Δh)
        for Ie in range(1, length = length(Nes) - 1)
            @test convergence_rate_∇⁴[Ie] > (Nq - 4)
        end
    end
end
