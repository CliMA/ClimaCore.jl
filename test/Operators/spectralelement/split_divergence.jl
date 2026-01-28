using Test
using StaticArrays
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore:
    Geometry,
    Fields,
    Domains,
    Topologies,
    Meshes,
    Spaces,
    Operators,
    Quadratures
using LinearAlgebra, IntervalSets

import Random
Random.seed!(1234)

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

function SpectralElem2D(FT, context, Ne, Nq)
    # Doubly periodic Cartesian domain
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-pi) .. Geometry.XPoint{FT}(pi),
        Geometry.YPoint{FT}(-pi) .. Geometry.YPoint{FT}(pi);
        x1periodic = true,
        x2periodic = true,
        x1boundary = nothing,
        x2boundary = nothing,
    )
    quad = Quadratures.GLL{Nq}()
    mesh = Meshes.RectilinearMesh(domain, Ne, Ne)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, quad)
end

function SphereSpace(FT, context, Ne, Nq)
    radius = FT(1)
    domain = Domains.SphereDomain(radius)
    quad = Quadratures.GLL{Nq}()
    mesh = Meshes.EquiangularCubedSphere(domain, Ne)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, quad)
end

for FT in (Float32, Float64)
    @testset "split divergence (FT = $FT)" begin
        split_div = Operators.SplitDivergence()
        wdiv = Operators.WeakDivergence()
        wgrad = Operators.WeakGradient()
        div = Operators.Divergence()
        grad = Operators.Gradient()

        context = ClimaComms.context()
        Nq = 4 # number of quadrature points; polynomial order is Nq - 1

        @testset "consistency and conservation" begin
            Ne = 6 # number of horizontal elements
            for HorizontalSpace in (SpectralElem2D, SphereSpace)
                space = HorizontalSpace(FT, context, Ne, Nq)
                psi = ones(space)
                u = similar(psi, Geometry.UVVector{FT})
                parent(psi) .= rand.(FT)
                parent(u) .= rand.(FT)

                # Test consistency with full forms of split divergence operator
                uʰ = Geometry.Contravariant12Vector.(u)
                full_form_div_1 =
                    @. (wdiv(u * psi) + psi * wdiv(u) + dot(uʰ, grad(psi))) / 2
                full_form_div_2 =
                    @. (wdiv(u * psi) + psi * div(u) + dot(uʰ, wgrad(psi))) / 2
                @test norm(split_div.(u, psi) .- full_form_div_1) < 40 * eps(FT)
                @test norm(split_div.(u, psi) .- full_form_div_2) < 40 * eps(FT)

                # Test conservation in comparison to weak divergence operator
                @test abs(sum(split_div.(u, psi))) < 200 * eps(FT)
                @test abs(sum(wdiv.(u .* psi))) < 10 * eps(FT)
            end
        end

        @testset "convergence" begin
            # Solid body rotation should be divergence-free after applying DSS,
            # with the error converging to 0 as resolution increases.
            Nes = [6, 12]
            errors = map(Nes) do Ne
                sphere_space = SphereSpace(FT, context, Ne, Nq)
                sphere_coords = Fields.coordinate_field(sphere_space)
                u_solid_body = Geometry.UVVector.(cosd.(sphere_coords.lat), 0)
                div_error = split_div.(u_solid_body, 1) # use constant psi = 1
                Spaces.weighted_dss!(div_error)
                norm(div_error)
            end
            convergence_rate = log(errors[1] / errors[2]) / log(Nes[2] / Nes[1])
            @test convergence_rate ≈ Nq - 1 rtol = 0.02
        end
    end
end
