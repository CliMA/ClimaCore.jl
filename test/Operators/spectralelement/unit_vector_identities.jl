using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    Geometry,
    Operators

import ClimaCore  # for `pkgdir` below
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

@testset "Spectral Element Vector Identities & Integration by Parts" begin
    device = ClimaComms.device()
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext(device)
        radius = FT(6.371e6)
        domain = Domains.SphereDomain(radius)
        mesh = Meshes.EquiangularCubedSphere(domain, 4)
        topology = Topologies.Topology2D(context, mesh)
        quad = Quadratures.GLL{4}()
        space = Spaces.SpectralElementSpace2D(topology, quad)

        coords = Fields.coordinate_field(space)

        # Smooth scalar and vector fields
        f = @. sind(coords.long) * cosd(coords.lat)^2
        u_uv = @. Geometry.UVVector(
            cosd(coords.long) * cosd(coords.lat),
            -sind(coords.long) * sind(coords.lat),
        )
        u_contra = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), u_uv)

        grad_op = Operators.Gradient()
        div_op = Operators.Divergence()
        curl_op = Operators.Curl()
        wdiv_op = Operators.Divergence{Operators.WeakForm}()

        # 1. curl(grad(f)) == 0
        grad_f = grad_op.(f)
        curl_grad_f = curl_op.(grad_f)
        max_curl_grad = maximum(abs, parent(curl_grad_f))
        max_f = maximum(abs, parent(f))
        @test max_curl_grad / max_f < 1000 * eps(FT)

        # 2. div(curl(v)) == 0 (where v is a Covariant3Vector on 2D surface)
        v3 = @. Geometry.Covariant3Vector(sind(coords.long) * cosd(coords.lat))
        curl_v3 = curl_op.(v3)
        div_curl_v3 =
            div_op.(Geometry.transform.(Ref(Geometry.Contravariant12Axis()), curl_v3))
        max_div_curl = maximum(abs, parent(div_curl_v3))
        @test max_div_curl < 1000 * eps(FT)

        # 3. Discrete Integration by Parts for Divergence: <div(u), f> = - <u, grad(f)>
        grad_f = grad_op.(f)
        u_dot_grad_f = @. Geometry.dot(u_contra, grad_f)
        lhs_div = sum(@. div_op(u_contra) * f)
        rhs_div = sum(@. -(u_dot_grad_f))
        @test isapprox(lhs_div, rhs_div, rtol = (FT == Float32 ? 2e-2 : 1e-2))

        # 4. Gradient of a constant field is zero up to machine precision
        grad_const = grad_op.(ones(space))
        max_grad_const = maximum(abs, parent(grad_const))
        @test max_grad_const < 10 * eps(FT)

        # 5. In-place zero allocation sentinel
        target_grad = similar(grad_f)
        @noinline function _eval_grad!(tgt, op, src)
            @. tgt = op(src)
            return nothing
        end
        _eval_grad!(target_grad, grad_op, f)
        if device isa ClimaComms.CPUSingleThreaded
            allocs = @allocated _eval_grad!(target_grad, grad_op, f)
            TU.allocation_checks_meaningful() && @test allocs == 0
        end
    end

end
