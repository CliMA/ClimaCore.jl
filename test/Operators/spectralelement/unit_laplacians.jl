# Tests the horizontal Laplacian atoms against the equivalent hand-written
# operator compositions, on CG spaces with two and one horizontal dimensions
# and on a DG sphere (where the scalar atom must match the interior-penalty
# Laplacian and the vector atom must fail loudly).
using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

FT = Float64
context = ClimaComms.context()
materialize = Base.Broadcast.materialize

@testset "scalar_laplacian on a CG sphere" begin
    space = TU.SphereSpectralElementSpace(FT; context)
    coord = Fields.coordinate_field(space)
    f = @. sind(coord.long) * cosd(coord.lat)
    ρ = @. 2 + sind(coord.lat)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    grad = Operators.Gradient()

    expected = @. wdiv(grad(f))
    got = materialize(Operators.scalar_laplacian(f))
    @test parent(got) ≈ parent(expected)

    expected_weighted = @. wdiv(ρ * grad(f))
    got_weighted = materialize(Operators.scalar_laplacian(f; weight = ρ))
    @test parent(got_weighted) ≈ parent(expected_weighted)

    # The lazy result fuses into a consuming broadcast.
    out = similar(f)
    out .= 3 .* Operators.scalar_laplacian(f)
    @test parent(out) ≈ 3 .* parent(expected)
end

@testset "vector_laplacian with two horizontal dimensions" begin
    space = TU.SphereSpectralElementSpace(FT; context)
    coord = Fields.coordinate_field(space)
    u = @. Geometry.Covariant12Vector(
        Geometry.UVVector(cosd(coord.lat), sind(coord.long) * cosd(coord.lat)),
    )
    div = Operators.Divergence()
    wgrad = Operators.Gradient{Operators.WeakForm}()
    curl = Operators.Curl()
    wcurl = Operators.Curl{Operators.WeakForm}()
    C12 = Geometry.Covariant12Vector
    C3 = Geometry.Covariant3Vector

    expected = @. wgrad(div(u)) - C12(wcurl(C3(curl(u))))
    got = materialize(Operators.vector_laplacian(u))
    @test parent(got) ≈ parent(expected)

    factor = FT(1 / 2)
    expected_damped = @. factor * wgrad(div(u)) - C12(wcurl(C3(curl(u))))
    got_damped =
        materialize(Operators.vector_laplacian(u; divergence_factor = factor))
    @test parent(got_damped) ≈ parent(expected_damped)
end

@testset "vector_laplacian with one horizontal dimension" begin
    space = TU.SpectralElementSpace1D(FT; context)
    coord = Fields.coordinate_field(space)
    div = Operators.Divergence()
    wgrad = Operators.Gradient{Operators.WeakForm}()
    C12 = Geometry.Covariant12Vector
    u = @. C12(wgrad(sin(coord.x)))

    expected = @. C12(wgrad(div(u)))
    got = materialize(Operators.vector_laplacian(u))
    @test parent(got) ≈ parent(expected)

    factor = FT(1 / 2)
    expected_damped = @. factor * C12(wgrad(div(u)))
    got_damped =
        materialize(Operators.vector_laplacian(u; divergence_factor = factor))
    @test parent(got_damped) ≈ parent(expected_damped)
end

@testset "hyperdiffusion example usage on an extruded sphere" begin
    space = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 4,
        helem = 4,
        Nq = 4,
        context,
    )
    coord = Fields.coordinate_field(space)
    ρe = @. sind(coord.long) * cosd(coord.lat) * (1 + coord.z)
    p = @. cosd(coord.long) + coord.z
    ρ = @. 2 + sind(coord.lat)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    grad = Operators.Gradient()
    div = Operators.Divergence()
    wgrad = Operators.Gradient{Operators.WeakForm}()
    curl = Operators.Curl()
    wcurl = Operators.Curl{Operators.WeakForm}()
    C12 = Geometry.Covariant12Vector
    C3 = Geometry.Covariant3Vector
    κ₄ = FT(1e-3)

    # First pass over a lazy argument, as the hyperdiffusion example uses it.
    h_tot = Base.Broadcast.broadcasted((ρeᵢ, pᵢ, ρᵢ) -> (ρeᵢ + pᵢ) / ρᵢ, ρe, p, ρ)
    χ = similar(ρe)
    χ .= Operators.scalar_laplacian(h_tot)
    @test parent(χ) ≈ parent(@. wdiv(grad((ρe + p) / ρ)))

    # Second pass fused into a tendency update.
    Yₜ = zeros(space)
    Yₜ .-= κ₄ .* Operators.scalar_laplacian(χ; weight = ρ)
    @test parent(Yₜ) ≈ parent(@. -κ₄ * wdiv(ρ * grad(χ)))

    u = @. C12(
        Geometry.UVVector(cosd(coord.lat), sind(coord.long) * cosd(coord.lat)),
    )
    χu = similar(u)
    χu .= Operators.vector_laplacian(u)
    @test parent(χu) ≈ parent(@. wgrad(div(u)) - C12(wcurl(C3(curl(u)))))

    factor = FT(1 / 2)
    Yₜu = similar(u)
    fill!(parent(Yₜu), FT(0))
    Yₜu .-= κ₄ .* Operators.vector_laplacian(χu; divergence_factor = factor)
    @test parent(Yₜu) ≈
          parent(@. -κ₄ * (factor * wgrad(div(χu)) - C12(wcurl(C3(curl(χu))))))
end

@testset "Laplacian atoms on a DG sphere" begin
    domain = Domains.SphereDomain(FT(6.371e6))
    mesh = Meshes.EquiangularCubedSphere(domain, 4)
    topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{4}()
    dg_space =
        Spaces.SpectralElementSpace2D(topology, quad; discontinuous = true)
    coord = Fields.coordinate_field(dg_space)
    f = @. sind(coord.long) * cosd(coord.lat)
    ρ = @. 2 + sind(coord.lat)

    κ = one(FT)
    τ = Operators.ldg_penalty_parameter(κ, dg_space)
    expected = Operators.ldg_laplacian_tendency(f, nothing, κ, τ)
    got = Operators.scalar_laplacian(f)
    @test parent(got) ≈ parent(expected)

    expected_weighted = Operators.ldg_laplacian_tendency(f, ρ, κ, τ)
    got_weighted = Operators.scalar_laplacian(f; weight = ρ)
    @test parent(got_weighted) ≈ parent(expected_weighted)

    u = @. Geometry.Covariant12Vector(
        Geometry.UVVector(cosd(coord.lat), sind(coord.long) * cosd(coord.lat)),
    )
    @test_throws ErrorException Operators.vector_laplacian(u)
end
