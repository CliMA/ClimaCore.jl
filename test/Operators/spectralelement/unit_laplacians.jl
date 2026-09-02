# Tests the horizontal Laplacian atoms against the equivalent hand-written
# operator compositions, on CG spaces with two and one horizontal dimensions
# and on a DG sphere (where the scalar atom must match the interior-penalty
# Laplacian and the vector atom must fail loudly), and then assembles the DG
# operator as a matrix to pin the two properties it has to have: it must be
# symmetric, and it must only ever damp — plus their dependence on the penalty.
using Test
using LinearAlgebra
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Grids,
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
        Spaces.SpectralElementSpace2D(
            topology,
            quad;
            discretization = Grids.DG(),
        )
    coord = Fields.coordinate_field(dg_space)
    f = @. sind(coord.long) * cosd(coord.lat)
    ρ = @. 2 + sind(coord.lat)

    κ = one(FT)
    τ = Operators.sipg_penalty_parameter(κ, dg_space)
    expected = Operators.sipg_laplacian_tendency(f, nothing, κ, τ)
    got = Operators.scalar_laplacian(f)
    @test parent(got) ≈ parent(expected)

    τ_weighted = Operators.sipg_penalty_parameter(κ, dg_space; weight = ρ)
    expected_weighted = Operators.sipg_laplacian_tendency(f, ρ, κ, τ_weighted)
    got_weighted = Operators.scalar_laplacian(f; weight = ρ)
    @test parent(got_weighted) ≈ parent(expected_weighted)

    # The in-place form writes into a caller-owned field, including when the
    # destination aliases the argument.
    out = similar(f)
    @test Operators.scalar_laplacian!(out, f) === out
    @test parent(out) ≈ parent(expected)

    aliased = copy(f)
    Operators.scalar_laplacian!(aliased, aliased)
    @test parent(aliased) ≈ parent(expected)

    # A lazy argument is materialized into a scratch field; the operator is
    # linear, so the result of doubling the input is exactly double.
    got_lazy =
        Operators.scalar_laplacian(Base.Broadcast.broadcasted(*, 2, f))
    @test parent(got_lazy) ≈ 2 .* parent(expected)

    # A DG ∇⁴ is two passes with no DSS between them.
    ∇²f = Operators.scalar_laplacian(f)
    expected_∇⁴ = Operators.sipg_laplacian_tendency(∇²f, nothing, κ, τ)
    got_∇⁴ = Operators.scalar_laplacian(Operators.scalar_laplacian(f))
    @test parent(got_∇⁴) ≈ parent(expected_∇⁴)

    u = @. Geometry.Covariant12Vector(
        Geometry.UVVector(cosd(coord.lat), sind(coord.long) * cosd(coord.lat)),
    )
    @test_throws ErrorException Operators.vector_laplacian(u)
end

# Unit square periodic DG plane
function dg_periodic_plane(FT, nx, ny, Nq; context)
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(FT(0)),
            Geometry.XPoint(FT(1));
            periodic = true,
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(FT(0)),
            Geometry.YPoint(FT(1));
            periodic = true,
        ),
    )
    topology = Topologies.Topology2D(context, Meshes.RectilinearMesh(domain, nx, ny))
    return Spaces.SpectralElementSpace2D(
        topology,
        Quadratures.GLL{Nq}();
        discretization = Grids.DG(),
    )
end

# The assembled matrix and its spectrum live on the host, so each column is
# copied back once. The unit vector that generates the column is set with a
# one-element `copyto!` rather than `parent(q)[j] = 1`, which would be scalar
# indexing into a GPU array.
function assemble_weighted_laplacian(space, op)
    FT = Spaces.undertype(space)
    q = zeros(space)
    n = length(parent(q))
    A = zeros(FT, n, n)
    unit = FT[1]
    for j in 1:n
        fill!(parent(q), FT(0))
        copyto!(parent(q), j, unit, 1, 1)
        copyto!(view(A, :, j), vec(Array(parent(materialize(op(q))))))
    end
    # `.WJ` is a view into the local geometry; broadcast it into a field of its
    # own so that `parent` is a plain array laid out like the columns of `A`.
    lgWJ = Fields.local_geometry_field(space).WJ
    wj = @. lgWJ + zero(FT)
    WJ = Diagonal(vec(Array(parent(wj))))
    return WJ * A
end

# Eigenvalues of the symmetric half of `B`
symmetric_spectrum(B) = eigvals(Symmetric(Matrix((B + B') / 2)))

@testset "DG scalar Laplacian is symmetric" begin
    space = dg_periodic_plane(FT, 2, 2, 3; context)
    coord = Fields.coordinate_field(space)
    weights = (
        "unweighted" => nothing,
        # One gentle weight and one spanning three decades, as a density does
        # over the depth of an atmosphere. The penalty has to follow both.
        "smooth weight" => (@. 2 + sin(2 * FT(π) * coord.x) * cos(2 * FT(π) * coord.y)),
        "steep weight" => (@. exp(-8 * coord.y)),
    )
    for (name, weight) in weights
        B = assemble_weighted_laplacian(
            space,
            q -> Operators.scalar_laplacian(q; weight),
        )
        @testset "$name" begin
            @test norm(B - B') / norm(B) < 1e-13
            # Every mode damped, apart from constants, which a periodic
            # domain leaves untouched.
            ev = symmetric_spectrum(B)
            @test maximum(ev) < 1e-10 * abs(minimum(ev))
            @test minimum(ev) < 0
        end
    end
end

@testset "DG scalar Laplacian only damps if the penalty is large enough" begin
    space = dg_periodic_plane(FT, 2, 2, 3; context)
    κ = one(FT)
    τ = Operators.sipg_penalty_parameter(κ, space)
    laplacian(τ) = q -> Operators.sipg_laplacian_tendency(q, nothing, κ, τ)

    B = assemble_weighted_laplacian(space, laplacian(τ))
    ev = symmetric_spectrum(B)
    @test maximum(ev) < 1e-10 * abs(minimum(ev))

    B_under = assemble_weighted_laplacian(space, laplacian(FT(0.02) .* τ))
    ev_under = symmetric_spectrum(B_under)
    # Too small a penalty and the operator amplifies the very modes it is
    # meant to damp (measured max eigenvalue 1.9 against |min| 9.1).
    @test maximum(ev_under) > FT(0.01) * abs(minimum(ev_under))
end

@testset "sipg_penalty_parameter varies over the mesh and carries the weight" begin
    κ = one(FT)

    plane = dg_periodic_plane(FT, 2, 2, 3; context)
    τ_plane = Operators.sipg_penalty_parameter(κ, plane)
    global_τ =
        κ * (2 * 3 - 1)^2 / Spaces.node_horizontal_length_scale(plane)
    @test all(≈(global_τ), parent(τ_plane))

    domain = Domains.SphereDomain(FT(6.371e6))
    mesh = Meshes.EquiangularCubedSphere(domain, 4)
    topology = Topologies.Topology2D(context, mesh)
    sphere = Spaces.SpectralElementSpace2D(
        topology,
        Quadratures.GLL{4}();
        discretization = Grids.DG(),
    )
    τ_sphere = Operators.sipg_penalty_parameter(κ, sphere)
    lo, hi = extrema(parent(τ_sphere))
    @test hi / lo > 1.3
    @test lo < κ * (2 * 4 - 1)^2 / Spaces.node_horizontal_length_scale(sphere) < hi

    coord = Fields.coordinate_field(sphere)
    w = @. 2 + sind(coord.lat)
    τ_weighted = Operators.sipg_penalty_parameter(κ, sphere; weight = w)
    @test parent(τ_weighted) ≈ parent(w) .* parent(τ_sphere)

    τ_out = similar(τ_sphere)
    @test Operators.sipg_penalty_parameter!(τ_out, κ; weight = w) === τ_out
    @test parent(τ_out) ≈ parent(τ_weighted)
end

@testset "scalar_laplacian! does not allocate on DG" begin
    space = dg_periodic_plane(FT, 2, 2, 3; context)
    coord = Fields.coordinate_field(space)
    q = @. sin(2 * FT(π) * coord.x) * cos(2 * FT(π) * coord.y)
    ρ = @. 2 + sin(2 * FT(π) * coord.y)
    out = similar(q)
    # Compile first, then measure with @allocated
    Operators.scalar_laplacian!(out, q)
    Operators.scalar_laplacian!(out, q; weight = ρ)
    if !(ClimaComms.device() isa ClimaComms.CUDADevice) &&
       TU.allocation_checks_meaningful()
        @test (@allocated Operators.scalar_laplacian!(out, q)) < 256
        @test (@allocated Operators.scalar_laplacian!(out, q; weight = ρ)) < 256
    end
end
