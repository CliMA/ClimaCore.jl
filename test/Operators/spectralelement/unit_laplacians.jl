# Tests the horizontal Laplacian atoms against the equivalent hand-written
# operator compositions, on CG spaces with two and one horizontal dimensions
# and on a DG sphere (where both atoms must match the interior-penalty
# operators), and then assembles the DG operators as matrices to pin the two
# properties they have to have: they must be symmetric, and they must only
# ever damp — plus their dependence on the penalty.
using Test
using LinearAlgebra
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

    # The in-place form writes into a caller-owned field, including when the
    # destination aliases the argument.
    out = similar(u)
    @test Operators.vector_laplacian!(out, u) === out
    @test parent(out) ≈ parent(expected)

    aliased = copy(u)
    Operators.vector_laplacian!(aliased, aliased; divergence_factor = factor)
    @test parent(aliased) ≈ parent(expected_damped)
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
            discretization = Spaces.DG(),
        )
    coord = Fields.coordinate_field(dg_space)
    f = @. sind(coord.long) * cosd(coord.lat)
    ρ = @. 2 + sind(coord.lat)

    κ = one(FT)
    τ = Operators.ldg_penalty_parameter(κ, dg_space)
    expected = Operators.ldg_laplacian_tendency(f, nothing, κ, τ)
    got = Operators.scalar_laplacian(f)
    @test parent(got) ≈ parent(expected)

    τ_weighted = Operators.ldg_penalty_parameter(κ, dg_space; weight = ρ)
    expected_weighted = Operators.ldg_laplacian_tendency(f, ρ, κ, τ_weighted)
    got_weighted = Operators.scalar_laplacian(f; weight = ρ)
    @test parent(got_weighted) ≈ parent(expected_weighted)

    # Each call owns its result, so two results may be live at once. A shared
    # output buffer would make the second call overwrite the first, and
    # `∇²f_live + ∇²g_live` would silently evaluate to `2∇²g`.
    g = @. cosd(coord.long) * sind(coord.lat)
    ∇²f_live = Operators.scalar_laplacian(f)
    ∇²g_live = Operators.scalar_laplacian(g)
    @test ∇²f_live !== ∇²g_live
    @test parent(∇²f_live) ≈ parent(expected)
    @test parent(∇²g_live) ≈
          parent(Operators.ldg_laplacian_tendency(g, nothing, κ, τ))

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
        Operators.scalar_laplacian(Base.Broadcast.broadcasted(x -> 2 * x, f))
    @test parent(got_lazy) ≈ 2 .* parent(expected)

    # Feeding a result straight back in (a DG ∇⁴ needs no DSS between the
    # passes) is well defined, since each call owns its result.
    ∇²f = Operators.scalar_laplacian(f)
    expected_∇⁴ = Operators.ldg_laplacian_tendency(∇²f, nothing, κ, τ)
    got_∇⁴ = Operators.scalar_laplacian(Operators.scalar_laplacian(f))
    @test parent(got_∇⁴) ≈ parent(expected_∇⁴)

    u = @. Geometry.Covariant12Vector(
        Geometry.UVVector(cosd(coord.lat), sind(coord.long) * cosd(coord.lat)),
    )
    expected_u = Operators.ldg_vector_laplacian_tendency(u, one(FT), τ)
    got_u = Operators.vector_laplacian(u)
    @test parent(got_u) ≈ parent(expected_u)

    factor = FT(1 / 2)
    @test parent(Operators.vector_laplacian(u; divergence_factor = factor)) ≈
          parent(Operators.ldg_vector_laplacian_tendency(u, factor, τ))

    out_u = similar(u)
    @test Operators.vector_laplacian!(out_u, u) === out_u
    @test parent(out_u) ≈ parent(expected_u)

    aliased_u = copy(u)
    Operators.vector_laplacian!(aliased_u, aliased_u)
    @test parent(aliased_u) ≈ parent(expected_u)

    # The result is in the basis of the destination, so a caller that holds
    # its velocity in the local orthonormal frame gets it back in that frame.
    lgeom = Fields.local_geometry_field(dg_space)
    u_uv = @. Geometry.UVVector(u, lgeom)
    got_uv = Operators.vector_laplacian(u_uv)
    @test eltype(got_uv) <: Geometry.UVVector
    @test parent(got_uv) ≈ parent(@. Geometry.UVVector(expected_u, lgeom))
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
        discretization = Spaces.DG(),
    )
end

function assemble_weighted_laplacian(space, op)
    FT = Spaces.undertype(space)
    q = zeros(space)
    n = length(parent(q))
    A = zeros(FT, n, n)
    for j in 1:n
        fill!(parent(q), FT(0))
        parent(q)[j] = FT(1)
        A[:, j] .= vec(parent(op(q)))
    end
    WJ = Diagonal(vec(parent(Fields.local_geometry_field(space).WJ)))
    return WJ * A
end

# The same for an operator on horizontal vectors, whose two components share
# the node's `WJ`.
function assemble_weighted_vector_laplacian(space, op)
    FT = Spaces.undertype(space)
    u = Fields.Field(Geometry.UVVector{FT}, space)
    n = length(parent(u))
    A = zeros(FT, n, n)
    for j in 1:n
        fill!(parent(u), FT(0))
        parent(u)[j] = FT(1)
        A[:, j] .= vec(parent(op(u)))
    end
    lgeom = Fields.local_geometry_field(space)
    WJ = Fields.Field(Geometry.UVVector{FT}, space)
    @. WJ = Geometry.UVVector(lgeom.WJ, lgeom.WJ)
    return Diagonal(vec(parent(WJ))) * A
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
    τ = Operators.ldg_penalty_parameter(κ, space)
    laplacian(τ) = q -> Operators.ldg_laplacian_tendency(q, nothing, κ, τ)

    B = assemble_weighted_laplacian(space, laplacian(τ))
    ev = symmetric_spectrum(B)
    @test maximum(ev) < 1e-10 * abs(minimum(ev))

    B_under = assemble_weighted_laplacian(space, laplacian(FT(0.02) .* τ))
    ev_under = symmetric_spectrum(B_under)
    # Too small a penalty and the operator amplifies the very modes it is
    # meant to damp (measured max eigenvalue 1.9 against |min| 9.1).
    @test maximum(ev_under) > FT(0.01) * abs(minimum(ev_under))
end

@testset "hyperdiffusion example usage on a DG extruded sphere" begin
    # The continuous calling sequence, unchanged, on a discontinuous space:
    # both atoms carry their own face terms and `weighted_dss!` is a no-op, so
    # the same tendency code runs on either discretization.
    hspace = Spaces.SpectralElementSpace2D(
        Topologies.Topology2D(
            context,
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(FT(6.371e6)), 2),
        ),
        Quadratures.GLL{3}();
        discretization = Spaces.DG(),
    )
    vspace = Spaces.CenterFiniteDifferenceSpace(
        Meshes.IntervalMesh(
            Domains.IntervalDomain(
                Geometry.ZPoint(FT(0)),
                Geometry.ZPoint(FT(30e3));
                boundary_names = (:bottom, :top),
            );
            nelems = 4,
        ),
    )
    space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    coord = Fields.coordinate_field(space)
    lgeom = Fields.local_geometry_field(space)
    ᶜρ = @. 2 + sind(coord.lat)
    ᶜρe = @. sind(coord.long) * cosd(coord.lat) * (1 + coord.z / FT(30e3))
    ᶜp = @. cosd(coord.long) + coord.z / FT(30e3)
    ᶜuₕ = @. Geometry.Covariant12Vector(
        Geometry.UVVector(cosd(coord.lat), sind(coord.long) * cosd(coord.lat)),
        lgeom,
    )
    κ₄ = FT(1e-3)
    factor = FT(1 / 2)

    ᶜχ = similar(ᶜρe)
    ᶜχuₕ = similar(ᶜuₕ)
    buffer_χ = Spaces.create_dss_buffer(ᶜχ)
    buffer_χuₕ = Spaces.create_dss_buffer(ᶜχuₕ)
    Yₜρe = zeros(space)
    Yₜuₕ = similar(ᶜuₕ)
    fill!(parent(Yₜuₕ), FT(0))

    ᶜh_tot = Base.Broadcast.broadcasted((e, p, ρ) -> (e + p) / ρ, ᶜρe, ᶜp, ᶜρ)
    Operators.scalar_laplacian!(ᶜχ, ᶜh_tot)
    ᶜχuₕ .= Operators.vector_laplacian(ᶜuₕ)
    Spaces.weighted_dss!(ᶜχ => buffer_χ, ᶜχuₕ => buffer_χuₕ)
    Yₜρe .-= κ₄ .* Operators.scalar_laplacian(ᶜχ; weight = ᶜρ)
    Yₜuₕ .-=
        κ₄ .* Operators.vector_laplacian(ᶜχuₕ; divergence_factor = factor)

    # A no-op DSS leaves the intermediates as the atoms wrote them, and each
    # pass is the interior-penalty operator.
    τ = Operators.ldg_penalty_parameter(one(FT), space)
    @test parent(ᶜχuₕ) ≈
          parent(Operators.ldg_vector_laplacian_tendency(ᶜuₕ, one(FT), τ))
    @test parent(Yₜuₕ) ≈
          -κ₄ .* parent(
        Operators.ldg_vector_laplacian_tendency(ᶜχuₕ, factor, τ),
    )
    @test all(isfinite, parent(Yₜρe))
end

@testset "DG vector Laplacian is symmetric" begin
    space = dg_periodic_plane(FT, 2, 2, 3; context)
    for factor in (FT(1), FT(5), FT(1 / 2))
        B = assemble_weighted_vector_laplacian(
            space,
            u -> Operators.vector_laplacian(u; divergence_factor = factor),
        )
        @testset "divergence_factor = $factor" begin
            @test norm(B - B') / norm(B) < 1e-13
            ev = symmetric_spectrum(B)
            # Every mode damped, apart from the two constant vector fields,
            # which a periodic domain leaves untouched.
            @test maximum(ev) < 1e-10 * abs(minimum(ev))
            @test minimum(ev) < 0
            @test count(<(1e-10 * abs(minimum(ev))), abs.(ev)) == 2
        end
    end
end

@testset "divergence_factor scales exactly the grad-div part" begin
    space = dg_periodic_plane(FT, 2, 2, 3; context)
    assemble(factor) = assemble_weighted_vector_laplacian(
        space,
        u -> Operators.vector_laplacian(u; divergence_factor = factor),
    )
    A1, A2, A3 = assemble(FT(1)), assemble(FT(2)), assemble(FT(3))
    # Affine in the factor: the grad-div part carries it, and so does the
    # penalty on the normal jump that holds that part together — nothing else
    # does, so second differences in the factor vanish.
    @test norm(A3 - 2 * A2 + A1) < 1e-12 * norm(A1)
    # ... and it is not a no-op.
    @test norm(A2 - A1) > FT(0.1) * norm(A1)
end

@testset "DG vector Laplacian only damps if the penalty is large enough" begin
    space = dg_periodic_plane(FT, 2, 2, 3; context)
    τ = Operators.ldg_penalty_parameter(one(FT), space)
    laplacian(τ) =
        u -> Operators.ldg_vector_laplacian_tendency(u, one(FT), τ)

    B = assemble_weighted_vector_laplacian(space, laplacian(τ))
    ev = symmetric_spectrum(B)
    @test maximum(ev) < 1e-10 * abs(minimum(ev))

    B_under = assemble_weighted_vector_laplacian(space, laplacian(FT(0.02) .* τ))
    ev_under = symmetric_spectrum(B_under)
    @test maximum(ev_under) > FT(0.01) * abs(minimum(ev_under))
end

@testset "DG vector Laplacian converges to ∇²" begin
    # The truncation error of a second derivative taken from degree-`p`
    # polynomials, so one order less than the scalar atom's — the same rate
    # the CG atom has for the same identity.
    k = 2 * FT(π)
    errors = map((4, 8)) do nx
        space = dg_periodic_plane(FT, nx, nx, 4; context)
        coord = Fields.coordinate_field(space)
        u = @. Geometry.UVVector(
            sin(k * coord.x) * cos(k * coord.y),
            cos(k * coord.x) * sin(k * coord.y),
        )
        got = Operators.vector_laplacian(u)
        exact = @. (-2 * k^2) * u
        return sqrt(
            sum(@. norm(got - exact)^2) / sum(@. norm(exact)^2),
        )
    end
    @test errors[2] < errors[1] / 3
    @test errors[2] < FT(0.06)
end

@testset "vector_laplacian! does not allocate on DG" begin
    space = dg_periodic_plane(FT, 2, 2, 3; context)
    coord = Fields.coordinate_field(space)
    u = @. Geometry.UVVector(
        sin(2 * FT(π) * coord.x) * cos(2 * FT(π) * coord.y),
        cos(2 * FT(π) * coord.x) * sin(2 * FT(π) * coord.y),
    )
    out = similar(u)
    # Compile first, then measure with @allocated
    Operators.vector_laplacian!(out, u)
    Operators.vector_laplacian!(out, u; divergence_factor = FT(2))
    if !(ClimaComms.device() isa ClimaComms.CUDADevice) &&
       TU.allocation_checks_meaningful()
        @test (@allocated Operators.vector_laplacian!(out, u)) < 256
        @test (@allocated Operators.vector_laplacian!(
            out,
            u;
            divergence_factor = FT(2),
        )) < 256
    end
end

@testset "ldg_penalty_parameter varies over the mesh and carries the weight" begin
    κ = one(FT)

    plane = dg_periodic_plane(FT, 2, 2, 3; context)
    τ_plane = Operators.ldg_penalty_parameter(κ, plane)
    global_τ =
        κ * (2 * 3 - 1)^2 / Spaces.node_horizontal_length_scale(plane)
    @test all(≈(global_τ), parent(τ_plane))

    domain = Domains.SphereDomain(FT(6.371e6))
    mesh = Meshes.EquiangularCubedSphere(domain, 4)
    topology = Topologies.Topology2D(context, mesh)
    sphere = Spaces.SpectralElementSpace2D(
        topology,
        Quadratures.GLL{4}();
        discretization = Spaces.DG(),
    )
    τ_sphere = Operators.ldg_penalty_parameter(κ, sphere)
    lo, hi = extrema(parent(τ_sphere))
    @test hi / lo > 1.3
    @test lo < κ * (2 * 4 - 1)^2 / Spaces.node_horizontal_length_scale(sphere) < hi

    coord = Fields.coordinate_field(sphere)
    w = @. 2 + sind(coord.lat)
    τ_weighted = Operators.ldg_penalty_parameter(κ, sphere; weight = w)
    @test parent(τ_weighted) ≈ parent(w) .* parent(τ_sphere)

    τ_out = similar(τ_sphere)
    @test Operators.ldg_penalty_parameter!(τ_out, κ; weight = w) === τ_out
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
