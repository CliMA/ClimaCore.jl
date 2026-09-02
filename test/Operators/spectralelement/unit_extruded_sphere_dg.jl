using Test
using ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra
using Random
using IntervalSets
import ClimaCore
import ClimaCore:
    Fields,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Operators,
    Geometry,
    Quadratures

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# DG interior-face flux loops on an extruded cubed-sphere shell
# (SpectralElementSpace2D horizontal × finite-difference vertical):
# conservation, consistency at continuous fields, and equivalence of
# weak divergence + central numerical flux with the strong divergence
# (validates sWJ and normal scaling).

function extruded_sphere_spaces(
    ::Type{FT};
    radius = FT(6.371e6),
    zmax = FT(30e3),
    helem = 4,
    zelem = 4,
    Nq = 4,
) where {FT}
    context = ClimaComms.context()
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(zmax);
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = zelem)
    vtopology = Topologies.IntervalTopology(ClimaComms.device(context), vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    hspace = Spaces.SpectralElementSpace2D(
        htopology,
        Quadratures.GLL{Nq}();
        discretization = Spaces.DG(),
    )

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    return center_space, face_space
end

central_flux(normal, (y⁻,), (y⁺,)) =
    ((y⁻.q * y⁻.uv + y⁺.q * y⁺.uv) / 2)' * normal

jump_penalty(normal, (q⁻,), (q⁺,)) = (q⁻ - q⁺) / 2

const grad_lift = Operators.central_gradient_lift
const curl3_lift = Operators.central_curl3_lift
const jump_lift = Operators.jump_penalty_lift

@noinline function measured_fddg_allocs(fn::F, r, y) where {F}
    Operators.add_flux_differencing_divergence!(fn, r, y)
    return @allocated Operators.add_flux_differencing_divergence!(fn, r, y)
end

@testset "Extruded-sphere DG interface fluxes" begin
    TU.@test_precisions FT begin
        tol = FT == Float32 ? FT(1e-4) : FT(1e-10)
        tol_sum = FT == Float32 ? FT(1e-5) : FT(1e-12)

        center_space, face_space = extruded_sphere_spaces(FT)
        @test !Spaces.is_continuous(center_space)
        @test !Spaces.is_continuous(face_space)
        ccoords = Fields.coordinate_field(center_space)
        fcoords = Fields.coordinate_field(face_space)

        smooth_scalar(coords) =
            @. sind(coords.long) * cosd(coords.lat)^2 + coords.z / FT(30e3)

        @testset "penalty flux vanishes for continuous fields (center & face) [$FT]" begin
            for (space, coords) in
                ((center_space, ccoords), (face_space, fcoords))
                q = smooth_scalar(coords)
                lgeom = Fields.local_geometry_field(space)
                r = similar(q)
                r .= 0
                Operators.add_numerical_flux_interior!(jump_penalty, r, q)
                rn = @. r / lgeom.WJ
                @test maximum(abs, parent(rn)) < tol * maximum(abs, parent(q))
            end
        end

        @testset "LDG penalty flux vanishes for continuous fields [$FT]" begin
            # Isolate τ[[q]]: with G ≡ 0 the consistency term drops out, so the
            # face residual must vanish when [[q]] ≈ 0.
            q = smooth_scalar(ccoords)
            grad = Operators.Gradient()
            lgeom = Fields.local_geometry_field(center_space)
            G0 = @. Geometry.UVVector(zero(grad(q)))
            r = similar(q)
            r .= 0
            Operators.add_ldg_laplacian_flux_interior!(r, q, G0, FT(1), FT(1))
            rn = @. r / lgeom.WJ
            @test maximum(abs, parent(rn)) < tol * maximum(abs, parent(q))
        end

        @testset "lifting flux (UVVector-valued) vanishes for continuous fields [$FT]" begin
            q = smooth_scalar(ccoords)
            lgeom = Fields.local_geometry_field(center_space)
            r = similar(q, Geometry.UVVector{FT})
            fill!(parent(r), 0)
            Operators.add_lifting_flux_interior!(grad_lift, r, q)
            rn = @. r / lgeom.WJ
            @test maximum(abs, parent(rn)) < tol * maximum(abs, parent(q))
        end

        @testset "central_curl3_lift vanishes for continuous fields [$FT]" begin
            # Radial curl lifting from orthonormal (u, v) components: continuous
            # tangential fields ⇒ zero interface correction.
            u = @. cosd(ccoords.long)
            v = @. -sind(ccoords.long) * sind(ccoords.lat)
            lgeom = Fields.local_geometry_field(center_space)
            r = similar(u)
            fill!(parent(r), 0)
            Operators.add_lifting_flux_interior!(curl3_lift, r, u, v)
            rn = @. r / lgeom.WJ
            scale = max(maximum(abs, parent(u)), maximum(abs, parent(v)))
            @test maximum(abs, parent(rn)) < tol * scale
        end

        @testset "jump_penalty_lift vanishes for continuous fields & is sign-definite [$FT]" begin
            # λ-scaled interface penalty: zero on continuous fields; for a
            # discontinuous field the lifting energy ∑ q·r is < 0.
            q = smooth_scalar(ccoords)
            λ = ones(center_space)
            lgeom = Fields.local_geometry_field(center_space)
            rc = similar(q)
            fill!(parent(rc), 0)
            Operators.add_lifting_flux_interior!(jump_lift, rc, q, λ)
            rn = @. rc / lgeom.WJ
            @test maximum(abs, parent(rn)) < tol * maximum(abs, parent(q))

            qd = copy(q)
            Random.seed!(42)
            qd_cpu = Array(parent(qd))
            qd_cpu .+= FT(0.1) .* (rand(FT, size(qd_cpu)) .- FT(0.5))
            copyto!(parent(qd), qd_cpu)
            rd = similar(qd)
            fill!(parent(rd), 0)
            Operators.add_lifting_flux_interior!(jump_lift, rd, qd, λ)
            @test sum(parent(qd) .* parent(rd)) < 0
        end

        @testset "central numerical flux conserves (node sum zero) [$FT]" begin
            q = smooth_scalar(ccoords)
            uv = @. Geometry.UVVector(
                cosd(ccoords.long),
                -sind(ccoords.long) * sind(ccoords.lat),
            )
            y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)
            r = similar(q)
            r .= 0
            Operators.add_numerical_flux_interior!(central_flux, r, y)
            scale = sum(abs, parent(r))
            @test abs(sum(parent(r))) < tol_sum * scale
        end

        @testset "pure-2D sphere: lifting & penalty fluxes vanish [$FT]" begin
            context = ClimaComms.context()
            hdomain = Domains.SphereDomain(FT(6.371e6))
            hmesh = Meshes.EquiangularCubedSphere(hdomain, 4)
            htopology = Topologies.Topology2D(context, hmesh)
            hspace = Spaces.SpectralElementSpace2D(
                htopology,
                Quadratures.GLL{4}();
                discretization = Spaces.DG(),
            )
            @test !Spaces.is_continuous(hspace)
            hcoords = Fields.coordinate_field(hspace)
            lgeom = Fields.local_geometry_field(hspace)

            q = @. sind(hcoords.long) * cosd(hcoords.lat)^2

            r = similar(q)
            r .= 0
            Operators.add_numerical_flux_interior!(jump_penalty, r, q)
            rn = @. r / lgeom.WJ
            @test maximum(abs, parent(rn)) < tol * maximum(abs, parent(q))

            rl = similar(q, Geometry.UVVector{FT})
            fill!(parent(rl), 0)
            Operators.add_lifting_flux_interior!(grad_lift, rl, q)
            rln = @. rl / lgeom.WJ
            @test maximum(abs, parent(rln)) < tol * maximum(abs, parent(q))

            u = @. cosd(hcoords.long)
            v = @. -sind(hcoords.long) * sind(hcoords.lat)
            rcurl = similar(q)
            fill!(parent(rcurl), 0)
            Operators.add_lifting_flux_interior!(curl3_lift, rcurl, u, v)
            rcurln = @. rcurl / lgeom.WJ
            scale = max(maximum(abs, parent(u)), maximum(abs, parent(v)))
            @test maximum(abs, parent(rcurln)) < tol * scale

            λ = ones(hspace)
            rj = similar(q)
            fill!(parent(rj), 0)
            Operators.add_lifting_flux_interior!(jump_lift, rj, q, λ)
            rjn = @. rj / lgeom.WJ
            @test maximum(abs, parent(rjn)) < tol * maximum(abs, parent(q))
        end

        @testset "flux-differencing: weak-form equivalence on flat rectangle [$FT]" begin
            # On constant metrics, flux differencing with the arithmetic mean of
            # pointwise fluxes reduces exactly to the conservative (weak) form.
            context = ClimaComms.context()
            L = FT(2e3)
            rdomain = Domains.RectangleDomain(
                Geometry.XPoint(zero(FT)) .. Geometry.XPoint(L),
                Geometry.YPoint(zero(FT)) .. Geometry.YPoint(L),
                x1periodic = true,
                x2periodic = true,
            )
            rmesh = Meshes.RectilinearMesh(rdomain, 5, 5)
            rtopology = Topologies.Topology2D(context, rmesh)
            rspace = Spaces.SpectralElementSpace2D(
                rtopology,
                Quadratures.GLL{4}();
                discretization = Spaces.DG(),
            )
            @test !Spaces.is_continuous(rspace)
            rcoords = Fields.coordinate_field(rspace)
            rlgeom = Fields.local_geometry_field(rspace)

            q =
                @. sin(2 * FT(π) * rcoords.x / L) *
                   cos(2 * FT(π) * rcoords.y / L) + FT(2)
            uv = @. Geometry.UVVector(
                cos(2 * FT(π) * rcoords.x / L),
                sin(2 * FT(π) * rcoords.y / L),
            )
            y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)

            central_2pt(nvec_a, nvec_b, a, b) =
                ((a.q * a.uv)' * nvec_a + (b.q * b.uv)' * nvec_b) / 2

            r_fd = similar(q)
            r_fd .= 0
            Operators.add_flux_differencing_divergence!(central_2pt, r_fd, y)

            hwdiv = Operators.Divergence{Operators.WeakForm}()
            F = @. q * uv
            r_weak = @. hwdiv(F) * (-(rlgeom.WJ))

            scale = maximum(abs, parent(r_weak))
            @test maximum(abs, parent(r_fd .- r_weak)) < tol_sum * scale

            # LDG/SIPG consistency term: for a continuous flux G, weak κ∇·G
            # plus −{{κG}}·n̂ recovers strong κ∇·G (same SBP as the first-order
            # central-flux identity). One-sided grad(q) is discontinuous, so use
            # the analytic gradient of q here.
            κ = FT(1)
            τ = FT(0)
            hdiv = Operators.Divergence()
            G_uv = @. Geometry.UVVector(
                (2 * FT(π) / L) *
                cos(2 * FT(π) * rcoords.x / L) *
                cos(2 * FT(π) * rcoords.y / L),
                -(2 * FT(π) / L) *
                sin(2 * FT(π) * rcoords.x / L) *
                sin(2 * FT(π) * rcoords.y / L),
            )
            G_c12 =
                Geometry.transform.(Ref(Geometry.Contravariant12Axis()), G_uv)
            r_ldg = @. (-(rlgeom.WJ)) * κ * (-hwdiv(G_c12))
            Operators.add_ldg_laplacian_flux_interior!(r_ldg, q, G_uv, κ, τ)
            r_lap_strong = @. rlgeom.WJ * κ * hdiv(G_c12)
            scale_lap = maximum(abs, parent(r_lap_strong))
            @test maximum(abs, parent(r_ldg .- r_lap_strong)) < tol * scale_lap

            # Check allocations
            allocs = measured_fddg_allocs(central_2pt, r_fd, y)
            if !(ClimaComms.device() isa ClimaComms.CUDADevice) &&
               TU.allocation_checks_meaningful()
                @test allocs == 0
            end
        end

        @testset "flux-differencing: per-element telescoping on the sphere [$FT]" begin
            # By SBP, the FD volume sum and the own-side lifts cancel in the node
            # sum of every element, for any symmetric consistent two-point flux.
            q = smooth_scalar(ccoords)
            uv = @. Geometry.UVVector(
                cosd(ccoords.long),
                -sind(ccoords.long) * sind(ccoords.lat),
            )
            y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)

            kg_2pt(nvec_a, nvec_b, a, b) =
                ((a.q + b.q) / 2) * ((a.uv' * nvec_a + b.uv' * nvec_b) / 2)

            r = similar(q)
            r .= 0
            Operators.add_flux_differencing_divergence!(kg_2pt, r, y)
            scale = sum(abs, parent(r))
            @test abs(sum(parent(r))) < tol_sum * scale

            # Check allocations
            allocs = measured_fddg_allocs(kg_2pt, r, y)
            if !(ClimaComms.device() isa ClimaComms.CUDADevice) &&
               TU.allocation_checks_meaningful()
                @test allocs == 0
            end
        end

        @testset "weak divergence + central flux equals strong divergence [$FT]" begin
            hwdiv = Operators.Divergence{Operators.WeakForm}()
            hdiv = Operators.Divergence()
            lgeom = Fields.local_geometry_field(center_space)

            uv = @. Geometry.UVVector(
                cosd(ccoords.long),
                -sind(ccoords.long) * sind(ccoords.lat),
            )
            F = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), uv)
            q = ones(center_space)
            y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)

            dy_mw = @. hwdiv(F) * (-(lgeom.WJ))
            Operators.add_numerical_flux_interior!(central_flux, dy_mw, y)
            dy = @. dy_mw / lgeom.WJ

            dy_strong = @. -hdiv(F)
            scale = maximum(abs, parent(dy_strong))
            @test maximum(abs, parent(dy .- dy_strong)) < tol * scale
        end
    end
end
