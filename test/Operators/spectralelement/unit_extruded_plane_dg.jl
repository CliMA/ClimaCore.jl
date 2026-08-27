using Test
using ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra
using Random
import ClimaCore
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry, Quadratures

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# DG internal-face flux loops on an extruded plane x–z shell
# (`SpectralElementSpace1D` horizontal × finite-difference vertical): the
# 1D-horizontal counterpart of unit_extruded_sphere_dg.jl. This exercises
# `add_numerical_flux_internal_extruded_1d!` and the extruded-1D branch of
# `add_lifting_flux_internal!`, whose `UVector`-normal surface geometry and
# 1D face-node indexing are otherwise only run by the plane FDDG examples.
# (Flux-differencing and the LDG Laplacian are 2D-horizontal only.)

function extruded_plane_spaces(
    ::Type{FT};
    xmax = FT(2π),
    zmax = FT(1),
    xelem = 8,
    zelem = 5,
    Nq = 4,
) where {FT}
    context = ClimaComms.context()
    device = ClimaComms.device(context)
    xdomain = Domains.IntervalDomain(
        Geometry.XPoint(zero(FT)),
        Geometry.XPoint(xmax);
        periodic = true,
    )
    htopology =
        Topologies.IntervalTopology(device, Meshes.IntervalMesh(xdomain, nelems = xelem))
    hspace = Spaces.SpectralElementSpace1D(
        htopology,
        Quadratures.GLL{Nq}();
        discontinuous = true,
    )

    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(zmax);
        boundary_names = (:bottom, :top),
    )
    vspace = Spaces.CenterFiniteDifferenceSpace(
        device,
        Meshes.IntervalMesh(zdomain, nelems = zelem),
    )

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    return center_space, face_space
end

# Central flux of the physical flux q·u through the (plane x) face normal; the
# velocity is a `UVector`, matching the 1D-horizontal surface geometry.
central_flux(normal, (y⁻,), (y⁺,)) =
    ((y⁻.q * y⁻.uv + y⁺.q * y⁺.uv) / 2)' * normal

jump_penalty(normal, (q⁻,), (q⁺,)) = (q⁻ - q⁺) / 2

const grad_lift = Operators.central_gradient_lift
const jump_lift = Operators.jump_penalty_lift

if ClimaComms.device() isa ClimaComms.CUDADevice
    @testset "Extruded-plane (x–z) DG interface fluxes" begin
        # `_dg_face_apply!` (ext/cuda/operators_dg.jl) requires a 2D
        # horizontal grid; the extruded-1D face operators run on CPU only.
        @test_skip "extruded-1D DG face operators have no CUDA kernels"
    end
else
    @testset "Extruded-plane (x–z) DG interface fluxes" begin
        TU.@test_precisions FT begin
            tol = FT == Float32 ? FT(1e-4) : FT(1e-10)
            tol_sum = FT == Float32 ? FT(1e-5) : FT(1e-12)

            center_space, face_space = extruded_plane_spaces(FT)
            @test !Spaces.is_continuous(center_space)
            @test !Spaces.is_continuous(face_space)
            ccoords = Fields.coordinate_field(center_space)
            fcoords = Fields.coordinate_field(face_space)

            smooth_scalar(coords) = @. sin(coords.x) + coords.z / FT(2)

            @testset "weak divergence + central flux == strong divergence [$FT]" begin
                # The strong spectral divergence is element-local; the weak
                # form plus the central numerical flux reconstructs it at
                # every node only if the face term carries the correct sWJ
                # and outward normal, so this pins the
                # `compute_surface_geometry_1d` scaling that the vanishing
                # and conservation testsets below cannot see.
                hwdiv = Operators.WeakDivergence()
                hdiv = Operators.Divergence()
                lgeom = Fields.local_geometry_field(center_space)
                q = smooth_scalar(ccoords)
                uv = @. Geometry.UVector(cos(ccoords.x))
                y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)
                F = @. Geometry.transform(
                    Geometry.Contravariant1Axis(),
                    q * uv,
                )
                dy_mw = @. hwdiv(F) * (-(lgeom.WJ))
                Operators.add_numerical_flux_internal!(central_flux, dy_mw, y)
                dy = @. dy_mw / lgeom.WJ
                dy_strong = @. -hdiv(F)
                err = maximum(abs, parent(dy) .- parent(dy_strong))
                @test err < tol * maximum(abs, parent(dy_strong))
            end

            @testset "penalty flux vanishes for continuous fields (center & face) [$FT]" begin
                for (space, coords) in
                    ((center_space, ccoords), (face_space, fcoords))
                    q = smooth_scalar(coords)
                    lgeom = Fields.local_geometry_field(space)
                    r = similar(q)
                    r .= 0
                    Operators.add_numerical_flux_internal!(jump_penalty, r, q)
                    rn = @. r / lgeom.WJ
                    @test maximum(abs, parent(rn)) <
                          tol * maximum(abs, parent(q))
                end
            end

            @testset "lifting flux (UVector-valued) vanishes for continuous fields [$FT]" begin
                q = smooth_scalar(ccoords)
                lgeom = Fields.local_geometry_field(center_space)
                r = similar(q, Geometry.UVector{FT})
                fill!(parent(r), 0)
                Operators.add_lifting_flux_internal!(grad_lift, r, q)
                rn = @. r / lgeom.WJ
                @test maximum(abs, parent(rn)) < tol * maximum(abs, parent(q))
            end

            @testset "jump_penalty_lift vanishes for continuous fields & is sign-definite [$FT]" begin
                # λ-scaled interface penalty: zero on continuous fields; for a
                # discontinuous field the lifting energy ∑ q·r is < 0.
                q = smooth_scalar(ccoords)
                λ = ones(center_space)
                lgeom = Fields.local_geometry_field(center_space)
                rc = similar(q)
                fill!(parent(rc), 0)
                Operators.add_lifting_flux_internal!(jump_lift, rc, q, λ)
                rn = @. rc / lgeom.WJ
                @test maximum(abs, parent(rn)) < tol * maximum(abs, parent(q))

                qd = copy(q)
                Random.seed!(42)
                qd_cpu = Array(parent(qd))
                qd_cpu .+= FT(0.1) .* (rand(FT, size(qd_cpu)) .- FT(0.5))
                copyto!(parent(qd), qd_cpu)
                rd = similar(qd)
                fill!(parent(rd), 0)
                Operators.add_lifting_flux_internal!(jump_lift, rd, qd, λ)
                @test sum(parent(qd) .* parent(rd)) < 0
            end

            @testset "central numerical flux conserves (node sum zero) [$FT]" begin
                # On the periodic-x plane, each interior face adds ∓sWJ·flux
                # to its two nodes, so the total node sum of the residual is
                # zero.
                q = smooth_scalar(ccoords)
                uv = @. Geometry.UVector(cos(ccoords.x))
                y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)
                r = similar(q)
                r .= 0
                Operators.add_numerical_flux_internal!(central_flux, r, y)
                scale = sum(abs, parent(r))
                @test abs(sum(parent(r))) < tol_sum * scale
            end

            @testset "two-argument penalty vanishes for continuous fields [$FT]" begin
                # Exercises the multi-argument face gather (two Field
                # arguments).
                two_field_jump(normal, (q⁻, s⁻), (q⁺, s⁺)) =
                    ((q⁻ + s⁻) - (q⁺ + s⁺)) / 2
                q = smooth_scalar(ccoords)
                s = @. cos(ccoords.x)
                lgeom = Fields.local_geometry_field(center_space)
                r = similar(q)
                r .= 0
                Operators.add_numerical_flux_internal!(two_field_jump, r, q, s)
                rn = @. r / lgeom.WJ
                @test maximum(abs, parent(rn)) < tol * maximum(abs, parent(q))
            end
        end
    end
end
