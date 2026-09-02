using Test
using ClimaComms
using LinearAlgebra
using IntervalSets
import ClimaCore:
    Fields,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Operators,
    Geometry,
    Quadratures

# DG internal-face flux loops on an extruded cubed-sphere shell
# (SpectralElementSpace2D horizontal × finite-difference vertical):
# conservation, consistency at continuous fields  and equivalence of 
# weak divergence + central numerical flux with the strong divergence 
# (validates sWJ and normal scaling).

# TODO Extend to FT = (Float32, Float64)
const FT = Float64

function extruded_sphere_spaces(;
    radius = FT(6.371e6),
    zmax = FT(30e3),
    helem = 4,
    zelem = 4,
    Nq = 4,
)
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
    hspace = Spaces.SpectralElementSpace2D(htopology, Quadratures.GLL{Nq}())

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    return center_space, face_space
end

center_space, face_space = extruded_sphere_spaces()
ccoords = Fields.coordinate_field(center_space)
fcoords = Fields.coordinate_field(face_space)

smooth_scalar(coords) =
    @. sind(coords.long) * cosd(coords.lat)^2 + coords.z / FT(30e3)

central_flux(normal, (y⁻,), (y⁺,)) =
    ((y⁻.q * y⁻.uv + y⁺.q * y⁺.uv) / 2)' * normal

jump_penalty(normal, (q⁻,), (q⁺,)) = (q⁻ - q⁺) / 2

const grad_lift = Operators.central_gradient_lift

@testset "penalty flux vanishes for continuous fields (center & face spaces)" begin
    for (space, coords) in
        ((center_space, ccoords), (face_space, fcoords))
        q = smooth_scalar(coords)
        lgeom = Fields.local_geometry_field(space)
        r = similar(q)
        r .= 0
        Operators.add_numerical_flux_interior!(jump_penalty, r, q)
        rn = @. r / lgeom.WJ
        @test maximum(abs, parent(rn)) < 1e-10 * maximum(abs, parent(q))
    end
end

@testset "LDG penalty flux vanishes for continuous fields" begin
    q = smooth_scalar(ccoords)
    lgeom = Fields.local_geometry_field(center_space)
    r = similar(q)
    r .= 0
    Operators.add_ldg_laplacian_flux_interior!(r, q, FT(1))
    rn = @. r / lgeom.WJ
    @test maximum(abs, parent(rn)) < 1e-10 * maximum(abs, parent(q))
end

@testset "lifting flux (UVVector-valued) vanishes for continuous fields" begin
    q = smooth_scalar(ccoords)
    lgeom = Fields.local_geometry_field(center_space)
    r = similar(q, Geometry.UVVector{FT})
    fill!(parent(r), 0)
    Operators.add_lifting_flux_interior!(grad_lift, r, q)
    rn = @. r / lgeom.WJ
    @test maximum(abs, parent(rn)) < 1e-10 * maximum(abs, parent(q))
end

@testset "central numerical flux conserves (node sum of residual is zero)" begin
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
    @test abs(sum(parent(r))) < 1e-12 * scale
end

@testset "pure-2D sphere: lifting & penalty fluxes vanish for continuous fields" begin
    context = ClimaComms.context()
    hdomain = Domains.SphereDomain(FT(6.371e6))
    hmesh = Meshes.EquiangularCubedSphere(hdomain, 4)
    htopology = Topologies.Topology2D(context, hmesh)
    hspace = Spaces.SpectralElementSpace2D(htopology, Quadratures.GLL{4}())
    hcoords = Fields.coordinate_field(hspace)
    lgeom = Fields.local_geometry_field(hspace)

    q = @. sind(hcoords.long) * cosd(hcoords.lat)^2

    r = similar(q)
    r .= 0
    Operators.add_numerical_flux_interior!(jump_penalty, r, q)
    rn = @. r / lgeom.WJ
    @test maximum(abs, parent(rn)) < 1e-10 * maximum(abs, parent(q))

    rl = similar(q, Geometry.UVVector{FT})
    fill!(parent(rl), 0)
    Operators.add_lifting_flux_interior!(grad_lift, rl, q)
    rln = @. rl / lgeom.WJ
    @test maximum(abs, parent(rln)) < 1e-10 * maximum(abs, parent(q))
end

@testset "flux-differencing divergence: exact weak-form equivalence on a flat rectangle" begin
    # On constant metrics, flux differencing with the arithmetic mean of
    # pointwise fluxes reduces exactly to the conservative (weak) form.
    context = ClimaComms.context()
    rdomain = Domains.RectangleDomain(
        Geometry.XPoint(zero(FT)) .. Geometry.XPoint(FT(2e3)),
        Geometry.YPoint(zero(FT)) .. Geometry.YPoint(FT(2e3)),
        x1periodic = true,
        x2periodic = true,
    )
    rmesh = Meshes.RectilinearMesh(rdomain, 5, 5)
    rtopology = Topologies.Topology2D(context, rmesh)
    rspace = Spaces.SpectralElementSpace2D(rtopology, Quadratures.GLL{4}())
    rcoords = Fields.coordinate_field(rspace)
    rlgeom = Fields.local_geometry_field(rspace)

    q = @. sin(2pi * rcoords.x / 2e3) * cos(2pi * rcoords.y / 2e3) + 2
    uv = @. Geometry.UVVector(
        cos(2pi * rcoords.x / 2e3),
        sin(2pi * rcoords.y / 2e3),
    )
    y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)

    central_2pt(nvec_a, nvec_b, a, b) =
        ((a.q * a.uv)' * nvec_a + (b.q * b.uv)' * nvec_b) / 2

    r_fd = similar(q)
    r_fd .= 0
    Operators.add_flux_differencing_divergence!(central_2pt, r_fd, y)

    hwdiv = Operators.WeakDivergence()
    F = @. q * uv
    r_weak = @. hwdiv(F) * (-(rlgeom.WJ))

    scale = maximum(abs, parent(r_weak))
    @test maximum(abs, parent(r_fd .- r_weak)) < 1e-12 * scale
end

@testset "flux-differencing divergence: per-element telescoping on the sphere" begin
    # By SBP, the FD volume sum and the own-side lifts cancel in the node sum
    # of every element, for any symmetric consistent two-point flux.
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
    @test abs(sum(parent(r))) < 1e-12 * scale
end

@testset "weak divergence + central flux equals strong divergence" begin
    hwdiv = Operators.WeakDivergence()
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
    @test maximum(abs, parent(dy .- dy_strong)) < 1e-10 * scale
end
