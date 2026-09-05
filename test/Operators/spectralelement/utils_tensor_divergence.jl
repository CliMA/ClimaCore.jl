# Shared helpers for the Cartesian tensor-divergence tests. Definitions only —
# no top-level @test (see test/README.md on `utils_` files).
import ClimaComms
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Hypsography,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies
import ClimaCore.Geometry: ⊗

# A cubed-sphere spectral-element space on either discretization.
function tensor_div_sphere_space(
    ::Type{FT};
    radius = FT(6.371e6),
    helem = 4,
    Nq = 4,
    discretization = Spaces.CG(),
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    return Spaces.SpectralElementSpace2D(
        htopology,
        Quadratures.GLL{Nq}();
        discretization,
    )
end

# The same sphere extruded over terrain: the geometry where the momentum
# rotation meets `LatLongZPoint` coordinates, a product `WJ`, and a `∂ξ∂x` the
# warp has tilted. The O(2 km) surface moves `J` by ~13% from the flat grid.
function tensor_div_topography_space(
    ::Type{FT};
    radius = FT(6.371e6),
    helem = 4,
    Nq = 4,
    nz = 10,
    ztop = FT(3e4),
    discretization = Spaces.CG(),
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    hspace = tensor_div_sphere_space(
        FT;
        radius,
        helem,
        Nq,
        discretization,
        context,
    )
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(ztop);
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain; nelems = nz)
    vspace =
        Spaces.CenterFiniteDifferenceSpace(Topologies.IntervalTopology(context, vmesh))
    hcoords = Fields.coordinate_field(hspace)
    z_surface = Geometry.ZPoint.(
        @. FT(2000) * (cosd(hcoords.lat)^2 * sind(3 * hcoords.long) + 1)
    )
    return Spaces.ExtrudedFiniteDifferenceSpace(
        hspace,
        vspace,
        Hypsography.LinearAdaption(z_surface),
    )
end

# The interface completion the operator dispatches on, built from a
# model-shaped state: that holds the operator to working with the completion a
# model already carries.
function tensor_div_completion(space; numflux)
    FT = Spaces.undertype(space)
    state = Fields.Field(
        NamedTuple{(:ρ, :ρu), Tuple{FT, Geometry.UVVector{FT}}},
        space,
    )
    return Operators.tendency_completion(state; numflux)
end

# A constant global-Cartesian vector expressed in the local frame. The closure
# captures the typed constant, which keeps StaticArrays' broadcast style out of
# the Field broadcast.
function local_cartesian_field(space, vcart::Geometry.Cartesian123Vector)
    gg = Spaces.global_geometry(space)
    coords = Fields.coordinate_field(space)
    rgg = Ref(gg)
    f(geom, coord) = Geometry.LocalVector(vcart, geom, coord)
    return f.(rgg, coords)
end

# Weak divergence of a vector field, completed across interfaces the same way
# `cartesian_tensor_divergence` completes the tensor one — the same volume term
# and central interface flux — since it is the right-hand side of the
# `∇ₕ·(v⊗m) = (∇ₕ·v) m` identity.
function completed_divergence(v)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    dv = @. -wdiv(v)
    completion = Operators.tendency_completion(
        dv;
        numflux = Operators.CentralNumericalFlux(identity),
    )
    Operators.complete_tendency!(completion, dv, v)
    return @. -dv
end

# Tensor divergence with the same weak volume term and central interface flux
# as `cartesian_tensor_divergence` and no momentum-axis rotation, so on a
# curved space it omits the connection term.
function naive_tensor_divergence(T)
    space = axes(T)
    lgeom = Fields.local_geometry_field(space)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    r = @. wdiv(T) * (-(lgeom.WJ))
    Operators.add_numerical_flux_interior!(
        Operators.CentralNumericalFlux(identity),
        r,
        T,
    )
    return @. -r / lgeom.WJ
end

# ∇·(u⊗u) for the solid-body zonal flow u = U cosφ êλ on a sphere of radius R.
# The flow is non-divergent, so this is the advective acceleration
# (u·∇)u = (u²tanφ/R) êφ − (u²/R) êr: Cartesian momentum components that vary
# with position, and a radial curvature term of the same order as the
# tangential ones.
solid_body_velocity(coords, U) =
    @. Geometry.UVWVector(U * cosd(coords.lat), zero(U), zero(U))

solid_body_flux_divergence(coords, U, R) = @. Geometry.UVWVector(
    zero(U),
    U^2 * cosd(coords.lat) * sind(coords.lat) / R,
    -U^2 * cosd(coords.lat)^2 / R,
)
