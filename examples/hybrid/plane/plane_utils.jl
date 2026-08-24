# Shared setup for the 2D (x–z) hybrid plane examples: the physical constants
# they all use and the extruded-space constructor, with or without topography.
#
# Each example includes this file, builds its spaces with `hvspace_2D`, and
# supplies its own initial condition and tendency. This mirrors what
# `sphere/baroclinic_wave_utils.jl` does for the sphere cases.

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

# ============================================================================
# Physical parameters
# ============================================================================

const MSLP = 1e5          # mean sea level pressure (Pa)
const grav = 9.8          # gravitational acceleration (m/s²)
const R_d = 287.058       # gas constant for dry air (J/kg/K)
const γ = 1.4             # heat capacity ratio
const C_p = R_d * γ / (γ - 1)   # heat capacity at constant pressure
const C_v = R_d / (γ - 1)       # heat capacity at constant volume
const T_0 = 273.16        # triple point temperature (K)

# ============================================================================
# Spaces
# ============================================================================

"""
    hvspace_2D(xlim, zlim; xelem, zelem, npoly, warp_fn)

Extruded 2D (x–z) space pair: a periodic spectral element discretization in x
extruded over a finite difference discretization in z.

Passing `warp_fn` — a function of a horizontal coordinate returning a surface
elevation — applies `Hypsography.LinearAdaption`, giving a terrain-following
mesh. With `warp_fn = nothing` (the default) the mesh is flat.
"""
function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π);
    xelem = 10,
    zelem = 40,
    npoly = 4,
    warp_fn = nothing,
    context = ClimaComms.context(),
)
    FT = Float64
    device = ClimaComms.device(context)

    xdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(xdomain, nelems = xelem)
    horztopology = Topologies.IntervalTopology(device, horzmesh)
    horzspace =
        Spaces.SpectralElementSpace1D(horztopology, Quadratures.GLL{npoly + 1}())

    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(zdomain, nelems = zelem)

    if isnothing(warp_fn)
        vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)
        center_space =
            Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
        face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    else
        # A terrain-following mesh is built from the faces, so that the surface
        # coincides with the lowest face rather than a cell center.
        vert_face_space = Spaces.FaceFiniteDifferenceSpace(device, vertmesh)
        z_surface =
            Geometry.ZPoint.(warp_fn.(Fields.coordinate_field(horzspace)))
        face_space = Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_face_space,
            Hypsography.LinearAdaption(z_surface),
        )
        center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(face_space)
    end
    return (center_space, face_space)
end

# ============================================================================
# Thermodynamics
# ============================================================================

"""
    geopotential(z)

Geopotential at height `z`.
"""
geopotential(z) = grav * z

"""
    pressure_from_ρe(ρe, K, Φ, ρ)

Pressure diagnosed from total energy density, given the specific kinetic energy
`K` and geopotential `Φ`.
"""
pressure_from_ρe(ρe, K, Φ, ρ) = ρ * R_d * ((ρe / ρ - K - Φ) / C_v + T_0)
