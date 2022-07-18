using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack
import ClimaCore

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators,
    InputOutput
using ClimaCore.Geometry

function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    xelem = 10,
    zelem = 40,
    npoly = 4,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    stretch = Meshes.ExponentialStretching{FT}(FT(100.0))
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = xelem)
    horztopology = Topologies.IntervalTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
# Prognostic thermodynamic variable: Total Energy
function init_dry_rising_bubble_2d(x, z, params)
    (; MSLP, grav, R_d, γ, T_0) = params
    C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
    C_v = R_d / (γ - 1) # heat capacity at constant volume
    x_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0
    θ_c = 0.5
    cp_d = C_p
    cv_d = C_v
    p_0 = MSLP
    g = grav

    # auxiliary quantities
    r = sqrt((x - x_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation

    θ = θ_b + θ_p # potential temperature
    π_exn = 1.0 - g * z / cp_d / θ # exner function
    T = π_exn * θ # temperature
    p = p_0 * π_exn^(cp_d / R_d) # pressure
    ρ = p / R_d / T # density
    e = cv_d * (T - T_0) + g * z
    ρe = ρ * e # total energy

    return (ρ = ρ, ρe = ρe)
end

@testset "HDF5 restart test for a 2D hybrid box domain with stretched vertical mesh" begin
    params = (
        MSLP = 1e5, # mean sea level pressure
        grav = 9.8, # gravitational constant
        R_d = 287.058, # R dry (gas constant / mol mass dry air)
        γ = 1.4, # heat capacity ratio
        T_0 = 273.16, # triple point temperature
    )
    # set up 2D domain - doubly periodic box
    hv_center_space, hv_face_space = hvspace_2D((-500, 500), (0, 1000))

    Φ(z) = grav * z

    # initial conditions
    coords = Fields.coordinate_field(hv_center_space)
    face_coords = Fields.coordinate_field(hv_face_space)

    Yc = map(
        coord -> init_dry_rising_bubble_2d(coord.x, coord.z, params),
        coords,
    )
    uₕ = map(_ -> Geometry.Covariant1Vector(0.0), coords)
    w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
    Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

    # write field vector to hdf5 file
    InputOutput.write!("hybrid2dboxstretched.hdf5", "Y" => Y) # write field vector from hdf5 file
    restart_Y = InputOutput.read("hybrid2dboxstretched.hdf5", "Y") # read fieldvector from hdf5 file
    @test restart_Y == Y # test if restart is exact
end
