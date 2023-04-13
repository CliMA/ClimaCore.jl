using ClimaComms
using Test
using LinearAlgebra

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
    Hypsography
import ClimaCore.Utilities: half
import UnPack

using Statistics

using Interpolations
using ImageFiltering
using NCDatasets

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33
using DiffEqCallbacks

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const context = ClimaComms.SingletonCommsContext()

const n_vert = 45
const n_horz = 16
const p_horz = 3

const R = 6.4e6 # radius
const Ω = 7.2921e-5 # Earth rotation (radians / sec)
const z_top = 3.0e4 # height position of the model top
const grav = 9.8 # gravitational constant
const p_0 = 1e5 # mean sea level pressure

const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const T_tri = 273.16 # triple point temperature
const γ = 1.4 # heat capacity ratio
const cv_d = R_d / (γ - 1)
const cp_d = R_d + cv_d

const T_0 = 300 # isothermal atmospheric temperature
const H = R_d * T_0 / grav # scale height
const hyperdiffusivity = 1e16 #m²/s

# set up function space
function topography_dcmip200(coords)
    λ, ϕ = coords.long, coords.lat
    FT = eltype(λ)
    ϕₘ = FT(0) # degrees (equator)
    λₘ = FT(3 / 2 * 180)  # degrees
    rₘ = @. FT(acos(sind(ϕₘ) * sind(ϕ) + cosd(ϕₘ) * cosd(ϕ) * cosd(λ - λₘ))) # Great circle distance (rads)
    Rₘ = FT(3π / 4) # Moutain radius
    ζₘ = FT(π / 16) # Mountain oscillation half-width
    h₀ = FT(2000)
    zₛ = ifelse(
        rₘ < Rₘ,
        FT(h₀ / 2) * (1 + cospi(rₘ / Rₘ)) * (cospi(rₘ / ζₘ))^2,
        FT(0),
    )
    return zₛ
end

data_path = "./ETOPO1_coarse.nc"
earth_spline = NCDataset(data_path) do data
    zlevels = data["elevation"][:]
    lon = data["longitude"][:]
    lat = data["latitude"][:]
    # Apply Smoothing
    smooth_degree = 15
    esmth = imfilter(zlevels, Kernel.gaussian(smooth_degree))
    linear_interpolation(
        (lon, lat),
        zlevels,
        extrapolation_bc = (Periodic(), Flat()),
)
end
function generate_topography_warp(earth_spline)
    function topography_earth(coords)
        λ, Φ = coords.long, coords.lat
        FT = eltype(λ)
        elevation = @. FT(earth_spline(λ, Φ))
        zₛ = @. ifelse(elevation > FT(0), elevation, FT(0))
        return zₛ
    end
    return topography_earth
end

function sphere_3D(
    R = 6.4e6,
    zlim = (0, 30.0e3),
    helem = 4,
    zelem = 15,
    npoly = 5,
    warp_fn = generate_topography_warp(earth_spline),
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(context, horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    z_surface = warp_fn.(Fields.coordinate_field(horzspace))
    hv_face_space = Spaces.ExtrudedFiniteDifferenceSpace(horzspace, 
                                                         vert_face_space, 
                                                         Hypsography.LinearAdaption(z_surface))
    hv_center_space =
        Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
    return (hv_center_space, hv_face_space)
end

# geopotential
Φ(z) = grav * z

# pressure
function pressure(ρ, e, normuvw, z)
    I = e - Φ(z) - normuvw^2 / 2
    T = I / cv_d + T_tri
    return ρ * R_d * T
end

# initial conditions for density and total energy
function init_sbr_thermo(z)

    p = p_0 * exp(-z / H)
    ρ = 1 / R_d / T_0 * p

    e = cv_d * (T_0 - T_tri) + Φ(z)
    ρe = ρ * e

    return (ρ = ρ, ρe = ρe)
end

function rayleigh_sponge(
    z;
    z_sponge = 15000.0,
    z_max = 30000.0,
    α = 0.5,  # Relaxation timescale
    τ = 0.5,
    γ = 2.0,
)
    if z >= z_sponge
        r = (z - z_sponge) / (z_max - z_sponge)
        β_sponge = α * sinpi(τ * r)^γ
        return β_sponge
    else
        return eltype(z)(0)
    end
end

function rhs!(dY, Y, parameters, t)
    UnPack.@unpack f_coords, c_coords, cuvw, cw, cω³, fω¹², fu¹², fu³, cp, cE = parameters
    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant1Vector on centers
    cρe = Y.Yc.ρe

    FT = eltype(cp)

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    z = c_coords.z
    fz = f_coords.z

    # Define Operators 
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.WeakCurl()
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    WIc2f = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    f_upwind_product1 = Operators.UpwindBiasedProductC2F()
    f_upwind_product3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
    )


    # Get cell center Jacobian 
    ᶜJ = Fields.local_geometry_field(axes(cρ)).J

    # Initialise density tendency
    dρ .= 0 .* cρ
    fuₕ = Ic2f.(cuₕ)

    # Boundary condition (Diagnostic Equation)
    u₁_bc =
        Geometry.contravariant3.(
            Fields.level(fuₕ, ClimaCore.Utilities.half),
            Fields.level(
                Fields.local_geometry_field(hv_face_space),
                ClimaCore.Utilities.half,
            ),
        )
    gⁱʲ =
        Fields.level(
            Fields.local_geometry_field(hv_face_space),
            ClimaCore.Utilities.half,
        ).gⁱʲ
    g33 = gⁱʲ.components.data.:9
    u₃_bc = Geometry.Covariant3Vector.(-1 .* u₁_bc ./ g33) 
    apply_boundary_w =
        Operators.SetBoundaryOperator(bottom = Operators.SetValue(u₃_bc))
    @. fw = apply_boundary_w(fw)
    cw = If2c.(fw)

    cuw = @. Geometry.Covariant123Vector(cuₕ) + Geometry.Covariant123Vector(cw)
    fuw = @. WIc2f(cρ * ᶜJ, Geometry.Contravariant123Vector(cuₕ)) + Geometry.Contravariant123Vector(fw)
    fu¹ = @. Geometry.project(Geometry.Contravariant12Axis(), fuw)
    fu³ = @. Geometry.project(Geometry.Contravariant3Axis(), fuw)

    ce = @. cρe / cρ
    cI = @. ce - Φ(z) - (norm(cuw)^2) / 2
    cT = @. cI / cv_d + T_0
    cp = @. cρ * R_d * cT

    h_tot = @. ce + cp / cρ # Total enthalpy at cell centers

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients
    χe = @. dρe = hwdiv(hgrad(h_tot)) 
    χuₕ = @. duₕ = hwgrad(hdiv(cuₕ)) - Geometry.project(Geometry.Covariant12Axis(), hwcurl(Geometry.project(Geometry.Covariant3Axis(), hcurl(cuₕ))))

    Spaces.weighted_dss!(dρe)
    Spaces.weighted_dss!(duₕ)

    κ₄_dynamic = hyperdiffusivity # m^4/s
    @. dρe = -κ₄_dynamic * hwdiv(cρ * hgrad(χe))
    @. duₕ = -κ₄_dynamic * (
                hwgrad(hdiv(χuₕ)) -
                Geometry.project(
                    Geometry.Covariant12Axis(),
                    hwcurl(
                        Geometry.project(
                            Geometry.Covariant3Axis(),
                            hcurl(χuₕ),
                        ),
                    ),
                )
             )

    # 1) Mass conservation
    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuw))

    # 1.b) vertical divergence

    # Apply n ⋅ ∇(X) = F
    # n^{i} * ∂X/∂_{x^{i}} 
    # Contravariant3Vector(1) ⊗ (Flux Tensor)
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    vdivc2f = Operators.DivergenceC2F(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    #dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    dρ .-= vdivf2c.(WIc2f.(ᶜJ, cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(WIc2f.(ᶜJ, cρ) .* fw)

    # 2) Momentum equation

    # curl term
    hcurl = Operators.Curl()
    # effectively a homogeneous Neumann condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(
            Geometry.Contravariant12Vector(FT(0), FT(0)),
        ),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
    )

    fω¹ = hcurl.(fw)
    fω¹ .+= vcurlc2f.(cuₕ)
    cω3 = @. hcurl(cuₕ)

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    @. dw -= fω¹ × fu¹ # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹ × (Ic2f(cρ * ᶜJ) * fu³)) / (cρ * ᶜJ)
    @. duₕ -= hgrad(cp) / cρ
    @. duₕ -= cω3 × Geometry.project(Geometry.Contravariant12Axis(), cuw)

    vgradc2fP = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    vgradc2fE = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )

    cE = @. (norm(cuw)^2) / 2 + Φ(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2fE(cE)
    @. dw -= vgradc2fP(cp) / Ic2f(cρ)

    # 3) total energy

    ## @. dρe -= hdiv(cuw * (cρe + cp))
    @. dρe -= hdiv(cuw * cρe)
    @. dρe -= hdiv(cuw * cp)

    @. dρe -= vdivf2c(WIc2f(ᶜJ, cρ) * fw * Ic2f((cρe + cp)/cρ))

    ##@. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))
    @. dρe -= vdivf2c(Ic2f(cuₕ * cρe))
    @. dρe -= vdivf2c(Ic2f(cuₕ * cp))

    # Sponge tendency
    β = @. rayleigh_sponge(z)
    βf = @. rayleigh_sponge(fz)
    @. dw -= βf * fw

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY

end

# set up 3D spherical domain and coords
hv_center_space, hv_face_space =
    sphere_3D(R, (0, z_top), n_horz, n_vert, p_horz)
c_coords = Fields.coordinate_field(hv_center_space)
f_coords = Fields.coordinate_field(hv_face_space)

# Coriolis
const f = @. Geometry.Contravariant3Vector(
    Geometry.WVector(2 * Ω * sind(c_coords.lat)),
)

# discrete hydrostatic profile
zc_vec = parent(c_coords.z) |> unique

N = length(zc_vec)
ρ = zeros(Float64, N)
p = zeros(Float64, N)
ρe = zeros(Float64, N)

# compute ρ, ρe, p from analytical formula; not discretely balanced
for i in 1:N
    var = init_sbr_thermo(zc_vec[i])
    ρ[i] = var.ρ
    ρe[i] = var.ρe
    p[i] = pressure(ρ[i], ρe[i] / ρ[i], 0.0, zc_vec[i])
end

ρ_ana = copy(ρ) # keep a copy for analytical ρ which will be used in correction ρe

function discrete_hydrostatic_balance!(ρ, p, dz, grav)
    for i in 1:(length(ρ) - 1)
        ρ[i + 1] = -ρ[i] - 2 * (p[i + 1] - p[i]) / dz / grav
    end
end

discrete_hydrostatic_balance!(ρ, p, z_top / n_vert, grav)
# now ρ (after correction) and p (computed from analytical relation) are in discrete hydrostatic balance
# only need to correct ρe without changing ρ and p, i.e., keep ρT unchanged before vs after the correction on ρ
ρe = @. ρe + (ρ - ρ_ana) * Φ(zc_vec) - (ρ - ρ_ana) * cv_d * T_tri

# Note: In princile, ρe = @. cv_d * p /R_d - ρ * cv_d * T_tri + ρ * Φ(zc_vec) should work,
#       however, it is not as accurate as the above correction

# set up initial condition: not discretely balanced; only create a Field as a place holder
Yc = map(coord -> init_sbr_thermo(coord.z), c_coords)
# put the dicretely balanced ρ and ρe into Yc

#parent(Yc.ρ) .= ρ  # Yc.ρ is a VIJFH layout
#parent(Yc.ρe) .= ρe

# initialize velocity: at rest
uₕ = map(_ -> Geometry.Covariant12Vector(0.0, 0.0), c_coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), f_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

cuvw = Geometry.Covariant123Vector.(Y.uₕ)
If2c = Operators.InterpolateF2C()
hcurl = Operators.Curl()
Ic2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

cw = If2c.(Y.w)
cω³ = hcurl.(Y.uₕ)
fω¹² = hcurl.(Y.w)
fu¹² =
    @. Geometry.Contravariant12Vector(Geometry.Covariant123Vector(Ic2f(Y.uₕ)))
fu³ = @. Geometry.Contravariant3Vector(Geometry.Covariant123Vector(Y.w))
cp = @. pressure(Y.Yc.ρ, Y.Yc.ρe / Y.Yc.ρ, norm(cuvw), c_coords.z)
cE = @. (norm(cuvw)^2) / 2 + Φ(c_coords.z)

parameters = (; c_coords, cuvw, cw, cω³, fω¹², fu¹², fu³, cp, cE, f_coords)
# initialize tendency
dYdt = similar(Y)
# set up rhs
rhs!(dYdt, Y, parameters, 0.0)

# run!
using OrdinaryDiffEq
# Solve the ODE
dt = 1.0
T = 2.0
prob = ODEProblem(rhs!, Y, (0.0, T), parameters)

function make_dss_func()
    _dss!(x::Fields.Field) = Spaces.weighted_dss!(x)
    _dss!(::Any) = nothing
    dss_func(Y, t, integrator) = foreach(_dss!, Fields._values(Y))
    return dss_func
end
dss_func = make_dss_func()
dss_callback = FunctionCallingCallback(dss_func, func_start = true)

integrator = OrdinaryDiffEq.init(
    prob,
    SSPRK33(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
    callback = dss_callback,
)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

# solve ode
sol = @timev OrdinaryDiffEq.solve!(integrator)
If2c = Operators.InterpolateF2C()
cuw = @. Geometry.Covariant123Vector(sol.u[end].uₕ) + If2c(Geometry.Covariant123Vector(sol.u[end].w))
uₕ_phy = Geometry.project.(Ref(Geometry.UVAxis()), cuw)
w_phy = Geometry.project.(Ref(Geometry.WAxis()), cuw)
fz = Fields.coordinate_field(f_coords).z
fz1 = Fields.level(fz,half)

#warp_fn = generate_topography_warp(earth_spline),
#z_surface = warp_fn.(Fields.coordinate_field(hv_face_space.horizontal_space))
#z00 = z_surface
#for iter = 1:20000
#  d2z = wdiv.(grad.(z00))
#  Spaces.weighted_dss!(d2z)
#  z00 .+= 1e8 * 1e-1 .* d2z
#end


using Plots
using ClimaCorePlots

@test maximum(abs.(uₕ_phy.components.data.:1)) ≤ 1e-11
@test maximum(abs.(uₕ_phy.components.data.:2)) ≤ 1e-11
@test maximum(abs.(w_phy |> parent)) ≤ 1e-11
@test norm(sol.u[end].Yc.ρ) ≈ norm(sol.u[1].Yc.ρ) rtol = 1e-2
@test norm(sol.u[end].Yc.ρe) ≈ norm(sol.u[1].Yc.ρe) rtol = 1e-2




