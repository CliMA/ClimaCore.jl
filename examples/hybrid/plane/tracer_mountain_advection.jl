push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack

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
using ClimaCore.Geometry

using DiffEqCallbacks

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


### This file follows the test problem described in 
# https://doi.org/10.1175/1520-0493(2002)130<2459:ANTFVC>2.0.CO;2
# Section 3(b)

const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const T_0 = 273.16 # triple point temperature
const uᵣ = 1.0
const kinematic_viscosity = 75.0 #m²/s
const hyperdiffusivity = 1e8*1.0 #m²/s
 
function warp_surface(coord)   
  x = Geometry.component(coord,1)
  FT = eltype(x)
  a = 25000
  λ = 8000
  h₀ = 3000
  if abs(x) <= a
    h = h₀ * (cos(π*x/2/a))^2 * (cos(π*x/λ))^2
  else
    h = FT(0)
  end
end

function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    xelem = 60,
    zelem = 50,
    npoly = 5,
    warp_fn = warp_surface,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = xelem)
    horztopology = Topologies.IntervalTopology(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)
  
    z_surface = warp_fn.(Fields.coordinate_field(horzspace))
    hv_face_space = Spaces.ExtrudedFiniteDifferenceSpace(
                    horzspace,
                    vert_face_space,
                    Hypsography.LinearAdaption(), 
                    z_surface
              )
    hv_center_space =
        Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
    return (hv_center_space, hv_face_space)
end

# set up 2D domain - doubly periodic box
hv_center_space, hv_face_space = hvspace_2D((-150000, 150000), (0, 25000))

Φ(z) = grav * z

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
# Prognostic thermodynamic variable: Total Energy 
function init_advection_over_mountain(x, z)
    θ₀ = 280.0
    cp_d = C_p
    cv_d = C_v
    p₀ = MSLP
    g = grav
    
    𝒩 = 0.01
    π_exner = @. exp(-g * z / (cp_d * θ₀))
    θ = @. θ₀ * exp(𝒩 ^2 * z / g)
    T = @. π_exner * θ # temperature
    ρ = @. p₀ / (R_d * θ) * (π_exner)^(cp_d/R_d)
    e = @. cv_d * (T - T_0) + Φ(z) + 50.0
    ρe = @. ρ * e

    x₀ = -50000.0
    z₀ = 9000.0
    A_x = 25000.0
    A_z = 3000.0
    r = @. sqrt((x-x₀)^2/A_x^2 + (z-z₀)^2/A_z^2)
    q₀ = 1.0
  
    if r <= 1
      q = q₀ * (cos(π*r/2))^2 
    else
      q = eltype(x)(0)
    end
      
    ρq = @. ρ * q
    return (ρ = ρ,
            ρe = ρe, 
            ρq = ρq)
end

function initial_velocity(x, z)
  FT = eltype(x)
  u₀ = 10.0
  z₁ = 4000.0
  z₂ = 5000.0
  if z<=z₁
    u = @. FT(0)
  elseif z >= z₂
    u = @. u₀
  else
    u = @. u₀ * sin(π/2 * (z-z₁)/(z₂-z₁))
  end
  return @. Geometry.UWVector(u, FT(0))
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

# Assign initial conditions to cell center, cell face variables
# Group scalars (ρ, ρe) in Yc 
# Retain uₕ and w as separate components of velocity vector (primitive variables)
Yc = map(coord -> init_advection_over_mountain(coord.x, coord.z), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
uₕ_local = map(coord -> initial_velocity(coord.x, coord.z), coords)
uₕ = Geometry.Covariant1Vector.(uₕ_local)

const u₀ = uₕ

ᶜlg = Fields.local_geometry_field(hv_center_space)
ᶠlg = Fields.local_geometry_field(hv_face_space)

Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

energy_0 = sum(Y.Yc.ρe)
mass_0 = sum(Y.Yc.ρ)

function rayleigh_sponge(z;
                         z_sponge=15000.0,
                         z_max=25000.0,
                         α = 0.5,  # Relaxation timescale
                         τ = 0.5,
                         γ = 2.0)
    if z >= z_sponge
        r = (z - z_sponge) / (z_max - z_sponge)
        β_sponge = α * sinpi(τ * r)^γ
        return β_sponge
    else
        return eltype(z)(0)
    end
end

function rhs_invariant!(dY, Y, _, t)

    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant1Vector on centers
    cρe = Y.Yc.ρe
    cρq = Y.Yc.ρq

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    dρq = dY.Yc.ρq
    z = coords.z

    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    dρ .= 0 .* cρ

    cw = If2c.(fw)
    fuₕ = Ic2f.(cuₕ)
    cuw = Geometry.Covariant13Vector.(cuₕ) .+ Geometry.Covariant13Vector.(cw)

    ce = @. cρe / cρ
    cq = @. cρq / cρ
    cI = @. ce - Φ(z) - (norm(cuw)^2) / 2
    cT = @. cI / C_v + T_0
    cp = @. cρ * R_d * cT

    h_tot = @. ce + cp / cρ # Total enthalpy at cell centers

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients
    χe = @. dρe = hwdiv(hgrad(h_tot)) # we store χe in dρe
    χq = @. dρq = hwdiv(hgrad(cq)) # we store χe in dρe
    χuₕ = @. duₕ = hwgrad(hdiv(cuₕ))

    Spaces.weighted_dss!(dρe)
    Spaces.weighted_dss!(duₕ)
    Spaces.weighted_dss!(dρq)

    κ₄_dynamic = hyperdiffusivity # m^4/s
    κ₄_tracer = hyperdiffusivity * 0 
    @. dρe = -κ₄_dynamic * hwdiv(cρ * hgrad(χe))
    @. dρq = -κ₄_tracer * hwdiv(cρ * hgrad(χq))
    @. duₕ = -κ₄_dynamic * (hwgrad(hdiv(χuₕ)))

    # 1) Mass conservation
    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuw))

    # 1.b) vertical divergence
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
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    # curl term
    hcurl = Operators.Curl()
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
        top = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
    )

    fω¹ = hcurl.(fw)
    fω¹ .+= vcurlc2f.(cuₕ)

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
#    fu¹ =
#        Geometry.Contravariant1Vector.(Geometry.Covariant13Vector.(Ic2f.(cuₕ)),)
#    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant13Vector.(fw))
    fu = Geometry.Contravariant13Vector.(Ic2f.(cuₕ)) .+ Geometry.Contravariant13Vector.(fw)
    fu¹ = Geometry.project.(Ref(Geometry.Contravariant1Axis()), fu)
    fu³ = Geometry.project.(Ref(Geometry.Contravariant3Axis()), fu)
    @. dw -= fω¹ × fu¹ # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹ × fu³)


    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Contravariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Contravariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE = @. (norm(cuw)^2) / 2 + Φ(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) total energy

    @. dρe -= hdiv(cuw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))
    
    # 4) tracer 
    @. dρq -= hdiv(cuw * (cρq))
    @. dρq -= vdivf2c(fw * Ic2f(cρq))

    # Uniform 2nd order diffusion
    ∂c = Operators.GradientF2C()
    fρ = @. Ic2f(cρ)
    κ₂ = kinematic_viscosity # m^2/s

    ᶠ∇ᵥuₕ = @. vgradc2f(cuₕ.components.data.:1)
    ᶜ∇ᵥw = @. ∂c(fw.components.data.:1)
    ᶠ∇ᵥh_tot = @. vgradc2f(h_tot)

    ᶜ∇ₕuₕ = @. hgrad(cuₕ.components.data.:1)
    ᶠ∇ₕw = @. hgrad(fw.components.data.:1)
    ᶜ∇ₕh_tot = @. hgrad(h_tot)

    hκ₂∇²uₕ = @. hwdiv(κ₂ * ᶜ∇ₕuₕ)
    vκ₂∇²uₕ = @. vdivf2c(κ₂ * ᶠ∇ᵥuₕ)
    hκ₂∇²w = @. hwdiv(κ₂ * ᶠ∇ₕw)
    vκ₂∇²w = @. vdivc2f(κ₂ * ᶜ∇ᵥw)
    hκ₂∇²h_tot = @. hwdiv(cρ * κ₂ * ᶜ∇ₕh_tot)
    vκ₂∇²h_tot = @. vdivf2c(fρ * κ₂ * ᶠ∇ᵥh_tot)

    dfw = dY.w.components.data.:1
    dcu = dY.uₕ.components.data.:1

    # Laplacian Diffusion (Uniform)
    @. dcu += hκ₂∇²uₕ
    @. dcu += vκ₂∇²uₕ
    @. dfw += hκ₂∇²w
    @. dfw += vκ₂∇²w
    @. dρe += hκ₂∇²h_tot
    @. dρe += vκ₂∇²h_tot

    # Sponge tendency
    β = @. rayleigh_sponge(z)
    @. duₕ -= β * (uₕ - u₀)
    @. dw -= Ic2f(β) * fw

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end

dYdt = similar(Y);
rhs_invariant!(dYdt, Y, nothing, 0.0);

# run!
using OrdinaryDiffEq
Δt = 0.1
timeend = 5000.0
function make_dss_func()
  _dss!(x::Fields.Field)=Spaces.weighted_dss!(x)
  _dss!(::Any)=nothing
  dss_func(Y,t,integrator) = foreach(_dss!,Fields._values(Y))
  return dss_func
end
dss_func = make_dss_func()
dss_callback = FunctionCallingCallback(dss_func, func_start=true)
prob = ODEProblem(rhs_invariant!, Y, (0.0, timeend))
integrator = OrdinaryDiffEq.init(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 100.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
    callback = dss_callback
);

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev OrdinaryDiffEq.solve!(integrator)

ENV["GKSwstype"] = "nul"
import Plots, ClimaCorePlots
Plots.GRBackend()

dir = "schar_tracer_noslip"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρe ./ u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "total_energy.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρq ./ u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 20)

If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    ᶜuw = @. Geometry.Covariant13Vector(u.uₕ) +
       Geometry.Covariant13Vector(If2c(u.w))
    w = @. Geometry.project(Geometry.WAxis(), ᶜuw)
    Plots.plot(w)
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    ᶜuw = @. Geometry.Covariant13Vector(u.uₕ) +
       Geometry.Covariant13Vector(If2c(u.w))
    u = @. Geometry.project(Geometry.UAxis(), ᶜuw)
    Plots.plot(u)
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

# post-processing
Es = [sum(u.Yc.ρe) for u in sol.u]
Mass = [sum(u.Yc.ρ) for u in sol.u]

Plots.png(
    Plots.plot((Es .- energy_0) ./ energy_0),
    joinpath(path, "energy_cons.png"),
)
Plots.png(
    Plots.plot((Mass .- mass_0) ./ mass_0),
    joinpath(path, "mass_cons.png"),
)


function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "energy_cons.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
linkfig(
    relpath(joinpath(path, "mass_cons.png"), joinpath(@__DIR__, "../..")),
    "Mass",
)
