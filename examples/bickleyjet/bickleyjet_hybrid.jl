push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using StaticArrays
using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: 
    ClimaCore, 
    Fields, 
    Domains,
    Topologies,
    Meshes, 
    Spaces, 
    slab, 
    Operators, 
    Geometry
using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using ClimaCore.RecursiveApply
using ClimaCore.RecursiveApply: rdiv, rmap

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 50.0, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    MSLP = 1e5,
    γ = 1.4,
    c = 2,
    g = 10,
    R_d = 287.058,
    C_p = 1004.703,
    C_v = 717.645
   )

numflux_name = get(ARGS, 1, "roe")
boundary_name = get(ARGS, 2, "")

n1, n2 = 32, 32
Nq = 1
Nqh = 1
# set up function space
function hvspace_2D(
    xlim = (-100, 100),
    zlim = (0, 100),
    helem = 32,
    velem = 32,
    npoly = 1,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1])..Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)

    quad = Spaces.Quadratures.GL{npoly}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

hv_center_space, hv_face_space = hvspace_2D()

function pressure(ρθ, p)
    FT = eltype(ρθ)
    if ρθ >= 0
      return FT(p.MSLP * (p.R_d * ρθ / p.MSLP)^p.γ)
    else
      return NaN
    end
end

Φ(z,p) = p.g * z 

# horizontal flux terms
function flux(state, p)
    @unpack ρ, ρuₕ, ρθ = state
    u = ρuₕ / ρ
    return (ρ = ρuₕ, ρuₕ = ((ρuₕ ⊗ u) + (p.g * ρ ^ 2 / 2) * I), ρθ = ρθ * u)
end

# numerical fluxes
wavespeed(y, parameters) = sqrt(parameters.g)
roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))
function roeflux(n, (y⁻, parameters⁻), (y⁺, parameters⁺))
    Favg = rdiv(flux(y⁻, parameters⁻) ⊞ flux(y⁺, parameters⁺), 2)

    λ = sqrt(parameters⁻.g)

    ρ⁻, ρuₕ⁻, ρθ⁻ = y⁻.ρ, y⁻.ρuₕ, y⁻.ρθ
    ρ⁺, ρuₕ⁺, ρθ⁺ = y⁺.ρ, y⁺.ρuₕ, y⁺.ρθ
  
    u⁻ = ρuₕ⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n

    u⁺ = ρuₕ⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n

    # in general thermodynamics, (pressure, soundspeed)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)

    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!
    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρuₕ =
        (w1 * (u - c * n) + w2 * (u + c * n) + w3 * u + w4 * (Δu - Δuₙ * n)) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ + w5) * 0.5
    Δf = (ρ = -fluxᵀn_ρ, ρuₕ = -fluxᵀn_ρuₕ, ρθ = -fluxᵀn_ρθ)
    rmap(f -> f' * n, Favg) ⊞ Δf
end

function roeflux_f(n, (y⁻, parameters⁻), (y⁺, parameters⁺))
    Favg = rdiv(flux(y⁻, parameters⁻) ⊞ flux(y⁺, parameters⁺), 2)

    λ = sqrt(parameters⁻.g)

    ρ⁻, ρuₕ⁻, ρθ⁻ = y⁻.ρ, y⁻.ρuₕ, y⁻.ρθ
    ρ⁺, ρuₕ⁺, ρθ⁺ = y⁺.ρ, y⁺.ρuₕ, y⁺.ρθ

    u⁻ = ρuₕ⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n

    u⁺ = ρuₕ⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n

    # in general thermodynamics, (pressure, soundspeed)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)

    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!
    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρuₕ =
        (w1 * (u - c * n) + w2 * (u + c * n) + w3 * u + w4 * (Δu - Δuₙ * n)) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ + w5) * 0.5
    Δf = (ρ = -fluxᵀn_ρ, ρuₕ = -fluxᵀn_ρuₕ, ρθ = -fluxᵀn_ρθ)
    rmap(f -> f' * n, Favg) ⊞ Δf
end

numflux = roeflux

function rhs!(dY, Y, (parameters, numflux), t)

    # Unpack Staggered Fields @unpack
    ρw = Y.ρw # Face variables
    Yc = Y.Yc # Center Variables
    dYc = dY.Yc # Center Tendencies
    dρw = dY.ρw # Face Tendencies

    rparams = Ref(parameters)

    # Horizontal Flux/Source Contributions
    wdiv = Operators.WeakDivergence()
    local_geometry_field_c = Fields.local_geometry_field(Yc.ρ)
    local_geometry_field_f = Fields.local_geometry_field(ρw)
    dYc .= wdiv.(flux.(Yc, rparams)) .* (.-(local_geometry_field_c.WJ))
    Operators.add_numerical_flux_internal_horizontal!(numflux, dYc, Yc, parameters)
    
    # Vertical Flux/Source Contributions
    vdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(0.0)),
        top = Operators.SetDivergence(Geometry.WVector(0.0)),
    )
    uvdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0),
        ),
        top = Operators.SetValue(Geometry.WVector(0.0) ⊗ Geometry.UVector(0.0)),
    )
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Ic = Operators.InterpolateF2C()
    ∂ = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    ∂f = Operators.GradientC2F()
    ∂c = Operators.GradientF2C()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    fcc = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    uₕ = @. Yc.ρuₕ / Yc.ρ
    w = @. ρw / If(Yc.ρ)
    wc = @. Ic(ρw) / Yc.ρ
    p = @. pressure(Yc.ρθ,rparams)
    θ = @. Yc.ρθ / Yc.ρ
    Yfρ = @. If(Yc.ρ)
    # density
    @. dYc.ρ -= ∂(ρw)
    # potential temperature
    @. dYc.ρθ += -(∂(ρw * If(Yc.ρθ / Yc.ρ)))
    # horizontal momentum
    @. dYc.ρuₕ += -uvdivf2c(ρw ⊗ If(uₕ))
    # vertical momentum
    #@. dρw +=
    #    B(
    #        Geometry.transform(
    #            Geometry.WAxis(),
    #            -(∂f(p)) - If(Yc.ρ) * ∂f(Φ(coords.z, rparams)),
    #        ) - vvdivc2f(Ic(ρw ⊗ w)),
    #    )
    uₕf = @. If(Yc.ρuₕ / Yc.ρ) # requires boundary conditions
    upwind_correction = false
    if upwind_correction
        @. dYc.ρ += fcc(w, Yc.ρ)
        @. dYc.ρθ += fcc(w, Yc.ρθ)
        @. dYc.ρuₕ += fcc(w, Yc.ρuₕ)
        @. dρw += fcf(wc, ρw)
    end
    ### DIFFUSION
    κ₂ = 0.0 # m^2/s
    #  1b) vertical div of vertical grad of horiz momentun
    @. dYc.ρuₕ += uvdivf2c(κ₂ * (Yfρ * ∂f(Yc.ρuₕ / Yc.ρ)))
    #  1d) vertical div of vertical grad of vert momentun
    @. dρw += vvdivc2f(κ₂ * (Yc.ρ * ∂c(ρw / Yfρ)))
    #  2b) vertical div of vertial grad of potential temperature
    @. dYc.ρθ += ∂(κ₂ * (Yfρ * ∂f(Yc.ρθ / Yc.ρ)))
    return dY
end


# unpack coordinate fields
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)
# set initial conditions
# Hybrid - 1+1D Box configuration 
function init_state(x,z,p)
    # set initial state
    ρ = p.ρ₀ 
    # set initial velocity
    U₁ = cosh(z)^(-2)
    u = Geometry.UVector(10.0)
    θ = 300.0
    π_exn = 1.0 - p.g * z / p.C_p / θ # exner function
    T = π_exn * θ # temperature
    pres = p.MSLP * π_exn ^ (p.C_p / p.R_d) # pressure
    ρ = pres / p.R_d / T # density
    ρuₕ = ρ * u
    ρθ = ρ * θ
    return (ρ = ρ, ρuₕ = ρuₕ, ρθ = ρθ)
end

Y_center = init_state.(coords.x, coords.z, Ref(parameters))
ρw = map(face_coords) do coord
    Geometry.WVector(0.0)
end

Y = Fields.FieldVector(Yc = Y_center, ρw = ρw)
dY = similar(Y)
test = rhs!(dY,Y,(parameters,numflux), 0.0)
dt = eps(eltype(Y))
timeend = dt*3
# Solve the ODE operator
prob = ODEProblem(rhs!, Y, (0.0, timeend), (parameters, numflux))
sol = solve(
    prob,
    SSPRK33(),
    dt = dt,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)
#=
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "dg_$(numflux_name)"
if boundary_name != ""
    dirname = "$(dirname)_$(boundary_name)"
end
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]
Plots.png(Plots.plot(Es), joinpath(path, "energy.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
=#
