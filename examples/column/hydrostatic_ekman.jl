push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

import ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

using OrdinaryDiffEq: OrdinaryDiffEq, ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

const FT = Float64

# https://github.com/CliMA/CLIMAParameters.jl/blob/master/src/Planet/planet_parameters.jl#L5
const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacit at constant volume
const R_m = R_d # moist R, assumed to be dry

const f = 5e-5
const ν = 0.01
const L = 2e2
const nelems = 30
const Cd = ν / (L / nelems)
const ug = 1.0
const vg = 0.0
const d = sqrt(2 * ν / f)
domain = Domains.IntervalDomain(0.0, L; x3boundary = (:bottom, :top))
#mesh = Meshes.IntervalMesh(domain, Meshes.ExponentialStretching(7.5e3); nelems = 30)
mesh = Meshes.IntervalMesh(domain; nelems = nelems)

cspace = Spaces.CenterFiniteDifferenceSpace(mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)


function ρ_from_pθ(p, θ_liq)
    T = θ_liq * (p / MSLP)^(R_d / C_p)
    return p / (R_d * T)
end

# https://github.com/CliMA/Thermodynamics.jl/blob/main/src/TemperatureProfiles.jl#L115-L155
# https://clima.github.io/Thermodynamics.jl/dev/TemperatureProfiles/#DecayingTemperatureProfile
function adiabatic_temperature_profile(z; T_surf = 300.0, T_min_ref = 230.0)
    ## Initial conditions
    # Scale height for surface temperature



    # Temperature
    Γ = grav / C_p
    T = max(T_surf - Γ * z, T_min_ref)

    # Pressure
    p = MSLP * (T / T_surf)^(grav / (R_d * Γ))
    if T == T_min_ref
        z_top = (T_surf - T_min_ref) / Γ
        H_min = R_d * T_min_ref / grav
        p *= exp(-(z - z_top) / H_min)
    end


    θ = FT(T_surf)

    ρ = ρ_from_pθ(p, θ)
    ρθ = ρ * θ

    u = FT(1.0)
    v = FT(0.0)
    return (ρ = ρ, u = u, v = v, ρθ = ρθ)
end

Π(ρθ) = C_p * (R_d * ρθ / MSLP)^(R_m / C_v)



zc = Fields.coordinate_field(cspace)
Yc = adiabatic_temperature_profile.(zc)
w = zeros(Float64, fspace)

Y_init = copy(Yc)
w_init = copy(w)

# Y = (Yc, w)

function tendency!(dY, Y, _, t)

    (Yc, w) = Y.x
    (dYc, dw) = dY.x

    UnPack.@unpack ρ, u, v, ρθ = Yc
    dρ = dYc.ρ
    du = dYc.u
    dv = dYc.v
    dρθ = dYc.ρθ

    # S 4.4.1: potential temperature density
    # Mass conservation

    If = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ∂c = Operators.GradientF2C()
    @. dρ = -(∂c(w * If(ρ)))  # Eq. 4.11


    u_1 = parent(u)[1]
    v_1 = parent(v)[1]
    u_wind = sqrt(u_1^2 + v_1^2)
    A = Operators.AdvectionC2C(
        bottom = Operators.SetValue(0.0),
        top = Operators.SetValue(0.0),
    )

    # u-momentum
    bcs_bottom = Operators.SetValue(Cd * u_wind * u_1)  # Eq. 4.16
    bcs_top = Operators.SetValue(FT(ug))  # Eq. 4.18
    gradc2f = Operators.GradientC2F(top = bcs_top)
    gradf2c = Operators.GradientF2C(bottom = bcs_bottom)
    @. du = gradf2c(ν * gradc2f(u)) + f * (v - vg) - A(w, u) # Eq. 4.8

    # v-momentum
    bcs_bottom = Operators.SetValue(Cd * u_wind * v_1)  # Eq. 4.17
    bcs_top = Operators.SetValue(FT(vg))  # Eq. 4.19
    gradc2f = Operators.GradientC2F(top = bcs_top)
    gradf2c = Operators.GradientF2C(bottom = bcs_bottom)
    @. dv = gradf2c(ν * gradc2f(v)) - f * (u - ug) - A(w, v) # Eq. 4.9

    # w-momentum
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(0.0),
        top = Operators.SetValue(0.0),
    )
    # gradc2f = Operators.GradientC2F(;bottom =  Operators.SetValue(0.0), top = Operators.SetValue(0.0))
    gradc2f = Operators.GradientC2F()
    If = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    gradf2c = Operators.GradientF2C()
    @. dw = B(
        -(If(ρθ / ρ) * gradc2f(Π(ρθ))) - grav + gradc2f(ν * gradf2c(w)) -
        w * If(gradf2c(w)),
    )   # Eq. 4.10


    # potential temperature density conservation
    bcs_bottom = Operators.SetValue(FT(0.0))  # Eq. 4.20
    bcs_top = Operators.SetValue(FT(0.0))  # Eq. 4.21
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C(bottom = bcs_bottom, top = bcs_top)
    @. dρθ = -(∂c(w * If(ρθ))) + gradf2c(ν * gradc2f(ρθ / ρ))  # Eq. 4.12

    return dY
end

using LinearAlgebra
using RecursiveArrayTools

Y = ArrayPartition(Yc, w)
dY = tendency!(similar(Y), Y, nothing, 0.0)

Δt = 1.0 / 100.0
ndays = 0
# Solve the ODE operator
prob = ODEProblem(tendency!, Y, (0.0, 60 * 60))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 600, # save every hour
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "hydrostatic_ekman"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

z_centers = parent(Fields.coordinate_field(cspace))
z_faces = parent(Fields.coordinate_field(fspace))

function ekman_plot(u; title = "", size = (1024, 600))
    u_ref =
        ug .-
        exp.(-z_centers / d) .*
        (ug * cos.(z_centers / d) + vg * sin.(z_centers / d))
    sub_plt1 = Plots.plot(
        u_ref,
        z_centers,
        marker = :circle,
        xlabel = "u",
        label = "Ref",
    )
    sub_plt1 =
        Plots.plot!(sub_plt1, parent(u.x[1].u), z_centers, label = "Comp")

    v_ref =
        vg .+
        exp.(-z_centers / d) .*
        (ug * sin.(z_centers / d) - vg * cos.(z_centers / d))
    sub_plt2 = Plots.plot(
        v_ref,
        z_centers,
        marker = :circle,
        xlabel = "v",
        label = "Ref",
    )
    sub_plt2 =
        Plots.plot!(sub_plt2, parent(u.x[1].v), z_centers, label = "Comp")


    return Plots.plot(
        sub_plt1,
        sub_plt2,
        title = title,
        layout = (1, 2),
        size = size,
    )
end

anim = Plots.@animate for (i, u) in enumerate(sol.u)
    ekman_plot(u, title = "Hour $(i)")
end
Plots.mp4(anim, joinpath(path, "hydrostatic_ekman.mp4"), fps = 10)

Plots.png(ekman_plot(sol[end]), joinpath(path, "hydrostatic_ekman_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/hydrostatic_ekman_end.png", "ekman end")
