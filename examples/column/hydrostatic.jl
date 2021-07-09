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


domain = Domains.IntervalDomain(0.0, 30e3, x3boundary = (:bottom, :top))
#mesh = Meshes.IntervalMesh(domain, Meshes.ExponentialStretching(7.5e3); nelems = 30)
mesh = Meshes.IntervalMesh(domain; nelems = 30)

cspace = Spaces.CenterFiniteDifferenceSpace(mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

# https://github.com/CliMA/Thermodynamics.jl/blob/main/src/TemperatureProfiles.jl#L115-L155
# https://clima.github.io/Thermodynamics.jl/dev/TemperatureProfiles/#DecayingTemperatureProfile
function decaying_temperature_profile(z; T_virt_surf = 280.0, T_min_ref = 230.0)
    # Scale height for surface temperature
    H_sfc = R_d * T_virt_surf / grav
    H_t = H_sfc

    z′ = z / H_t
    tanh_z′ = tanh(z′)

    ΔTv = T_virt_surf - T_min_ref
    Tv = T_virt_surf - ΔTv * tanh_z′

    ΔTv′ = ΔTv / T_virt_surf
    p =
        MSLP * exp(
            (
                -H_t *
                (z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
            ) / (H_sfc * (1 - ΔTv′^2)),
        )
    ρ = p / (R_d * Tv)
    ρθ = ρ * Tv * (MSLP / p)^(R_m / C_p)
    return (ρ = ρ, ρθ = ρθ)
end

Π(ρθ) = C_p * (R_d * ρθ / MSLP)^(R_m / C_v)


function discrete_hydrostatic_balance!(
    ρ,
    w,
    ρθ,
    Δz::Float64,
    _grav::Float64,
    Π::Function,
)
    # compute θ such that
    #   I(θ)[i+1/2] = -g / ∂f(Π(ρθ))
    # discretely, then set
    #   ρ = ρθ/θ
    for i in 1:(length(ρ) - 1)
        #  ρ[i+1] = ρθ[i+1]/(-2Δz*_grav/(Π(ρθ[i+1]) - Π(ρθ[i])) - ρθ[i]/ρ[i])
        ρ[i + 1] =
            ρθ[i + 1] /
            (-2 * _grav / ((Π(ρθ[i + 1]) - Π(ρθ[i])) / Δz) - ρθ[i] / ρ[i])

        ρ[i + 1] =
            ρθ[i + 1] /
            (1 / ((-2 * _grav) * (Π(ρθ[i + 1]) - Π(ρθ[i]))Δz) - ρθ[i] / ρ[i])

        ∂Π∂z = (Π(ρθ[i + 1]) - Π(ρθ[i])) / Δz
    end
end

zc = Fields.coordinate_field(cspace)
Yc = decaying_temperature_profile.(zc)
w = zeros(Float64, fspace)

Y_init = copy(Yc)
w_init = copy(w)

Y = (Yc, w)

function tendency!(dY, Y, _, t)
    (Yc, w) = Y.x
    (dYc, dw) = dY.x

    If = Operators.InterpolateC2F()

    ∂c = Operators.GradientF2C(
        bottom = Operators.SetValue((ρ = 0.0, ρθ = 0.0)),
        top = Operators.SetValue((ρ = 0.0, ρθ = 0.0)),
    )

    ∂f = Operators.GradientC2F()

    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(0.0),
        top = Operators.SetValue(0.0),
    )

    # S 4.4.1: potential temperature density
    @. dYc = -(∂c(w * If(Yc)))
    @. dw = B(-(If(Yc.ρθ / Yc.ρ) * ∂f(Π(Yc.ρθ))) - grav)

    return dY
end


using RecursiveArrayTools

Y = ArrayPartition(Yc, w)
dY = tendency!(similar(Y), Y, nothing, 0.0)

Δt = 1.0
ndays = 10

# Solve the ODE operator
prob = ODEProblem(tendency!, Y, (0.0, 60 * 60 * 24 * ndays))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 60 * 60, # save every hour
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "hydrostatic"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

z_centers = parent(Fields.coordinate_field(cspace))
z_faces = parent(Fields.coordinate_field(fspace))

function hydrostatic_plot(u; title = "", size = (1024, 600))
    sub_plt1 = Plots.plot(
        parent(Y_init.ρ),
        z_centers,
        marker = :circle,
        xlabel = "ρ",
        label = "T=0",
    )
    sub_plt1 = Plots.plot!(sub_plt1, parent(u.x[1].ρ), z_centers, label = "T")

    sub_plt2 = Plots.plot(
        parent(w_init),
        z_faces,
        marker = :circle,
        xlim = (-0.2, 0.2),
        xlabel = "ω",
        label = "T=0",
    )
    sub_plt2 = Plots.plot!(sub_plt2, parent(u.x[2]), z_faces, label = "T")

    sub_plt3 = Plots.plot(
        parent(Y_init.ρθ),
        z_centers,
        marker = :circle,
        xlabel = "ρθ",
        label = "T=0",
    )
    sub_plt3 = Plots.plot!(sub_plt3, parent(u.x[1].ρθ), z_centers, label = "T")

    return Plots.plot(
        sub_plt1,
        sub_plt2,
        sub_plt3,
        title = title,
        layout = (1, 3),
        size = size,
    )
end

anim = Plots.@animate for (i, u) in enumerate(sol.u)
    hydrostatic_plot(u, title = "Hour $(i)")
end
Plots.mp4(anim, joinpath(path, "hydrostatic.mp4"), fps = 10)

Plots.png(hydrostatic_plot(sol[end]), joinpath(path, "hydrostatic_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/hydrostatic_end.png", "Hydrostatic End")
