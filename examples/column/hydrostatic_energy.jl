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
    E = p/(γ - 1)
    return (ρ = ρ, E = E)
end


zc = Fields.coordinate_field(cspace)
zf = Fields.coordinate_field(fspace)
Yc = decaying_temperature_profile.(zc)
w = zeros(Float64, fspace)
Y_init = copy(Yc)
w_init = copy(w)
Y = Fields.FieldVector(Yc = Yc, w = w)


function tendency!(dY, Y, _, t)
    Yc = Y.Yc
    w = Y.w
    dYc = dY.Yc
    dw = dY.w


    UnPack.@unpack ρ, E = Yc
    dρ = dYc.ρ
    dE = dYc.E
    

    If = Operators.InterpolateC2F(;
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    ∂c = Operators.GradientF2C(
        bottom = Operators.SetValue(0.0),
        top = Operators.SetValue(0.0),
    )
    # density equation
    @. dρ = -(∂c(w * If(ρ)))

    # energy equation
    Ef = If(E)
    ρf = If(ρ)
    pf = (γ - 1).*(Ef .- 0.5.*ρf.*w.*w)

    
    @. dE = -(∂c(w * (Ef + pf)))

    # w equation
    Ic = Operators.InterpolateF2C()

    ∂f = Operators.GradientC2F()

    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(0.0),
        top = Operators.SetValue(0.0),
    )

    pc = (γ - 1).*(E .- 0.5.*ρ.*Ic(w).^2)

    @. dw = B(- w .* If(∂c(w)) - If(1 ./ ρ) .* ∂f(pc) .- grav)

    return dY
end


using RecursiveArrayTools

# Y = ArrayPartition(Yc, w)

Δt = 1.0
ndays = 1

# Solve the ODE operator
prob = ODEProblem(tendency!, Y, (0.0, 60 * 60* 24 * ndays))
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

function hydrostatic_plot(u, Yc_init, w_init; title = "", size = (1024, 600))
    sub_plt1 = Plots.plot(
        parent(Yc_init.ρ),
        z_centers,
        marker = :circle,
        xlabel = "ρ",
        label = "T=0",
    )
    sub_plt1 = Plots.plot!(sub_plt1, parent(u.Yc.ρ), z_centers, label = "T")

    sub_plt2 = Plots.plot(
        parent(w_init),
        z_faces,
        marker = :circle,
        xlabel = "ω",
        label = "T=0",
    )
    sub_plt2 = Plots.plot!(sub_plt2, parent(u.w), z_faces, label = "T")

    sub_plt3 = Plots.plot(
        parent(Yc_init.E),
        z_centers,
        marker = :circle,
        xlabel = "E",
        label = "T=0",
    )
    sub_plt3 = Plots.plot!(sub_plt3, parent(u.Yc.E), z_centers, label = "T")

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
    hydrostatic_plot(u, Y_init, w_init,  title = "Hour $(i)")
end
Plots.mp4(anim, joinpath(path, "hydrostatic_energy.mp4"), fps = 10)

Plots.png(hydrostatic_plot(sol[end], Y_init, w_init), joinpath(path, "hydrostatic_energy_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/hydrostatic_energy_end.png", "Hydrostatic Energy End")
