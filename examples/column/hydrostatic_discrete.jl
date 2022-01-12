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

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
const FT = Float64

const z_top = FT(30e3)
const n_vert = 30

# https://github.com/CliMA/CLIMAParameters.jl/blob/master/src/Planet/planet_parameters.jl#L5
const MSLP = FT(1e5) # mean sea level pressure
const grav = FT(9.8) # gravitational constant
const R_d = FT(287.058) # R dry (gas constant / mol mass dry air)
const γ = FT(1.4) # heat capacity ratio
const C_p = FT(R_d * γ / (γ - 1)) # heat capacity at constant pressure
const C_v = FT(R_d / (γ - 1)) # heat capacit at constant volume
const R_m = FT(R_d) # moist R, assumed to be dry

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0.0),
    Geometry.ZPoint{FT}(z_top),
    boundary_names = (:bottom, :top),
)
#mesh = Meshes.IntervalMesh(domain, Meshes.ExponentialStretching(7.5e3); nelems = 30)
mesh = Meshes.IntervalMesh(domain; nelems = n_vert)

cspace = Spaces.CenterFiniteDifferenceSpace(mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

# https://github.com/CliMA/Thermodynamics.jl/blob/main/src/TemperatureProfiles.jl#L115-L155
# https://clima.github.io/Thermodynamics.jl/dev/TemperatureProfiles/#DecayingTemperatureProfile
function decaying_temperature_profile(
    z;
    T_virt_surf = FT(280.0),
    T_min_ref = FT(230.0),
)
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
Φ(z) = grav * z

function discrete_hydrostatic_balance!(ρ, ρθ, Δz::FT, _grav::FT, Π::Function)
    for i in 1:(length(ρ) - 1)
        ρ[i + 1] =
            ρθ[i + 1] /
            (-2 * _grav / ((Π(ρθ[i + 1]) - Π(ρθ[i])) / Δz) - ρθ[i] / ρ[i])

    end
end

zc = Fields.coordinate_field(cspace)
zc_vec = parent(zc)

N = length(zc_vec)
ρ = zeros(Float64, N)
ρθ = zeros(Float64, N)

for i = 1:N
    var = decaying_temperature_profile(zc_vec[i]; T_virt_surf = 280.0, T_min_ref = 230.0)
    ρ[i]  = var.ρ
    ρθ[i] = var.ρθ
end

discrete_hydrostatic_balance!(ρ, ρθ, z_top/n_vert, grav, Π)

Yc = decaying_temperature_profile.(zc.z)
parent(Yc.ρ) .= ρ
parent(Yc.ρθ) .= ρθ
w = Geometry.WVector.(zeros(FT, fspace))

Y_init = copy(Yc)
w_init = copy(w)

function tendency!(dY, Y, _, t)
    Yc = Y.Yc
    w = Y.w

    dYc = dY.Yc
    dw = dY.w

    If = Operators.InterpolateC2F()
    ∂ = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(zero(FT))),
        top = Operators.SetValue(Geometry.WVector(zero(FT))),
    )
    ∂f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(zero(FT))),
        top = Operators.SetValue(Geometry.WVector(zero(FT))),
    )

    @. dYc.ρ = -(∂(w * If(Yc.ρ)))
    @. dYc.ρθ = -(∂(w * If(Yc.ρθ)))
    @. dw =
        B(Geometry.WVector(-(If(Yc.ρθ / Yc.ρ) * ∂f(Π(Yc.ρθ))) - ∂f(Φ(zc.z))))
    return dY
end

Y = Fields.FieldVector(Yc = Yc, w = w)
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
using ClimaCorePlots, Plots
Plots.GRBackend()

dirname = "hydrostatic_discretely_balanced"
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
        parent(Y_init.ρθ),
        z_centers,
        marker = :circle,
        xlabel = "ρθ",
        label = "T=0",
    )
    sub_plt3 = Plots.plot!(sub_plt3, parent(u.Yc.ρθ), z_centers, label = "T")

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

linkfig(
    "examples/column/output/$(dirname)/hydrostatic_end.png",
    "Hydrostatic End",
)
