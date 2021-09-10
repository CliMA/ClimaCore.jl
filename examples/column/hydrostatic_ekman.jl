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

import ClimaCore.Geometry: ⊗

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
const uvg = Geometry.Cartesian12Vector(ug, vg)
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

    uv = Geometry.Cartesian12Vector(FT(1.0), FT(0.0))
    return (ρ = ρ, uv = uv, ρθ = ρθ)
end

Π(ρθ) = C_p * (R_d * ρθ / MSLP)^(R_m / C_v)
Φ(z) = grav * z

zc = Fields.coordinate_field(cspace)
Yc = adiabatic_temperature_profile.(zc)
w = Geometry.Cartesian3Vector.(zeros(FT, fspace))

Y_init = copy(Yc)
w_init = copy(w)

# Y = (Yc, w)

function tendency!(dY, Y, _, t)
    (Yc, w) = Y.x
    (dYc, dw) = dY.x

    UnPack.@unpack ρ, uv, ρθ = Yc
    dρ = dYc.ρ
    duv = dYc.uv
    dρθ = dYc.ρθ

    # density
    If = Operators.InterpolateC2F()
    ∂f = Operators.GradientC2F()
    ∂c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
    )
    @. dρ = -∂c(w * If(ρ))

    # potential temperature
    If = Operators.InterpolateC2F()
    ∂f = Operators.GradientC2F()
    ∂c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
    )
    # TODO!: Undesirable casting to vector required
    @. dρθ = -∂c(w * If(ρθ)) + ρ * ∂c(Geometry.CartesianVector(ν * ∂f(ρθ / ρ)))

    uv_1 = Operators.getidx(uv, Operators.Interior(), 1)
    u_wind = LinearAlgebra.norm(uv_1)

    A = Operators.AdvectionC2C(
        bottom = Operators.SetValue(Geometry.Cartesian12Vector(0.0, 0.0)),
        top = Operators.SetValue(Geometry.Cartesian12Vector(0.0, 0.0)),
    )

    # uv
    bcs_bottom =
        Operators.SetValue(Geometry.Cartesian3Vector(Cd * u_wind) ⊗ uv_1)
    bcs_top = Operators.SetValue(uvg)
    ∂c = Operators.DivergenceF2C(bottom = bcs_bottom)
    ∂f = Operators.GradientC2F(top = bcs_top)
    duv .= (uv .- Ref(uvg)) .× Ref(Geometry.Cartesian3Vector(f))
    @. duv += ∂c(ν * ∂f(uv)) - A(w, uv)

    # w
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ∂f = Operators.GradientC2F()
    ∂c = Operators.GradientF2C()
    Af = Operators.AdvectionF2F()
    divf = Operators.DivergenceC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
    )
    @. dw = B(
        Geometry.CartesianVector(
            -(If(Yc.ρθ / Yc.ρ) * ∂f(Π(Yc.ρθ))) - ∂f(Φ(zc)),
        ) + divf(ν * ∂c(w)) - Af(w, w),
    )

    return dY
end

using LinearAlgebra
using RecursiveArrayTools

Y = ArrayPartition(Yc, w)
dY = tendency!(similar(Y), Y, nothing, 0.0)

Δt = 1.0 / 100.0
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
    # get u component of uv vector
    sub_plt1 = Plots.plot!(
        sub_plt1,
        parent(u.x[1].uv.components.data.:1),
        z_centers,
        label = "Comp",
    )

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
    # get v component of uv vector
    sub_plt2 = Plots.plot!(
        sub_plt2,
        parent(u.x[1].uv.components.data.:2),
        z_centers,
        label = "Comp",
    )


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


# w = ClimaCore.Geometry.AxisTensor{
#         Float64,
#         1,
#         Tuple{ClimaCore.Geometry.CartesianAxis{(3,)}},
#         StaticArrays.SVector{
#             1,
#             Float64
#         }
#     }

# If.(∂c.(w)) = ClimaCore.Geometry.AxisTensor{
#         ClimaCore.Geometry.AxisTensor{
#             Float64,
#             1,
#             Tuple{ClimaCore.Geometry.CartesianAxis{(3,)}},
#             StaticArrays.SVector{
#                 1,
#                 Float64
#             }
#         },
#         1,
#         Tuple{ClimaCore.Geometry.CovariantAxis{(3,)}},
#         StaticArrays.SVector{
#             1,
#             ClimaCore.Geometry.AxisTensor{
#                 Float64,
#                 1,
#                 Tuple{ClimaCore.Geometry.CartesianAxis{(3,)}},
#                 StaticArrays.SVector{
#                     1,
#                     Float64
#                 }
#             }
#         }
#     }
