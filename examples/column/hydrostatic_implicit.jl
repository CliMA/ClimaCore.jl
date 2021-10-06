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

using OrdinaryDiffEq:
    OrdinaryDiffEq,
    ODEProblem,
    ODEFunction,
    solve,
    SSPRK33,
    Rosenbrock23,
    ImplicitEuler

using Logging: global_logger
using TerminalLoggers: TerminalLogger
using LinearAlgebra
global_logger(TerminalLogger())

using UnPack

include("../implicit_solver_utils.jl")

const FT = Float64

# https://github.com/CliMA/CLIMAParameters.jl/blob/master/src/Planet/planet_parameters.jl#L5
const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacit at constant volume
const R_m = R_d # moist R, assumed to be dry


domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0.0),
    Geometry.ZPoint{FT}(30e3),
    boundary_tags = (:bottom, :top),
)
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
Φ(z) = grav * z

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
Yc = decaying_temperature_profile.(zc.z)
w = Geometry.Cartesian3Vector.(zeros(FT, fspace))
Y_init = copy(Yc)
w_init = copy(w)
Y = Fields.FieldVector(Yc = Yc, w = w)

function tendency!(dY, Y, _, t)
    Yc = Y.Yc
    w = Y.w
    dYc = dY.Yc
    dw = dY.w

    If = Operators.InterpolateC2F()
    ∂ = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
    )
    ∂f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
    )

    @. dYc.ρ = -(∂(w * If(Yc.ρ)))
    @. dYc.ρθ = -(∂(w * If(Yc.ρθ)))
    @. dw = B(
        Geometry.CartesianVector(
            -(If(Yc.ρθ / Yc.ρ) * ∂f(Π(Yc.ρθ))) - ∂f(Φ(zc.z)),
        ),
    )
    return dY
end

struct CustomWRepresentation{T,AT1,AT2,AT3}
    # reference to dtγ, which is specified by the ODE solver
    dtγ_ref::T

    # cache for the grid values used to compute the Jacobian
    Δz::AT1
    Δzf::AT1

    # cache for the variable values used to compute the Jacobian
    ρf::AT1
    ρθf::AT1
    Πf::AT1

    # nonzero blocks of the Jacobian (∂ρₜ/∂w, ∂ρθₜ/∂w, ∂wₜ/∂ρ, and ∂wₜ/∂ρθ)
    Jρ_w::AT2
    Jρθ_w::AT2
    Jw_ρ::AT2
    Jw_ρθ::AT2

    # cache for the Schur complement
    S::AT3
end

function CustomWRepresentation(FT = Float64)
    N = length(cspace)

    dtγ_ref = Ref(zero(FT))

    zf = parent(Fields.coordinate_field(fspace).z)
    Δz = zf[2:N + 1] - zf[1:N]
    zc = parent(Fields.coordinate_field(cspace).z)
    Δzf = zc[2:N] - zc[1:N - 1]

    ρf = Array{FT}(undef, N + 1)
    ρθf = Array{FT}(undef, N + 1)
    Πf = Array{FT}(undef, N + 1)

    Jρ_w = GeneralBidiagonal(Array{FT}, true, N, N + 1)
    Jρθ_w = GeneralBidiagonal(Array{FT}, true, N, N + 1)
    Jw_ρ = GeneralBidiagonal(Array{FT}, false, N + 1, N)
    Jw_ρθ = GeneralBidiagonal(Array{FT}, false, N + 1, N)

    S = Tridiagonal(
        Array{FT}(undef, N),
        Array{FT}(undef, N + 1),
        Array{FT}(undef, N),
    )

    CustomWRepresentation{typeof(dtγ_ref),typeof(ρf),typeof(Jρ_w),typeof(S)}(
        dtγ_ref,
        Δz,
        Δzf,
        ρf,
        ρθf,
        Πf,
        Jρ_w,
        Jρθ_w,
        Jw_ρ,
        Jw_ρθ,
        S,
    )
end

import Base: similar
# We only use Wfact, but the Rosenbrock23 solver requires us to pass
# jac_prototype, then calls similar(jac_prototype) to obtain J and Wfact. This
# is a temporary workaround to avoid unnecessary allocations.
Base.similar(cf::CustomWRepresentation{T,AT}) where {T, AT} = cf

function Wfact!(W, u, p, dtγ, t)
    @unpack dtγ_ref, Δz, Δzf, ρf, ρθf, Πf, Jρ_w, Jρθ_w, Jw_ρ, Jw_ρθ = W
    dtγ_ref[] = dtγ

    N = length(cspace)
    ρ = reshape(parent(u.Yc.ρ), N)
    ρθ = reshape(parent(u.Yc.ρθ), N)

    # Compute the variable values
    
    ρf[1] = ρ[1]
    @views @. ρf[2:N] = (ρ[1:N - 1] + ρ[2:N]) / 2
    ρf[N + 1] = ρ[N]

    ρθf[1] = ρθ[1]
    @views @. ρθf[2:N] = (ρθ[1:N - 1] + ρθ[2:N]) / 2
    ρθf[N + 1] = ρθ[N]

    @views @. Πf[1:N] = Π(ρθ) # temporarily store cell-center values in Πf
    @views @. Πf[2:N] = (Πf[1:N - 1] + Πf[2:N]) / 2

    # Compute the nonzero blocks of the Jacobian
    
    @views @. Jρ_w.d = ρf[1:N] / Δz
    @views @. Jρ_w.d2 = -ρf[2:N + 1] / Δz

    @views @. Jρθ_w.d = ρθf[1:N] / Δz
    @views @. Jρθ_w.d2 = -ρθf[2:N + 1] / Δz

    Jw_ρ.d[1] = 0
    Jw_ρ.d2[N] = 0
    @views @. Jw_ρ.d[2:N] = -grav / (2 * ρf[2:N])
    @views @. Jw_ρ.d2[1:N - 1] = Jw_ρ.d[2:N]

    # TODO: this is wrong
    Jw_ρθ.d[1] = 0
    Jw_ρθ.d2[N] = 0
    @views @. Jw_ρθ.d[2:N] = -(γ - 1) * Πf[2:N] / (ρf[2:N] * Δzf)
    @views @. Jw_ρθ.d2[1:N - 1] = -Jw_ρθ.d[2:N]
end

function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        @unpack dtγ_ref, Jρ_w, Jρθ_w, Jw_ρ, Jw_ρθ, S = A
        dtγ = dtγ_ref[]

        N = length(cspace)
        schur_solve!(
            reshape(parent(x.Yc.ρ), N),
            reshape(parent(x.Yc.ρθ), N),
            reshape(parent(x.w), N + 1),
            Jρ_w,
            Jρθ_w,
            Jw_ρ,
            Jw_ρθ,
            reshape(parent(b.Yc.ρ), N),
            reshape(parent(b.Yc.ρθ), N),
            reshape(parent(b.w), N + 1),
            dtγ,
            S,
        )
    end
end

Δt = 100.
ndays = 10.

# Solve the ODE operator
prob = ODEProblem(
    ODEFunction(
        tendency!,
        Wfact = Wfact!,
        jac_prototype = CustomWRepresentation(),
        tgrad = (dT, Y, p, t) -> fill!(dT, 0),
    ),
    Y,
    (0.0, 60 * 60 * 24 * ndays),
)

sol = solve(
    prob,
    Rosenbrock23(linsolve = linsolve!),
    dt = Δt,
    adaptive = false,
    saveat = 60 * 60, # save every hour
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "hydrostatic_implicit"
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
        xlim = (-0.2, 0.2),
        xlabel = "ω",
        label = "T=0",
    )
    sub_plt2 = Plots.plot!(sub_plt2, parent(u.w), z_faces, label = "T")

    sub_plt3 = Plots.plot(
        parent(Yc_init.ρθ),
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

# anim = Plots.@animate for (i, u) in enumerate(sol.u)
#     hydrostatic_plot(u, Y_init, w_init, title = "Hour $(i)")
# end
# Plots.mp4(anim, joinpath(path, "hydrostatic.mp4"), fps = 10)

Plots.png(
    hydrostatic_plot(sol[end], Y_init, w_init),
    joinpath(path, "hydrostatic_end.png"),
)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/hydrostatic_end.png", "Hydrostatic End")
