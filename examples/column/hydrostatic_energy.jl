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
    OrdinaryDiffEq, ODEProblem, ODEFunction, solve, SSPRK33, Rosenbrock23

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
    if z >= 1000 && z <= 5000
        Tv = Tv + 5
    end
    ρ = p / (R_d * Tv)
    E = p / (γ - 1)
    return (ρ = ρ, E = E)
end


zc = Fields.coordinate_field(cspace)
zf = parent(Fields.coordinate_field(fspace))
Δz = zf[2:end] - zf[1:(end - 1)]
Yc = decaying_temperature_profile.(zc)
w = zeros(Float64, fspace)
Y_init = copy(Yc)
w_init = copy(w)
Y = Fields.FieldVector(Yc = Yc, w = w)

dpdρθ =
    (ρθ) -> R_d / (1.0 - kappa) * (R_d * ρθ / MSLP) .^ (kappa / (1.0 - kappa))

function tendency!(dY, Y, _, t)
    println(" tendency ", t)
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
    pf = (γ - 1) .* (Ef .- 0.5 .* ρf .* w .* w)


    @. dE = -(∂c(w * (Ef + pf)))

    # w equation
    Ic = Operators.InterpolateF2C()

    ∂f = Operators.GradientC2F()

    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(0.0),
        top = Operators.SetValue(0.0),
    )

    pc = (γ - 1) .* (E .- 0.5 .* ρ .* Ic(w) .^ 2)

    @. dw = B(-w .* If(∂c(w)) - If(1 ./ ρ) .* ∂f(pc) .- grav)

    return dY
end

function jacobian!(J, Y, p, t)

    println(" jacobian ", t)

    J .= 0.0

    #   @info "Jacobian computation!!!!"
    # N cells
    N = div(length(Y) - 1, 3)

    ρ, E, w = Y[1:N], Y[(N + 1):(2N)], Y[(2N + 1):(3N + 1)]

    # construct cell center
    ρh = [ρ[1]; (ρ[1:(N - 1)] + ρ[2:N]) / 2.0; ρ[N]]
    Eh = [E[1]; (E[1:(N - 1)] + E[2:N]) / 2.0; E[N]]
    pf = (γ - 1) .* (Eh .- 0.5 .* ρh .* w .* w)
    hf = Eh + pf


    Δzh = [NaN64; (Δz[1:(N - 1)] + Δz[2:N]) / 2.0; NaN64]

    for i in 1:N
        J[i, i + 2N] = ρh[i] / Δz[i]
        J[i, i + 2N + 1] = -ρh[i + 1] / Δz[i]
    end

    for i in 1:N
        J[i + N, i + 2N] = hf[i] / Δz[i]
        J[i + N, i + 2N + 1] = -hf[i + 1] / Δz[i]
    end

    # 0 for i = 1, N+1
    for i in 2:N
        J[i + 2N, (i - 1)] = -grav / (2 * ρh[i])
        J[i + 2N, (i - 1) + 1] = -grav / (2 * ρh[i])

        J[i + 2N, (i - 1) + N] = (γ - 1) ./ (ρh[i] * Δzh[i])
        J[i + 2N, (i - 1) + 1 + N] = -(γ - 1) ./ (ρh[i] * Δzh[i])
    end


    return J

    # D_ρ = diagm(0=>-ρh/Δz, -1=>ρh/Δz)[1:N, 1:N-1]
    # D_Θ = diagm(0=>-ρθh/Δz, -1=>ρθh/Δz)[1:N, 1:N-1]
    # G_W = (γ - 1) * diagm(0=>Πh./ρh/Δz, 1=>-Πh./ρh/Δz)[1:N-1, 1:N]
    # A_W = diagm(0=>-ones(N-1)./ρh/2, 1=>-ones(N-1)./ρh/2)[1:N-1, 1:N]

    # P = ([zeros(N,N)     D_ρ       zeros(N,N);
    #      A_W*_grav       zeros(N-1,N-1)      G_W
    #      zeros(N,N)     D_Θ              zeros(N,N)])

end

using RecursiveArrayTools

# Y = ArrayPartition(Yc, w)

Δt = 600
ndays = 1
nhours = 0
explicit = false
if explicit
    # Solve the ODE operator
    prob = ODEProblem(tendency!, Y, (0.0, 60 * 60 * (nhours + 24 * ndays)))
    sol = solve(
        prob,
        SSPRK33(),
        dt = Δt,
        saveat = 60 * 60, # save every hour
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )
else
    prob = ODEProblem(
        ODEFunction(
            tendency!,
            jac = jacobian!,
            jac_prototype = zeros(length(Y), length(Y)),
            tgrad = (dT, Y, p, t) -> fill!(dT, 0),
        ),
        Y,
        (0.0, 60 * 60 * 24 * ndays),
    )
    sol = solve(
        prob,
        Rosenbrock23(),
        reltol = 1e-2,
        saveat = 60 * 60, # save every hour
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )
end

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
    hydrostatic_plot(u, Y_init, w_init, title = "Hour $(i)")
end
Plots.mp4(anim, joinpath(path, "hydrostatic_energy.mp4"), fps = 10)

Plots.png(
    hydrostatic_plot(sol[end], Y_init, w_init),
    joinpath(path, "hydrostatic_energy_end.png"),
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
    "output/$(dirname)/hydrostatic_energy_end.png",
    "Hydrostatic Energy End",
)
