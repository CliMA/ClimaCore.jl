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
Yc = decaying_temperature_profile.(zc)
w = Geometry.Cartesian3Vector.(zeros(FT, fspace))
zf = parent(Fields.coordinate_field(fspace))
Δz = zf[2:end] - zf[1:(end - 1)]
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
            -(If(Yc.ρθ / Yc.ρ) * ∂f(Π(Yc.ρθ))) - ∂f(Φ(zc)),
        ),
    )
    return dY
end

dY = tendency!(similar(Y), Y, nothing, 0.0)

function jacobian!(J, Y, p, t)
    # copyto!(J, LinearAlgebra.I)
    # @info length(Y)
    # @info Y[1]

    J .= 0.0

    @info "Jacobian computation!!!!"
    # N cells
    N = div(length(Y) - 1, 3)

    ρ, ρθ, w = Y[1:N], Y[(N + 1):(2N)], Y[(2N + 1):(3N + 1)]

    # construct cell center
    ρh = [ρ[1]; (ρ[1:(N - 1)] + ρ[2:N]) / 2.0; ρ[N]]
    ρθh = [ρθ[1]; (ρθ[1:(N - 1)] + ρθ[2:N]) / 2.0; ρθ[N]]


    Πc = Π.(ρθ)
    Πh = [NaN64; (Πc[1:(N - 1)] + Πc[2:N]) / 2.0; NaN64]
    Δzh = [NaN64; (Δz[1:(N - 1)] + Δz[2:N]) / 2.0; NaN64]

    # A_W = diagm(0=>-ones(N-1)./ρh/2, 1=>-ones(N-1)./ρh/2)[1:N-1, 1:N]

    # J = ([zeros(N,N)      zeros(N,N)      D_ρ (dρ/dw);
    #       zeros(N,N)      zeros(N,N)      D_Θ (dρΘ/dw)
    #       A_W*_grav (dw/dρ)  G_W (dw/dρΘ)  zeros(N+1,N+1)])

    #
    # D_ρ = diagm(0=>-ρh/Δz, -1=>ρh/Δz)[1:N, 1:N-1]
    # (dρₜ/dw)
    # Bidiagonal
    for i in 1:N
        J[i, i + 2N] = ρh[i] / Δz[i]
        J[i, i + 2N + 1] = -ρh[i + 1] / Δz[i]
    end

    # D_Θ = diagm(0=>-ρθh/Δz, -1=>ρθh/Δz)[1:N, 1:N-1]
    # (dρΘₜ/dw)
    # Bidiagonal
    for i in 1:N
        J[i + N, i + 2N] = ρθh[i] / Δz[i]
        J[i + N, i + 2N + 1] = -ρθh[i + 1] / Δz[i]
    end

    # (dwₜ/dρ) = A_W*_grav
    # A_W = diagm(0=>-ones(N-1)./ρh/2, 1=>-ones(N-1)./ρh/2)[1:N-1, 1:N]
    # Bidiagonal
    for i in 2:N
        J[i + 2N, (i - 1)] = -grav / (2 * ρh[i])
        J[i + 2N, (i - 1) + 1] = -grav / (2 * ρh[i])
    end

    # G_W = (γ - 1) * diagm(0=>Πh./ρh/Δz, 1=>-Πh./ρh/Δz)[1:N-1, 1:N]
    # (dwₜ/dρΘ) = G_W
    # Bidiagonal
    for i in 2:N
        J[i + 2N, (i - 1) + N] = (γ - 1) * Πh[i] ./ (ρh[i] * Δzh[i])
        J[i + 2N, (i - 1) + 1 + N] = -(γ - 1) * Πh[i] ./ (ρh[i] * Δzh[i])
    end


    return J
end

struct CustomFactorization
    γ
    Jρ_w::Bidiagonal
    Jρθ_w::Bidiagonal
    Jw_ρ::Bidiagonal
    Jw_ρθ::Bidiagonal
end

function Wfact!(iW,u,p,gamma,t)
    # TODO: this should return some sort of useful object
    # e.g. "CustomFactorization"
    jacobian!(iW, u, p, t)
    iW .= I - gamma .* iW

end
function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        # TODO: A "should" be the factorization object returned from Wfact!

        #     _A = RecursiveFactorization.lu(A)
        #   ldiv!(x,_A,b)
        #   @info "norm: ", norm(x), norm(A), norm(b)
        x = copy(b)
        N = div(length(x) - 1, 3)

        #=
        # A = I - gamma J

        # J = ([zeros(N,N)      zeros(N,N)      D_ρ (dρ/dw);
        #       zeros(N,N)      zeros(N,N)      D_Θ (dρΘ/dw)
        #       A_W*_grav (dw/dρ)  G_W (dw/dρΘ)  zeros(N+1,N+1)])

        # A = ([I               0           -γ*(dρₜ/dw);
        #       0               I           -γ*(dρΘₜ/dw)
        #       -γ*(dwₜ/dρ)   -γ*(dwₜ/dρΘ)       I           ])


        # A = ([I               0           A13;
        #       0               I           A23
        #       A31             A32         I           ])
        # b = ([b1;
        #       b2
        #       b3])

        # solve for x
        # A* [x1; x2; x3] = [b1; b2; b3]

        # x1 = b1 - A13 * x3  (1)
        # x2 = b2 - A23 * x3  (2)
        # A31 x1 + A32 x2 + x3 = b3 (3)
        # bring x1 and x2 into eq(3)
        # A31 ( b1 - A13 * x3) + A32 (b2 - A23 * x3) + x3 = b3
        # S:= I - A31*A13 - A32*A23 "Schur complement" =>  Tridiagonal

        # 1) Form tridiagonal matrix
        S = Tridiagonal(I)
        # S = S + γ^2 * Jw_ρ * Jρ_w
        mul!(S, Jw_ρ, Jρ_w, γ^2, 1)
        # S = S + γ^2 * Jw_ρθ * Jρθ_w
        mul!(S, Jw_ρθ, Jρθ_w, γ^2, 1)
        # S * x3 = b3 - A31 *b1 - A32 * b2

        # 2) form RHS
        # x3 = S\(b3 - A31 *b1 - A32 * b2)
        # x3 = b3 + γ * Jw_ρ *b1 + γ * Jw_ρθ * b2
        x3 .= b3
        # mul!(C, A, B, α, β) =>   C = α * A * B + β * C
        mul!(x3, Jw_ρ, b1, γ, 1)
        mul!(x3, Jw_ρθ, b2, γ, 1)

        # 3) solve for x3
        # TODO: LinearAlgebra will compute the LU factorization, then solve
        # Thomas' algorithm can do this in one step:
        # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        # can also reuse x3 storage for t3, and solve in-place
        ldiv!(lu!(S), x3)


        # 4) compute x1
        # use (1) & (2) to get x1, x2
        # x1 .= b1 + γ * Jρ_w * x3
        x1 .= b1
        mul!(x1, Jρ_w, x3, γ, 1)

        # 5) compute x2
        # x2 .= b2 + γ * Jρθ_w * x3
        x2 .= b2
        mul!(x2, Jρθ_w, x3, γ, 1)

        =#


        # J = (dwₜ/dw) = I
        J = A[(2N + 1):(3N + 1), (2N + 1):(3N + 1)]

        # A[]
        # J = I +  (-γ*(dρₜ/dw))   γ * (dwₜ/dρ)
        J +=
            -A[(2N + 1):(3N + 1), 1:N] *
            (Diagonal(A[1:N, 1:N]) \ A[1:N, (2N + 1):(3N + 1)])
        J +=
            -A[(2N + 1):(3N + 1), (N + 1):(2N)] * (
                Diagonal(A[(N + 1):(2N), (N + 1):(2N)]) \
                A[(N + 1):(2N), (2N + 1):(3N + 1)]
            )

        x[(2N + 1):(3N + 1)] +=
            -A[(2N + 1):(3N + 1), 1:N] * (Diagonal(A[1:N, 1:N]) \ b[1:N])
        x[(2N + 1):(3N + 1)] +=
            -A[(2N + 1):(3N + 1), (N + 1):(2N)] *
            (Diagonal(A[(N + 1):(2N), (N + 1):(2N)]) \ b[(N + 1):(2N)])

        x[(2N + 1):(3N + 1)] .= Tridiagonal(J) \ x[(2N + 1):(3N + 1)]
        x[1:N] .=
            Diagonal(A[1:N, 1:N]) \
            (b[1:N] - A[1:N, (2N + 1):(3N + 1)] * x[(2N + 1):(3N + 1)])
        x[(N + 1):(2N)] .=
            Diagonal(A[(N + 1):(2N), (N + 1):(2N)]) \ (
                b[(N + 1):(2N)] -
                A[(N + 1):(2N), (2N + 1):(3N + 1)] * x[(2N + 1):(3N + 1)]
            )

    end
end

Δt = 600.0
ndays = 1.0

# Solve the ODE operator
prob = ODEProblem(
    ODEFunction(
        tendency!,
        Wfact = Wfact!,
        jac_prototype = zeros(length(Y), length(Y)),
        tgrad = (dT, Y, p, t) -> fill!(dT, 0),
    ),
    Y,
    (0.0, 60 * 60 * 24 * ndays),
)
# 60 * 60 * 24 * ndays
sol = solve(
    prob,
    # ImplicitEuler(),
    Rosenbrock23(linsolve = linsolve!),
    dt = Δt,
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

anim = Plots.@animate for (i, u) in enumerate(sol.u)
    hydrostatic_plot(u, Y_init, w_init, title = "Hour $(i)")
end
Plots.mp4(anim, joinpath(path, "hydrostatic.mp4"), fps = 10)

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
