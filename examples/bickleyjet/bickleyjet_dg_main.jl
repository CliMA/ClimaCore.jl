push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using LinearAlgebra
using Logging: global_logger
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33
using TerminalLoggers: TerminalLogger
using UnPack

import ClimateMachineCore: Fields, Domains, Topologies, Meshes
import ClimateMachineCore: slab
import ClimateMachineCore.Operators
using ClimateMachineCore.Geometry
import ClimateMachineCore.Geometry: Abstract2DPoint
using ClimateMachineCore.RecursiveOperators
using ClimateMachineCore.RecursiveOperators: rdiv, rmap
using ClimateMachineCore.TicToc


const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
)


function init_state(x, p)
    @unpack x1, x2 = x
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(x2)^(-2)
    Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x1) * cos(p.k * x2)

    ## Vortical velocity fields
    u₁′ = Ψ′ * (p.k * tan(p.k * x2) + x2 / p.l^2)
    u₂′ = -Ψ′ * (p.k * tan(p.k * x1))

    u = Cartesian12Vector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    # set initial tracer
    θ = sin(p.k * x2)

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end


function flux(state, p)
    @unpack ρ, ρu, ρθ = state
    u = ρu ./ ρ
    return (ρ = ρu, ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * I), ρθ = ρθ .* u)
end


function energy(state, p)
    @unpack ρ, ρu = state
    u = ρu ./ ρ
    return ρ * (u.u1^2 + u.u2^2) / 2 + p.g * ρ^2 / 2
end


function total_energy(y, parameters)
    sum(state -> energy(state, parameters), y)
end


# numerical fluxes
wavespeed(y, parameters) = sqrt(parameters.g)


roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))


function roeflux(n, (y⁻, parameters⁻), (y⁺, parameters⁺))
    @tic roeflux

    Favg = rdiv(flux(y⁻, parameters⁻) ⊞ flux(y⁺, parameters⁺), 2)

    λ = sqrt(parameters⁻.g)

    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n

    u⁺ = ρu⁺ / ρ⁺
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
    fluxᵀn_ρu =
        (w1 * (u - c * n) + w2 * (u + c * n) + w3 * u + w4 * (Δu - Δuₙ * n)) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ + w5) * 0.5

    Δf = (ρ = -fluxᵀn_ρ, ρu = -fluxᵀn_ρu, ρθ = -fluxᵀn_ρθ)
    ret = rmap(f -> f' * n, Favg) ⊞ Δf

    @toc roeflux

    return ret
end


function rhs!(dydt, y, (parameters, numflux), t)
    @tic rhs

    # ϕ' K' W J K dydt =  -ϕ' K' I' [DH' WH JH flux.(I K y)]
    #  =>   K dydt = - K inv(K' WJ K) K' I' [DH' WH JH flux.(I K y)]

    # where:
    #  ϕ = test function
    #  K = DSS scatter (i.e. duplicates points at element boundaries)
    #  K y = stored input vector (with duplicated values)
    #  I = interpolation to higher-order mesh
    #  D = derivative operator
    #  H = suffix for higher-order mesh operations
    #  W = Quadrature weights
    #  J = Jacobian determinant of the transformation `ξ` to `x`
    #
    Nh = Topologies.nlocalelems(y)

    F = flux.(y, Ref(parameters))
    dydt .= Operators.slab_weak_divergence(F)

    Operators.add_numerical_flux_internal!(numflux, dydt, y, parameters)

    # 6. Solve for final result
    mesh = Fields.mesh(dydt)
    dydt_data = Fields.field_values(dydt)
    dydt_data .= rdiv.(dydt_data, mesh.local_geometry.WJ)

    M = Meshes.Quadratures.cutoff_filter_matrix(
        Float64,
        mesh.quadrature_style,
        3,
    )
    Operators.tensor_product!(dydt_data, M)

    @toc rhs
    return dydt
end


function main(numflux)
    global_logger(TerminalLogger())

    @tic domain
    domain = Domains.RectangleDomain(
        x1min = -2π,
        x1max = 2π,
        x2min = -2π,
        x2max = 2π,
        x1periodic = true,
        x2periodic = true,
    )
    @toc domain

    n1, n2 = 16, 16
    Nq = 4
    Nqh = 7
    @tic discretization
    discretization = Domains.EquispacedRectangleDiscretization(domain, n1, n2)
    @toc discretization
    @tic topology
    grid_topology = Topologies.GridTopology(discretization)
    @toc topology
    @tic quad
    quad = Meshes.Quadratures.GLL{Nq}()
    @toc quad
    @tic mesh
    mesh = Meshes.Mesh2D(grid_topology, quad)
    @toc mesh

    @tic iquad
    Iquad = Meshes.Quadratures.GLL{Nqh}()
    @toc iquad
    @tic imesh
    Imesh = Meshes.Mesh2D(grid_topology, Iquad)
    @toc imesh

    @tic init
    y0 = init_state.(Fields.coordinate_field(mesh), Ref(parameters))
    @toc init

    @tic field
    dydt = Fields.Field(similar(Fields.field_values(y0)), mesh)
    @toc field
    @tic rhs_init
    rhs!(dydt, y0, (parameters, numflux), 0.0);
    @toc rhs_init

    # Solve the ODE operator
    @tic odeproblem
    prob = ODEProblem(rhs!, y0, (0.0, 200.0), (parameters, numflux))
    @toc odeproblem
    @tic solve
    sol = solve(
        prob,
        SSPRK33(),
        dt = 0.02,
        #saveat = 1.0,
        #progress = true,
        #progress_message = (dt, u, p, t) -> t,
    )
    @toc solve

    return sol
end

function select_numflux(args::Vector{String})
    numflux_name = get(args, 1, "rusanov")

    numflux = if numflux_name == "central"
        Operators.CentralNumericalFlux(flux)
    elseif numflux_name == "rusanov"
        Operators.RusanovNumericalFlux(flux, wavespeed)
    elseif numflux_name == "roe"
        roeflux
    end

    return (numflux_name, numflux)
end

tictoc()
numflux_name, numflux = select_numflux(ARGS)
sol = @time main(numflux)

#=
using Plots
ENV["GKSwstype"] = "nul"

anim = @animate for u in sol.u
    heatmap(u.ρθ, clim = (-2, 2), color = :balance)
end
mp4(anim, joinpath(@__DIR__, "bickleyjet_dg_$numflux_name.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]
png(plot(Es), joinpath(@__DIR__, "energy_dg_$numflux_name.png"))
=#

# # figpath = joinpath(figure_save_directory, "posterior_$(param)_T_$(T)_w_$(ω_true).png")
# linkfig(figpath)
function linkfig(figpath)
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url =
            "artifact://" * join(split(figpath, '/')[(end - 3):end], '/')
        alt = split(splitdir(figpath)[2], '.')[1]
        @info "Linking Figure: $artifact_url"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end
