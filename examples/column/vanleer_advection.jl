using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33
using ClimaTimeSteppers

import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces


# Advection Equation, with constant advective velocity (so advection form = flux form)
# ∂_t y + w ∂_z y  = 0
# the solution translates to the right at speed w,
# so at time t, the solution is y(z - w * t)

# visualization artifacts
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
dir = "vanleer_advection"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)


function tendency!(yₜ, y, parameters, t)
    (; w, Δt, limiter_method) = parameters
    FT = Spaces.undertype(axes(y.q))
    bcvel = pulse(-π, t, z₀, zₕ, z₁)
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    VanLeerMethod = Operators.LinVanLeerC2F(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
        constraint = limiter_method,
    )

    If = Operators.InterpolateC2F()

    @. yₜ.q = -divf2c(VanLeerMethod(w, y.q, Δt))
end

# Define a pulse wave or square wave

FT = Float64
t₀ = FT(0.0)
t₁ = FT(6)
z₀ = FT(0.0)
zₕ = FT(2π)
z₁ = FT(1.0)
speed = FT(-1.0)
pulse(z, t, z₀, zₕ, z₁) = abs(z - speed * t) ≤ zₕ ? z₁ : z₀

n = 2 .^ 8
elemlist = 2 .^ [3, 4, 5, 6, 7, 8, 9, 10]
Δt = FT(0.1) * (20π / n)
@info "Timestep Δt[s]: $(Δt)"

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(-10π),
    Geometry.ZPoint{FT}(10π);
    boundary_names = (:bottom, :top),
)

stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(FT(7.0)))
plot_string = ["uniform", "stretched"]

for (i, stretch_fn) in enumerate(stretch_fns)
    @info stretch_fn
    limiter_methods = (
        Operators.AlgebraicMean(),
        Operators.PositiveDefinite(),
        Operators.MonotoneHarmonic(),
        Operators.MonotoneLocalExtrema(),
    )
    for (j, limiter_method) in enumerate(limiter_methods)
        mesh = Meshes.IntervalMesh(domain, stretch_fn; nelems = n)
        cent_space = Spaces.CenterFiniteDifferenceSpace(mesh)
        face_space = Spaces.FaceFiniteDifferenceSpace(cent_space)
        z = Fields.coordinate_field(cent_space).z
        O = ones(FT, face_space)

        # Initial condition
        q_init = pulse.(z, 0.0, z₀, zₕ, z₁)
        y = Fields.FieldVector(q = q_init)

        # Unitary, constant advective velocity
        w = Geometry.WVector.(speed .* O)

        # Solve the ODE
        parameters = (; w, Δt, limiter_method)
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = tendency!),
            y,
            (t₀, t₁),
            parameters,
        )
        sol = solve(
            prob,
            ExplicitAlgorithm(SSP33ShuOsher()),
            dt = Δt,
            saveat = Δt,
        )

        q_final = sol.u[end].q

        @info "Extrema with $(limiter_method), i=$i, j=$j: $(extrema(q_final))"
        @show maximum(q_final .- 1)
        @show minimum(q_final .- 0)
        @show abs(maximum(q_final .- 1))
        monotonicity_preserving =
            [Operators.MonotoneHarmonic, Operators.MonotoneLocalExtrema]
        if any(x -> limiter_method isa x, monotonicity_preserving) &&
           stretch_fn == Meshes.Uniform()
            @assert abs(maximum(q_final .- 1)) <= eps(FT)
            @assert abs(minimum(q_final .- 0)) <= eps(FT)
            @assert maximum(q_final) <= FT(1)
        elseif limiter_method != Operators.AlgebraicMean()
            @assert abs(maximum(q_final .- 1)) <= FT(0.05)
            @assert abs(minimum(q_final .- 0)) <= FT(0.05)
            @assert maximum(q_final) <= FT(1)
        end

        q_analytic = pulse.(z, t₁, z₀, zₕ, z₁)

        err = norm(q_final .- q_analytic)
        rel_mass_err = norm((sum(q_final) - sum(q_init)) / sum(q_init))

        if j == 1
            fig = Plots.plot(q_analytic; label = "Exact", color = :red)
        end
        linstyl = [:solid, :dot, :dashdot, :dash]
        clrs = [:orange, :blue, :green, :black]
        fig = plot!(
            q_final;
            label = "$(typeof(limiter_method))"[21:end],
            linestyle = linstyl[j],
            color = clrs[j],
            dpi = 400,
            xlim = (-0.5, 1.1),
            ylim = (-20, 20),
        )
        fig = plot!(legend = :outerbottom, legendcolumns = 2)
        if j == length(limiter_methods)
            Plots.png(
                fig,
                joinpath(
                    path,
                    "LinVanLeerFluxLimiter_" *
                    "$(typeof(limiter_method))"[21:end] *
                    plot_string[i] *
                    ".png",
                ),
            )
        end
        @test err ≤ 0.25
        @test rel_mass_err ≤ 10eps()
    end
end
