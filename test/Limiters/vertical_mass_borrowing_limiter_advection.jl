#=
julia --project=.buildkite
using Revise; include(joinpath("test", "Limiters", "vertical_mass_borrowing_limiter_advection.jl"))
=#
using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33
using ClimaTimeSteppers

import ClimaCore:
    Fields,
    Domains,
    Limiters,
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
dir = "vert_mass_borrow_advection"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

function lim!(y, parameters, t, y_ref)
    (; w, Δt, limiter) = parameters
    Limiters.apply_limiter!(y.q, limiter)
    return nothing
end

function tendency!(yₜ, y, parameters, t)
    (; w, Δt, limiter) = parameters
    FT = Spaces.undertype(axes(y.q))
    bcvel = pulse(-π, t, z₀, zₕ, z₁)
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(bcvel))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    upwind1 = Operators.UpwindBiasedProductC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    upwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    If = Operators.InterpolateC2F()
    @. yₜ.q =
    # -divf2c(w * If(y.q))
        -divf2c(upwind1(w, y.q) * If(y.q))
    # -divf2c(upwind3(w, y.q) * If(y.q))
    return nothing
end

# Define a pulse wave or square wave

FT = Float64
t₀ = FT(0.0)
Δt = 0.0001 * 25
t₁ = 200Δt
z₀ = FT(0.0)
zₕ = FT(1.0)
z₁ = FT(1.0)
speed = FT(-1.0)
pulse(z, t, z₀, zₕ, z₁) = abs(z - speed * t) ≤ zₕ ? z₁ : z₀

n = 2 .^ 6

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(-π),
    Geometry.ZPoint{FT}(π);
    boundary_names = (:bottom, :top),
)

# stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(FT(7.0)))
stretch_fns = (Meshes.Uniform(),)
plot_string = ["uniform", "stretched"]

@testset "VerticalMassBorrowingLimiter on advection" begin
    for (i, stretch_fn) in enumerate(stretch_fns)
        mesh = Meshes.IntervalMesh(domain, stretch_fn; nelems = n)
        cent_space = Spaces.CenterFiniteDifferenceSpace(mesh)
        face_space = Spaces.FaceFiniteDifferenceSpace(cent_space)
        z = Fields.coordinate_field(cent_space).z
        O = ones(FT, face_space)

        # Initial condition
        q_init = pulse.(z, 0.0, z₀, zₕ, z₁)
        q = q_init
        y = Fields.FieldVector(; q)
        limiter = Limiters.VerticalMassBorrowingLimiter(q, (0.0,))

        # Unitary, constant advective velocity
        w = Geometry.WVector.(speed .* O)

        # Solve the ODE
        parameters = (; w, Δt, limiter)
        prob = ODEProblem(
            ClimaODEFunction(; lim!, T_lim! = tendency!),
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

        q_init = sol.u[1].q
        q_final = sol.u[end].q
        q_analytic = pulse.(z, t₁, z₀, zₕ, z₁)
        err = norm(q_final .- q_analytic)
        rel_mass_err = norm((sum(q_final) - sum(q_init)) / sum(q_init))


        p = plot()
        Plots.plot!(q_init, label = "init")
        Plots.plot!(q_final, label = "computed")
        Plots.plot!(q_analytic, label = "analytic")
        Plots.plot!(; legend = :topright)
        Plots.plot!(; xlabel = "q", title = "VerticalMassBorrowingLimiter")
        f = joinpath(
            path,
            "VerticalMassBorrowingLimiter_comparison_$(plot_string[i]).png",
        )
        Plots.png(p, f)
        @test err ≤ 0.25
        @test rel_mass_err ≤ 10eps()
        @test minimum(q_final) ≥ 0
    end
end
