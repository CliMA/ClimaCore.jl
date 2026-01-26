#=
julia --project=.buildkite
using Revise; include(joinpath("test", "Limiters", "vertical_mass_borrowing_limiter_advection.jl"))
=#
using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33
using ClimaCore.CommonGrids
using ClimaCore.Grids
using ClimaTimeSteppers
import ClimaCore

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
import ClimaCorePlots
import Plots
Plots.GRBackend()
device_name = ClimaComms.device() isa ClimaComms.CUDADevice ? "GPU" : "CPU"
dir = "vert_mass_borrow_advection"
path = joinpath(@__DIR__, "output", dir, device_name)
mkpath(path)

function lim!(y, parameters, t, y_ref)
    (; w, Δt, limiter) = parameters
    Limiters.apply_limiter!(y.q, y.ρ, limiter)
    return nothing
end

function perturb_field!(f::Fields.Field; perturb_radius)
    device = ClimaComms.device(f)
    ArrayType = ClimaComms.array_type(device)
    rand_data = ArrayType(rand(size(parent(f))...)) # [0-1]
    rand_data = rand_data .- sum(rand_data) / length(rand_data) # make centered about 0 ([-0.5:0.5])
    rand_data = (rand_data ./ maximum(rand_data)) .* perturb_radius # rand_data now in [-perturb_radius:perturb_radius]
    parent(f) .= parent(f) .+ rand_data # f in [f ± perturb_radius]
    return nothing
end

function tendency!(yₜ, y, parameters, t)
    (; w, Δt, limiter) = parameters
    FT = Spaces.undertype(axes(y.q))
    bcvel = pulse(-π, t, z₀, zₕ, z₁, speed)
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
    @. yₜ.q = -divf2c(upwind1(w, y.q) * If(y.q))
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
pulse(z, t, z₀, zₕ, z₁, speed) = abs(z - speed * t) ≤ zₕ ? z₁ : z₀

stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(FT(7.0)))
plot_string = ["uniform", "stretched"]

@testset "VerticalMassBorrowingLimiter on advection" begin
    for (i, stretch) in enumerate(stretch_fns)
        i = 1
        stretch = Meshes.Uniform()

        z_elem = 2^6
        z_min = -π
        z_max = π
        device = ClimaComms.device()

        # use_column = true
        use_column = false
        if use_column
            grid = ColumnGrid(; z_elem, z_min, z_max, stretch, device)
            cspace = Spaces.FiniteDifferenceSpace(grid, Grids.CellCenter())
            fspace = Spaces.FaceFiniteDifferenceSpace(cspace)
        else
            grid = ExtrudedCubedSphereGrid(;
                z_elem,
                z_min,
                z_max,
                stretch,
                radius = 10,
                h_elem = 10,
                n_quad_points = 4,
                device,
            )
            cspace =
                Spaces.ExtrudedFiniteDifferenceSpace(grid, Grids.CellCenter())
            fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
        end

        z = Fields.coordinate_field(cspace).z
        O = ones(FT, fspace)

        # Initial condition
        q_init = pulse.(z, 0.0, z₀, zₕ, z₁, speed)
        q = q_init
        coords = Fields.coordinate_field(q)
        ρ = map(coord -> 1.0, coords)
        perturb_field!(ρ; perturb_radius = 0.1)
        y = Fields.FieldVector(; q, ρ)
        limiter = Limiters.VerticalMassBorrowingLimiter((0.0,))

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
            save_everystep = true,
        )

        q_init = sol.u[1].q
        q_final = sol.u[end].q
        q_analytic = pulse.(z, t₁, z₀, zₕ, z₁, speed)
        err = norm(q_final .- q_analytic)
        rel_mass_err = norm((sum(q_final) - sum(q_init)) / sum(q_init))


        if use_column
            p = Plots.plot()
            Plots.plot!(q_init |> ClimaCore.to_cpu, label = "init")
            Plots.plot!(q_final |> ClimaCore.to_cpu, label = "computed")
            Plots.plot!(q_analytic |> ClimaCore.to_cpu, label = "analytic")
            Plots.plot!(; legend = :topright)
            Plots.plot!(; xlabel = "q", title = "VerticalMassBorrowingLimiter")
            f = joinpath(
                path,
                "VerticalMassBorrowingLimiter_comparison_$(plot_string[i]).png",
            )
            Plots.png(p, f)
        else
            colidx = Fields.ColumnIndex((1, 1), 1)
            p = Plots.plot()
            Plots.plot!(
                vec(parent(q_init[colidx] |> ClimaCore.to_cpu)),
                vec(parent(z[colidx] |> ClimaCore.to_cpu)),
                label = "init",
            )
            Plots.plot!(
                vec(parent(q_final[colidx] |> ClimaCore.to_cpu)),
                vec(parent(z[colidx] |> ClimaCore.to_cpu)),
                label = "computed",
            )
            Plots.plot!(
                vec(parent(q_analytic[colidx] |> ClimaCore.to_cpu)),
                vec(parent(z[colidx] |> ClimaCore.to_cpu)),
                label = "analytic",
            )
            Plots.plot!(; legend = :topright)
            Plots.plot!(;
                xlabel = "q",
                ylabel = "z",
                title = "VerticalMassBorrowingLimiter",
            )
            f = joinpath(
                path,
                "VerticalMassBorrowingLimiter_comparison_$(plot_string[i]).png",
            )
            Plots.png(p, f)
        end
        @test err ≤ 0.25
        @test rel_mass_err ≤ 10eps()
        @test minimum(q_final) ≥ 0
    end
end
