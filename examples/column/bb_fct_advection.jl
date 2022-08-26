using Test
using LinearAlgebra
using OrdinaryDiffEq: ODEProblem, solve
using DiffEqBase
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
dir = "bb_fct_advection"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

function f!(dydt, y, parameters, t, alpha, beta)

    (; w, y_td) = parameters
    y = y.y
    dydt = dydt.y

    first_order_fluxᶠ = Operators.UpwindBiasedProductC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    third_order_fluxᶠ = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0.0))),
        top = Operators.SetValue(Geometry.WVector(FT(0.0))),
    )
    FCTBB = Operators.FCTBorisBook(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
    )

    @. y_td = beta * dydt - alpha * divf2c(first_order_fluxᶠ(w, y))
    @. dydt =
        y_td -
        alpha * divf2c(
            FCTBB(
                third_order_fluxᶠ(w, y) - first_order_fluxᶠ(w, y),
                y_td / alpha,
            ),
        )

    return dydt
end

# Define a pulse wave or square wave
pulse(z, t, z₀, zₕ, z₁) = abs(z.z - speed * t) ≤ zₕ ? z₁ : z₀

FT = Float64
t₀ = FT(0.0)
Δt = 0.0001
t₁ = 100Δt
z₀ = FT(0.0)
zₕ = FT(1.0)
z₁ = FT(1.0)
speed = FT(1.0)

n = 2 .^ 6

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(-π),
    Geometry.ZPoint{FT}(π);
    boundary_names = (:bottom, :top),
)

stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(FT(7.0)))
err = zeros(FT, length(stretch_fns))
initial_mass = zeros(FT, length(stretch_fns))
mass = zeros(FT, length(stretch_fns))
rel_mass_err = zeros(FT, length(stretch_fns))
plot_string = ["uniform", "stretched"]

for (i, stretch_fn) in enumerate(stretch_fns)

    mesh = Meshes.IntervalMesh(domain, stretch_fn; nelems = n)
    cs = Spaces.CenterFiniteDifferenceSpace(mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)
    zc = Fields.coordinate_field(cs)

    # Unitary, constant advective velocity
    w = Geometry.WVector.(speed .* ones(FT, fs))

    # Initial condition
    y0 = pulse.(zc, 0.0, z₀, zₕ, z₁)

    # ClimaTimeSteppers need a FieldVector
    y0 = Fields.FieldVector(y = y0)

    # Set up fields needed for time-stepping
    dydt = copy(y0)
    y_td = similar(dydt.y)

    parameters = (; w, y_td)
    # Call the RHS function
    f!(dydt, y0, parameters, 0.0, Δt, 1)

    prob =
        ODEProblem(IncrementingODEFunction(f!), copy(y0), (t₀, t₁), parameters)
    sol = solve(
        prob,
        SSPRK33ShuOsher(),
        dt = Δt,
        saveat = Δt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )
    computed_result = sol.u[end].y
    analytical_result = pulse.(zc, t₁, z₀, zₕ, z₁)
    err[i] = norm(computed_result .- analytical_result)
    initial_mass[i] = sum(sol.u[1].y)
    mass[i] = sum(sol.u[end].y)
    rel_mass_err[i] = norm((mass - initial_mass) / initial_mass)

    @test err[i] ≤ 0.11
    @test rel_mass_err[i] ≤ 5eps()

    plot(sol.u[end].y)
    Plots.png(
        Plots.plot!(analytical_result, title = "Boris and Book FCT"),
        joinpath(
            path,
            "exact_and_computed_advected_square_wave_BBFCT_" *
            plot_string[i] *
            ".png",
        ),
    )
end
