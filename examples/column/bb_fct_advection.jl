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

    (; w, A, y_td) = parameters
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
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )

    @. y_td = beta * dydt - alpha * divf2c(first_order_fluxᶠ(w, y))
    @. dydt =
        y_td -
        alpha * divf2c(
            FCTBB(
                third_order_fluxᶠ(w, y) - first_order_fluxᶠ(w, y),
                y_td * alpha,
            ),
        )

    return dydt
end

FT = Float64
t₀ = FT(0.0)
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
mesh = Meshes.IntervalMesh(domain, nelems = n)
cs = Spaces.CenterFiniteDifferenceSpace(mesh)
fs = Spaces.FaceFiniteDifferenceSpace(cs)
zc = Fields.coordinate_field(cs)

# Unitary, constant advective velocity
w = Geometry.WVector.(speed .* ones(FT, fs))

# Define a pulse wave or square wave
pulse(z, t, z₀, zₕ, z₁) = abs(z.z - speed * t) ≤ zₕ ? z₁ : z₀

# Initial condition
y0 = pulse.(zc, 0.0, z₀, zₕ, z₁)

# ClimaTimeSteppers need a FieldVector
y0 = Fields.FieldVector(y = y0)

# Set up parameters needed for time-stepping
Δt = 0.0001
dydt = copy(y0)
A = similar(dydt.y)
y_td = similar(dydt.y)

parameters = (; w, A, y_td)
# Call the RHS function
f!(dydt, y0, parameters, 0.0, Δt, 1)
t₁ = 100Δt
prob = ODEProblem(IncrementingODEFunction(f!), copy(y0), (t₀, t₁), parameters)
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
err = norm(computed_result .- analytical_result)
initial_mass = sum(sol.u[1].y)
mass = sum(sol.u[end].y)
rel_mass_err = norm((mass - initial_mass) / initial_mass)

@test err ≤ 0.018
@test rel_mass_err ≤ 13eps()

plot(sol.u[end].y)
Plots.png(
    Plots.plot!(analytical_result, title = "Boris and Book FCT"),
    joinpath(path, "exact_and_computed_advected_square_wave_BBFCT.png"),
)
