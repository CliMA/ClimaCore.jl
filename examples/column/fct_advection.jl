import ClimaComms
ClimaComms.@import_required_backends
using Test
using LinearAlgebra
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

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
dir = "fct_advection"
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

function f!(dydt, y, parameters, t)

    (; w, C, corrected_antidiff_flux) = parameters
    FT = Spaces.undertype(axes(y))

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

    @. corrected_antidiff_flux =
        divf2c(C * (third_order_fluxᶠ(w, y) - first_order_fluxᶠ(w, y)))
    @. dydt = -(divf2c(first_order_fluxᶠ(w, y)) + corrected_antidiff_flux)

    return dydt
end

# Define a pulse wave or square wave
pulse(z, t, z₀, zₕ, z₁) = abs(z.z - speed * t) ≤ zₕ ? z₁ : z₀

FT = Float64
t₀ = FT(0)
Δt = 0.0001
t₁ = 100Δt
z₀ = FT(0)
zₕ = FT(1)
z₁ = FT(1)
speed = FT(1.0)
C = FT(1.0) # flux-correction coefficient: ∈ [0,1] with
#                              0 = first-order upwinding
#                              1 = third-order upwinding
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
device = ClimaComms.device()
for (i, stretch_fn) in enumerate(stretch_fns)

    mesh = Meshes.IntervalMesh(domain, stretch_fn; nelems = n)
    cs = Spaces.CenterFiniteDifferenceSpace(device, mesh)
    fs = Spaces.FaceFiniteDifferenceSpace(cs)
    zc = Fields.coordinate_field(cs)
    O = ones(FT, fs)

    # Unitary, constant advective velocity
    w = Geometry.WVector.(speed .* O)

    # Initial condition
    y0 = pulse.(zc, 0.0, z₀, zₕ, z₁)

    # Set up fields needed for time-stepping
    dydt = copy(y0)
    corrected_antidiff_flux = similar(y0)
    parameters = (; w, C, corrected_antidiff_flux)
    # Call the RHS function
    f!(dydt, y0, parameters, t₀)

    prob = ODEProblem(f!, y0, (t₀, t₁), parameters)
    sol = solve(
        prob,
        SSPRK33(),
        dt = Δt,
        saveat = Δt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )
    computed_result = sol.u[end]
    analytical_result = pulse.(zc, t₁, z₀, zₕ, z₁)
    err[i] = norm(computed_result .- analytical_result)
    initial_mass[i] = sum(sol.u[1])
    mass[i] = sum(sol.u[end])
    rel_mass_err[i] = norm((mass[i] - initial_mass[i]) / initial_mass[i])

    @test err[i] ≤ 0.11
    @test rel_mass_err[i] ≤ 3eps()

    plot(sol.u[end])
    Plots.png(
        Plots.plot!(analytical_result, title = "FCT (C=0.5)"),
        joinpath(
            path,
            "exact_and_computed_advected_square_wave_C_05_" *
            plot_string[i] *
            ".png",
        ),
    )
end
