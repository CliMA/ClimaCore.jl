using Test
using LinearAlgebra
using OrdinaryDiffEq: ODEProblem, solve
using ClimaTimeSteppers

#=
Following
https://journals.ametsoc.org/view/journals/mwre/122/7/1520-0493_1994_122_1575_acotvl_2_0_co_2.xml?tab_body=pdf

Section 1 notes:
 - Can be applied to scalars only.
 - Upstream nature means that they are generally diffusive.
 - Prevents undershoots/overshoots/oscillations
 - Positive-definiteness maintained
 - "All filters previously previously applied to the moisture field except vertical turbulent fluxes, which include the flux of the water vapor (due to evaporation" from the earth's surface, are removed."
 - "...for moisture transport, one can simply set the lower bound to zero globally and he upper to be the locally determined saturation density or other physically based value"
 - in this paper we focus on eq 5

Section 2 notes:
 - Transport in multi-dimensions is achieved by alternating directional splitting.
 - Define ϕ_{i+1/2} as the mean (in space) density inside the cell bounded by x_i and `x_{i+1}`.
 - It is assumed that the subgrid distribution of ϕ is linear.

Implemented equation:

ϕᵢ = Φᵢ/Δt
fluxᵢ = if Uᵢ ≥ 0
    Uᵢ (Φᵢ₋₁₂ + ΔΦᵢ₋₁₂*(1-Cᵢ₋))
else
    Uᵢ (Φᵢ₋₁₂ + ΔΦᵢ₋₁₂*(1-Cᵢ₊))
end
where
Cᵢ₋ = Uᵢ*Δt/Δxᵢ₋₁₂
Cᵢ₊ = Uᵢ*Δt/Δxᵢ₊₁₂

δΦᵢ = Φᵢ₊₁₂-Φᵢ₋₁₂
ΔΦᵢ₊₁₂_ave = (δΦᵢ+δΦᵢ₊₁)/2

Φᵢ₊₁₂_min = min(Φᵢ₋₁₂,Φᵢ₊₁₂,Φᵢ₊₃₂)
Φᵢ₊₁₂_max = max(Φᵢ₋₁₂,Φᵢ₊₁₂,Φᵢ₊₃₂)
ΔΦᵢ₊₁₂_mono =
    sign(ΔΦᵢ₊₁₂_ave) *
    min(
        ΔΦᵢ₊₁₂_ave,
        2*(Φᵢ₊₁₂-Φᵢ₊₁₂_min),
        2*(Φᵢ₊₁₂_max - Φᵢ₊₁₂),
    )


Φᵢ₊₃₂
Φᵢ₋₁₂
ΔΦᵢ₋₂
Φᵢ₋₂
Φᵢ₋₁
Φᵢ
Φᵢ₊₁
Φᵢ₊₂

=#

# Advection Equation, with constant advective velocity (so advection form = flux form)
# ∂_t y + w ∂_z y  = 0
# the solution translates to the right at speed w,
# so at time t, the solution is y(z - w * t)

# visualization artifacts
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
dir = "van_leer_advection"
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

function tendency!(yₜ, y, parameters, t)
    (; w, Δt) = parameters
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
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
    VanLeer = Operators.VanLeer(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
    )
    @. yₜ.q =
        -divf2c(
            upwind1(w, y.q) + VanLeer(
                upwind3(w, y.q) - upwind1(w, y.q),
                y.q / Δt,
                y.q / Δt - divf2c(upwind1(w, y.q)),
            ),
        )
end

# Define a pulse wave or square wave
pulse(z, t, z₀, zₕ, z₁) = abs(z - speed * t) ≤ zₕ ? z₁ : z₀

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
plot_string = ["uniform", "stretched"]

for (i, stretch_fn) in enumerate(stretch_fns)
    mesh = Meshes.IntervalMesh(domain, stretch_fn; nelems = n)
    cent_space = Spaces.CenterFiniteDifferenceSpace(mesh)
    face_space = Spaces.FaceFiniteDifferenceSpace(cent_space)
    z = Fields.coordinate_field(cent_space).z

    # Initial condition
    q_init = pulse.(z, 0.0, z₀, zₕ, z₁)
    y = Fields.FieldVector(q = q_init)

    # Unitary, constant advective velocity
    w = Geometry.WVector.(speed .* ones(FT, face_space))

    # Solve the ODE
    parameters = (; w, Δt)
    prob = ODEProblem(
        ClimaODEFunction(; T_exp! = tendency!),
        y,
        (t₀, t₁),
        parameters,
    )
    sol = solve(prob, ExplicitAlgorithm(SSP33ShuOsher()), dt = Δt, saveat = Δt)

    q_final = sol.u[end].q
    q_analytic = pulse.(z, t₁, z₀, zₕ, z₁)
    err = norm(q_final .- q_analytic)
    rel_mass_err = norm((sum(q_final) - sum(q_init)) / sum(q_init))

    @test err ≤ 0.11
    @test rel_mass_err ≤ 10eps()

    plot(q_final)
    Plots.png(
        Plots.plot!(q_analytic, title = "VanLeer"),
        joinpath(
            path,
            "exact_and_computed_advected_square_wave_VanLeer_" *
            plot_string[i] *
            ".png",
        ),
    )
end
