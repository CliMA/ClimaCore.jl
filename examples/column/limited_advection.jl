# Vertical advection with a slope-limited, upwind-biased flux. A tracer with
# sharp features is advected on a column; the limiter should transport it without
# introducing new extrema, which the run asserts by checking that the tracer stays
# within its initial bounds. `limited_flux_operator` isolates the choice of
# limiter, so a different constraint can be swapped in.
using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaTimeSteppers as CTS

import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces


# Constant-velocity advection of a square pulse, used to compare limited face
# reconstructions:
#
#     ∂_t q + w ∂_z q = 0,
#
# so at time t the exact solution is the initial pulse translated by w * t. A
# limiter is judged on whether it keeps q within the bounds of the initial data
# (here [0, 1]) while staying accurate, on both uniform and stretched meshes.
#
# The scheme under test is built in one place, `limited_flux_operator`, and swept
# over the available constraints below.

# visualization artifacts
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
dir = "limited_advection"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)


"""
    limited_flux_operator(constraint)

Face reconstruction of the advected quantity, limited by `constraint`.

This is the only place that names a concrete upwind-biased operator. When those
operators are consolidated behind a single type that dispatches on a constraint
type parameter, this function is the one thing that needs to change.
"""
limited_flux_operator(constraint) = Operators.LinVanLeerC2F(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
    constraint = constraint,
)

function tendency!(yₜ, y, parameters, t)
    (; w, Δt, constraint) = parameters
    FT = Spaces.undertype(axes(y.q))
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    limited_flux = limited_flux_operator(constraint)
    @. yₜ.q = -divf2c(limited_flux(w, y.q, Δt))
end

# Define a pulse wave or square wave

const FT = Float64
const t₀ = FT(0.0)
const t₁ = FT(6)
const z₀ = FT(0.0)
const zₕ = FT(2π)
const z₁ = FT(1.0)
const speed = FT(-1.0)
pulse(z, t, z₀, zₕ, z₁) = abs(z - speed * t) ≤ zₕ ? z₁ : z₀

n = 2^8
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
    constraints = (
        Operators.AlgebraicMean(),
        Operators.PositiveDefinite(),
        Operators.MonotoneHarmonic(),
        Operators.MonotoneLocalExtrema(),
    )
    for (j, constraint) in enumerate(constraints)
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
        parameters = (; w, Δt, constraint)
        prob = CTS.ODEProblem(
            CTS.ClimaODEFunction(; T_exp! = tendency!),
            y,
            (t₀, t₁),
            parameters,
        )
        sol = CTS.solve(
            prob,
            CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
            dt = Δt,
            saveat = [t₀:Δt:t₁..., t₁],
        )

        q_final = sol.u[end].q

        @info "Extrema with $(constraint), i=$i, j=$j: $(extrema(q_final))"
        @show maximum(q_final .- 1)
        @show minimum(q_final .- 0)
        @show abs(maximum(q_final .- 1))
        monotonicity_preserving =
            [Operators.MonotoneHarmonic, Operators.MonotoneLocalExtrema]
        if any(x -> constraint isa x, monotonicity_preserving) &&
           stretch_fn == Meshes.Uniform()
            @assert abs(maximum(q_final .- 1)) <= eps(FT)
            @assert abs(minimum(q_final .- 0)) <= eps(FT)
            @assert maximum(q_final) <= FT(1)
        elseif constraint != Operators.AlgebraicMean()
            @assert abs(maximum(q_final .- 1)) <= FT(0.05)
            @assert abs(minimum(q_final .- 0)) <= FT(0.05)
            @assert maximum(q_final) <= FT(1)
        end

        q_analytic = pulse.(z, t₁, z₀, zₕ, z₁)

        err = norm(q_final .- q_analytic)
        rel_mass_err = norm((sum(q_final) - sum(q_init)) / sum(q_init))

        @test err ≤ 0.25
        # 30 eps rather than 10: the GPU reduction sums in a different order,
        # which lands at ~12 eps here (measured 2.8e-15 against a 2.2e-15 bound).
        @test rel_mass_err ≤ 30eps()

        device = ClimaComms.device(q_init)
        if device isa ClimaComms.CUDADevice
            continue
        end

        if j == 1
            fig = Plots.plot(q_analytic; label = "Exact", color = :red)
        end
        linstyl = [:solid, :dot, :dashdot, :dash]
        clrs = [:orange, :blue, :green, :black]
        fig = plot!(
            q_final;
            label = "$(typeof(constraint))"[21:end],
            linestyle = linstyl[j],
            color = clrs[j],
            dpi = 400,
            xlim = (-0.5, 1.1),
            ylim = (-20, 20),
        )
        fig = plot!(legend = :outerbottom, legendcolumns = 2)
        if j == length(constraints)
            Plots.png(
                fig,
                joinpath(
                    path,
                    "limited_advection_" *
                    "$(typeof(constraint))"[21:end] *
                    plot_string[i] *
                    ".png",
                ),
            )
        end
    end
end
