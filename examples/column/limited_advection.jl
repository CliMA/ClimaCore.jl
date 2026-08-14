# Vertical advection with a slope-limited, upwind-biased flux. A square pulse is
# advected along a column by each of the available constraints in turn, and the
# run asserts what each one promises: the two monotonicity-preserving
# constraints may not step outside the initial bounds at all on a uniform mesh,
# `PositiveDefinite` is allowed a small excursion, and `AlgebraicMean` — which
# constrains nothing — is expected to overshoot. All of them must conserve mass
# and land the pulse where the exact solution puts it. `limited_flux_operator`
# isolates the choice of limiter, so a different constraint can be swapped in.
using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaTimeSteppers as CTS

import ClimaCore
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
# The stretched mesh is the harder case: the monotonicity proofs assume uniform
# spacing, so the bounds are only asserted to hold approximately there.

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

# A square pulse of amplitude `q_pulse` on a background of `q_background`,
# translated at constant `speed`, so `pulse(z, t)` is the exact solution.
const FT = Float64
const t_start = FT(0)
const t_end = FT(6)
const q_background = FT(0)
const q_pulse = FT(1)
const pulse_half_width = FT(2π)
const speed = FT(-1)
pulse(z, t) =
    abs(z - speed * t) ≤ pulse_half_width ? q_pulse : q_background

nelems = 2^8
Δt = FT(0.1) * (20π / nelems)
@info "Timestep Δt[s]: $(Δt)"

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(-10π),
    Geometry.ZPoint{FT}(10π);
    boundary_names = (:bottom, :top),
)

stretch_fns = (Meshes.Uniform(), Meshes.ExponentialStretching(FT(7.0)))
mesh_names = ("uniform", "stretched")

constraints = (
    Operators.AlgebraicMean(),
    Operators.PositiveDefinite(),
    Operators.MonotoneHarmonic(),
    Operators.MonotoneLocalExtrema(),
)
# On a uniform mesh these two constraints are proven monotonicity-preserving,
# so they may not leave the initial bounds at all; the others are allowed a
# small excursion (`AlgebraicMean` imposes no constraint and is unbounded).
monotonicity_preserving =
    (Operators.MonotoneHarmonic, Operators.MonotoneLocalExtrema)

line_styles = (:solid, :dot, :dashdot, :dash)
line_colors = (:orange, :blue, :green, :black)

for (stretch_fn, mesh_name) in zip(stretch_fns, mesh_names)
    @info stretch_fn
    mesh = Meshes.IntervalMesh(domain, stretch_fn; nelems = nelems)
    device = ClimaComms.device()
    cent_space = Spaces.CenterFiniteDifferenceSpace(device, mesh)
    face_space = Spaces.FaceFiniteDifferenceSpace(cent_space)
    z = Fields.coordinate_field(cent_space).z

    q_init = pulse.(z, t_start)
    q_analytic = pulse.(z, t_end)
    # Constant advective velocity
    w = Geometry.WVector.(speed .* ones(FT, face_space))

    # Plotting requires scalar indexing, so fields are moved to the CPU first;
    # on a CPU device `to_cpu` is a copy.
    to_cpu(f) = ClimaCore.to_device(ClimaComms.CPUSingleThreaded(), f)
    fig = Plots.plot(to_cpu(q_analytic); label = "Exact", color = :red)
    for (j, constraint) in enumerate(constraints)
        y = Fields.FieldVector(q = copy(q_init))
        parameters = (; w, Δt, constraint)
        prob = CTS.ODEProblem(
            CTS.ClimaODEFunction(; T_exp! = tendency!),
            y,
            (t_start, t_end),
            parameters,
        )
        sol = CTS.solve(
            prob,
            CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
            dt = Δt,
            saveat = [t_start:Δt:t_end..., t_end],
        )

        q_final = sol.u[end].q
        constraint_name = nameof(typeof(constraint))
        @info "Extrema with $(constraint_name) on the $mesh_name mesh: \
               $(extrema(q_final))"

        overshoot = maximum(q_final) - q_pulse
        undershoot = q_background - minimum(q_final)
        if constraint isa Union{monotonicity_preserving...} &&
           stretch_fn == Meshes.Uniform()
            @test overshoot ≤ eps(FT)
            @test undershoot ≤ eps(FT)
        elseif !(constraint isa Operators.AlgebraicMean)
            @test overshoot ≤ FT(0.05)
            @test undershoot ≤ FT(0.05)
        end

        # The pulse is translated, not reshaped: the volume-weighted RMS
        # error stays under 0.25, about half the pulse's own RMS amplitude
        # of 0.45, and the zero-flux boundaries make the flux-form divergence
        # conserve mass exactly.
        @test norm(q_final .- q_analytic) ≤ 0.25
        # 30 eps rather than 10: the GPU reduction sums in a different
        # order and lands at ~12 eps (measured 2.8e-15 against 2.2e-15).
        @test abs(sum(q_final) - sum(q_init)) / sum(q_init) ≤ 30eps()

        Plots.plot!(
            fig,
            to_cpu(q_final);
            label = "$constraint_name",
            linestyle = line_styles[j],
            color = line_colors[j],
            dpi = 400,
            xlim = (-0.5, 1.1),
            ylim = (-20, 20),
        )
    end
    Plots.plot!(fig, legend = :outerbottom, legendcolumns = 2)
    Plots.png(fig, joinpath(path, "limited_advection_$mesh_name.png"))
end
