# 1D vertical advection of a sine wave, solved four ways: the first-order upwind
# flux form (`UpwindBiasedProductC2F`), the centered advective form (an
# interpolated center-to-face gradient), and each of those with a diffusive flux
# correction added. The exact solution is sin(z - t), which supplies the
# boundary values, so the errors measure what each formulation does to the wave:
# the centered form is nearly exact but relies on the problem's smoothness, the
# upwind form damps, and the flux correction adds numerical diffusion to
# whatever it is applied to.
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

import ClimaTimeSteppers as CTS
import LazyBroadcast: lazy

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
const FT = Float64

const z_left = FT(0)
const z_right = FT(4pi)
const nelems = 128

domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(z_left),
    Geometry.ZPoint{FT}(z_right),
    boundary_names = (:left, :right),
)
mesh = Meshes.IntervalMesh(domain, nelems = nelems)
device = ClimaComms.device()
cspace = Spaces.CenterFiniteDifferenceSpace(device, mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

# Unit upward velocity, so the exact solution of ∂θ/∂t = -∂(wθ)/∂z is the
# initial sine wave translated at unit speed. It also supplies the inflow
# boundary values.
w = Geometry.WVector.(ones(FT, fspace))
exact_θ(z, t) = sin(z - t)
θ_init = exact_θ.(Fields.coordinate_field(cspace).z, 0)

# A diffusive flux with diffusivity |w| Δz, added to whichever formulation it is
# applied to. Diffusion is symmetric rather than upwind-biased; what makes it a
# correction is that first-order upwinding introduces numerical diffusion of
# this form, of half this size (the difference between the upwind and centered
# fluxes is exactly half of this term). It carries no boundary values of its
# own: the zero gradient on each boundary face sets its flux through that face
# to zero.
function flux_correction(θ)
    zero_gradient =
        Operators.SetGradient(Geometry.Covariant3Vector(zero(FT)))
    gradc2f =
        Operators.GradientC2F(left = zero_gradient, right = zero_gradient)
    gradf2c = Operators.GradientF2C()
    lg_field = Fields.local_geometry_field(fspace)
    return @. lazy(
        adjoint(
            gradf2c(
                adjoint(gradc2f(θ)) * Geometry.Contravariant3Vector(
                    abs(Geometry.contravariant3(w, lg_field)),
                ),
            ),
        ) * Geometry.Contravariant3Vector(1),
    )
end

# Flux form: the face flux is reconstructed by first-order upwinding, with the
# exact value of θ standing in for the point outside the domain at each
# boundary face, imposed by `upwind_biased_product_c2f_dirichlet`.
upwind_flux(θ, t) = Operators.upwind_biased_product_c2f_dirichlet(
    w,
    θ;
    left = exact_θ(z_left, t),
    right = exact_θ(z_right, t),
)

# Advective form: centered differences, with the exact value at the inflow
# boundary and extrapolation at the outflow one. The gradient on the left
# boundary face is the one implied by θ = exact_θ(z_left, t) outside of the
# domain (imposed by `gradient_c2f_dirichlet`); on the right boundary face it
# is the gradient at the closest interior face, built from the last two center
# values and passed through as an explicit `SetGradient`.
function centered_advection_gradient(θ, t)
    θ_n = Fields.level(θ, Fields.nlevels(θ))
    θ_nm1 = Fields.level(θ, Fields.nlevels(θ) - 1)
    return Operators.gradient_c2f_dirichlet(
        θ;
        left = exact_θ(z_left, t),
        right = Operators.SetGradient(
            @. lazy(Geometry.Covariant3Vector(θ_n - θ_nm1))
        ),
    )
end

function tendency_upwind!(dθ, θ, _, t)
    flux = upwind_flux(θ, t)
    divf2c = Operators.DivergenceF2C()
    return @. dθ = -divf2c(flux)
end
function tendency_upwind_corrected!(dθ, θ, _, t)
    flux = upwind_flux(θ, t)
    divf2c = Operators.DivergenceF2C()
    correction = flux_correction(θ)
    return @. dθ = -divf2c(flux) + correction
end
function tendency_centered!(dθ, θ, _, t)
    ∇θ = centered_advection_gradient(θ, t)
    interpf2c = Operators.InterpolateF2C()
    return @. dθ = -interpf2c(
        Geometry.dot(Geometry.Contravariant3Vector(w), ∇θ),
    )
end
function tendency_centered_corrected!(dθ, θ, _, t)
    ∇θ = centered_advection_gradient(θ, t)
    interpf2c = Operators.InterpolateF2C()
    correction = flux_correction(θ)
    return @. dθ =
        -interpf2c(
            Geometry.dot(Geometry.Contravariant3Vector(w), ∇θ),
        ) + correction
end

tendency_upwind!(similar(θ_init), θ_init, nothing, 0.0)

# Solve the ODE operator
Δt = 0.001
t_end = 10.0
tendencies = (;
    upwind = tendency_upwind!,
    upwind_corrected = tendency_upwind_corrected!,
    centered = tendency_centered!,
    centered_corrected = tendency_centered_corrected!,
)
solutions = map(tendencies) do tendency!
    prob = CTS.ODEProblem(
        CTS.ClimaODEFunction(; T_exp! = tendency!),
        copy(θ_init),
        (0.0, t_end),
        nothing,
    )
    CTS.solve(
        prob,
        CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
        dt = Δt,
        saveat = collect(0.0:(10 * Δt):t_end),
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )
end

using Test
@testset "advected profiles vs the analytic solution" begin
    z = vec(parent(Fields.coordinate_field(cspace).z))
    exact = exact_θ.(z, t_end)
    final(sol) = vec(parent(sol.u[end]))
    Linf(sol) = maximum(abs, final(sol) .- exact)
    # Root-mean-square amplitude: the wave carries variance, and only the
    # dissipative formulations may lose it.
    rms(values) = sqrt(sum(abs2, values) / length(values))
    # Measured errors: upwind 0.39, upwind+correction 0.77, centered 0.018,
    # centered+correction 0.62. The centered scheme's bound is what pins the
    # physics: at this resolution its phase error is ~2e-2, so anything larger
    # means the discretization (not the formulation) has broken.
    @test Linf(solutions.upwind) < 0.5
    @test Linf(solutions.upwind_corrected) < 1.0
    @test Linf(solutions.centered) < 0.05
    @test Linf(solutions.centered_corrected) < 1.0
    # None of the schemes may create new extrema (exact solution is in [-1, 1]).
    for sol in solutions
        @test maximum(abs, final(sol)) < 1.05
    end
    # The centered form is non-dissipative: it carries the wave at full
    # amplitude (measured rms 0.7076 against the exact 0.7071). Upwinding and
    # the flux correction are both diffusive, so every other formulation loses
    # variance, and applying the correction on top of either form loses more
    # (measured rms: upwind 0.53, upwind+correction 0.35, centered+correction
    # 0.42). This ordering is the point of the example.
    @test rms(final(solutions.centered)) ≈ rms(exact) rtol = 0.01
    @test rms(final(solutions.upwind)) < 0.9 * rms(exact)
    @test rms(final(solutions.upwind_corrected)) < rms(final(solutions.upwind))
    @test rms(final(solutions.centered_corrected)) <
          rms(final(solutions.centered))
end

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "advect"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

for (name, sol) in pairs(solutions)
    anim = Plots.@animate for θ in sol.u
        Plots.plot(θ, xlim = (-1, 1))
    end
    Plots.mp4(anim, joinpath(path, "advect_$name.mp4"), fps = 10)
    Plots.png(
        Plots.plot(sol.u[end], xlim = (-1, 1)),
        joinpath(path, "advect_$(name)_end.png"),
    )
end

z_centers = vec(parent(Fields.coordinate_field(cspace).z))
line_styles = (:dash, :dot, :solid, :dashdot)
comparison = Plots.plot(
    exact_θ.(z_centers, t_end),
    z_centers,
    xlim = (-1.1, 1.1),
    color = :black,
    xlabel = "θ",
    ylabel = "z",
    label = "Exact",
)
for ((name, sol), style) in zip(pairs(solutions), line_styles)
    Plots.plot!(
        comparison,
        vec(parent(sol.u[end])),
        z_centers,
        ls = style,
        label = "$name",
    )
end
Plots.png(comparison, joinpath(path, "advect_all_end.png"))

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "advect_all_end.png"), joinpath(@__DIR__, "../..")),
    "Advect End Simulation",
)
