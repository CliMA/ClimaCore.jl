#=
julia --project=.buildkite
using Revise; include("examples/hybrid/sphere/deformation_flow.jl")
=#
import ClimaComms
ClimaComms.@import_required_backends
using SciMLBase: ODEProblem, init, solve
using Test
using Statistics: mean

using ClimaCore:
    Geometry,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Fields,
    Operators,
    Limiters,
    Quadratures
using ClimaTimeSteppers

using Logging
using TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const context = ClimaComms.SingletonCommsContext()
# 3D deformation flow (DCMIP 2012 Test 1-1)
# Reference:
# http://www-personal.umich.edu/~cjablono/DCMIP-2012_TestCaseDocument_v1.7.pdf,
# Section 1.1

const FT = Float64                # floating point type
const R = FT(6.37122e6)           # radius
const grav = FT(9.8)              # gravitational constant
const R_d = FT(287.058)           # R dry (gas constant / mol mass dry air)
const p_top = FT(25494.4)         # pressure at the model top
const T_0 = FT(300)               # isothermal atmospheric temperature
const H = R_d * T_0 / grav        # scale height
const p_0 = FT(1e5)               # reference pressure
const τ = FT(1036800)             # period of motion
const ω_0 = FT(π) * FT(23000) / τ # maxium of the vertical pressure velocity
const b = FT(0.2)                 # normalized pressure depth of divergent layer
const λ_c1 = FT(150)              # initial longitude of first tracer
const λ_c2 = FT(210)              # initial longitude of second tracer
const ϕ_c = FT(0)                 # initial latitude of tracers
const z_c = FT(5e3)               # initial altitude of tracers
const R_t = R / 2                 # horizontal half-width of tracers
const Z_t = FT(1000)              # vertical half-width of tracers
const D₄ = FT(1e16)               # hyperviscosity coefficient

# Parameters used in run_deformation_flow
const z_top = FT(1.2e4)
const zelem = 36
const helem = 4
const npoly = 4
const t_end = FT(60 * 60 * 24 * 12) # 12 days of simulation time
const _dt = FT(60 * 60) # 1 hour timestep
ode_algorithm = ExplicitAlgorithm(SSP33ShuOsher())

# Operators used in increment!
const hdiv = Operators.Divergence()
const hwdiv = Operators.WeakDivergence()
const hgrad = Operators.Gradient()
const interp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const winterp = Operators.WeightedInterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const vdiv = Operators.DivergenceF2C(
    top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
)
const upwind1 = Operators.UpwindBiasedProductC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const upwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.ThirdOrderOneSided(),
    top = Operators.ThirdOrderOneSided(),
)
const FCTZalesak = Operators.FCTZalesak(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
)
const SlopeLimitedFlux = Operators.TVDLimitedFluxC2F(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
    method = Operators.MinModLimiter(),
)
const LinVanLeerFlux = Operators.LinVanLeerC2F(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
    constraint = Operators.MonotoneLocalExtrema(),
)

# Reference pressure and density
p(z) = p_0 * exp(-z / H)
ρ_ref(z) = p(z) / R_d / T_0

# Local wind velocity forcing
function local_velocity(coord, t)
    ϕ = coord.lat
    λ = coord.long
    z = coord.z
    t = FT(t) # TODO: temporary bugfix for ClimaTimeSteppers.jl type instability

    k = 10 * R / τ
    λp = λ - FT(360) * t / τ

    ua =
        k * sind(λp)^2 * sind(2 * ϕ) * cos(FT(π) * t / τ) +
        2 * FT(π) * R / τ * cosd(ϕ)
    ud =
        ω_0 * R / b / p_top *
        cosd(λp) *
        cosd(ϕ)^2 *
        cos(2 * FT(π) * t / τ) *
        (-exp((p(z) - p_0) / b / p_top) + exp((p_top - p(z)) / b / p_top))
    uu = ua + ud

    uv = k * sind(2 * λp) * cosd(ϕ) * cos(FT(π) * t / τ)

    sp =
        1 + exp((p_top - p_0) / b / p_top) - exp((p(z) - p_0) / b / p_top) -
        exp((p_top - p(z)) / b / p_top)
    ω = ω_0 * sind(λp) * cosd(ϕ) * cos(2 * FT(π) * t / τ) * sp
    uw = -ω / ρ_ref(z) / grav

    return Geometry.UVWVector(uu, uv, uw)
end

function T_exp_T_lim!(Yₜ, Yₜ_lim, Y, cache, t)
    horizontal_tendency!(Yₜ_lim, Y, cache, t)
    vertical_tendency!(Yₜ, Y, cache, t)
end

function horizontal_tendency!(Yₜ, Y, cache, t)
    (; u, Δₕq) = cache
    coord = Fields.coordinate_field(Y.c)
    @. u = local_velocity(coord, t)
    @. Δₕq = hwdiv(hgrad(Y.c.ρq / Y.c.ρ))
    Spaces.weighted_dss!(Δₕq)
    @. Yₜ.c.ρ = -hdiv(Y.c.ρ * u)
    @. Yₜ.c.ρq = -hdiv(Y.c.ρq * u) - D₄ * hwdiv(Y.c.ρ * hgrad(Δₕq))
end

function vertical_tendency!(Yₜ, Y, cache, t)
    (; q, face_ρ, face_u, fct_op, dt) = cache
    (; J) = Fields.local_geometry_field(Y.c)
    face_coord = Fields.coordinate_field(face_u)
    @. q = Y.c.ρq / Y.c.ρ
    @. face_ρ = winterp(J, Y.c.ρ)
    @. face_u = local_velocity(face_coord, t)
    @. Yₜ.c.ρ = -vdiv(face_ρ * face_u)
    if isnothing(fct_op)
        @. Yₜ.c.ρq = -vdiv(face_ρ * face_u * interp(q))
    elseif fct_op == upwind1
        @. Yₜ.c.ρq = -vdiv(face_ρ * upwind1(face_u, q))
    elseif fct_op == upwind3
        @. Yₜ.c.ρq = -vdiv(face_ρ * upwind3(face_u, q))
    elseif fct_op == FCTZalesak
        @. Yₜ.c.ρq =
            -vdiv(
                face_ρ * upwind1(face_u, q) +
                FCTZalesak(
                    face_ρ * (upwind3(face_u, q) - upwind1(face_u, q)),
                    q / dt,
                    q / dt - vdiv(face_ρ * upwind1(face_u, q)) / Y.c.ρ,
                ),
            )
    elseif fct_op == SlopeLimitedFlux
        @. Yₜ.c.ρq =
            -vdiv(
                face_ρ * upwind1(face_u, q) +
                SlopeLimitedFlux(
                    face_ρ * (upwind3(face_u, q) - upwind1(face_u, q)),
                    q / dt,
                    face_u,
                ),
            )
    elseif fct_op == LinVanLeerFlux
        @. Yₜ.c.ρq = -vdiv(face_ρ * LinVanLeerFlux(face_u, q, dt))
    else
        error("unrecognized FCT operator $fct_op")
    end
end

function lim!(Y, cache, t, Y_ref)
    (; limiter) = cache
    if !isnothing(limiter)
        Limiters.compute_bounds!(limiter, Y_ref.c.ρq, Y_ref.c.ρ)
        Limiters.apply_limiter!(Y.c.ρq, Y.c.ρ, limiter; warn = false)
    end
end

function dss!(Y, cache, t)
    Spaces.weighted_dss!(Y.c)
end

function run_deformation_flow(use_limiter, fct_op, dt)
    vert_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(z_top);
        boundary_names = (:bottom, :top),
    )
    vert_mesh = Meshes.IntervalMesh(vert_domain, nelems = zelem)
    device = ClimaComms.device(context)
    vert_cent_space = Spaces.CenterFiniteDifferenceSpace(device, vert_mesh)

    horz_domain = Domains.SphereDomain(R)
    horz_mesh = Meshes.EquiangularCubedSphere(horz_domain, helem)
    horz_topology = Topologies.Topology2D(context, horz_mesh)
    quad = Quadratures.GLL{npoly + 1}()
    horz_space = Spaces.SpectralElementSpace2D(horz_topology, quad)

    cent_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horz_space, vert_cent_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(cent_space)

    cent_Y = map(Fields.coordinate_field(cent_space)) do coord
        ϕ = coord.lat
        z = coord.z
        zd = z - z_c

        centers = (
            Geometry.LatLongZPoint(ϕ_c, λ_c1, z),
            Geometry.LatLongZPoint(ϕ_c, λ_c2, z),
        )
        rds = map(centers) do center
            Geometry.great_circle_distance(
                coord,
                center,
                Spaces.global_geometry(cent_space),
            )
        end
        ds = @. min(1, (rds / R_t)^2 + (zd / Z_t)^2) # scaled distance functions

        q1 = (1 + cos(FT(π) * ds[1])) / 2 + (1 + cos(FT(π) * ds[2])) / 2
        q2 = FT(0.9) - FT(0.8) * q1^2
        q3 =
            if (ds[1] < FT(0.5) || ds[2] < FT(0.5)) && !(
                (z > z_c) &&
                (ϕ > ϕ_c - rad2deg(1 / 8)) &&
                (ϕ < ϕ_c + rad2deg(1 / 8))
            )
                FT(1)
            else
                FT(0.1)
            end
        q4 = 1 - FT(0.3) * (q1 + q2 + q3)
        q5 = FT(1)

        ρ = ρ_ref(z)
        return (; ρ, ρq = (ρ * q1, ρ * q2, ρ * q3, ρ * q4, ρ * q5))
    end
    Y = Fields.FieldVector(; c = cent_Y)

    cache = (;
        u = Fields.Field(Geometry.UVWVector{FT}, cent_space),
        q = Fields.Field(NTuple{5, FT}, cent_space),
        Δₕq = Fields.Field(NTuple{5, FT}, cent_space),
        face_ρ = Fields.Field(FT, face_space),
        face_u = Fields.Field(Geometry.UVWVector{FT}, face_space),
        limiter = use_limiter ? Limiters.QuasiMonotoneLimiter(Y.c.ρq) : nothing,
        fct_op,
        dt,
    )

    problem = ODEProblem(
        ClimaODEFunction(; T_exp_T_lim!, lim!, dss!),
        Y,
        (0, t_end),
        cache,
    )
    sol = solve(problem, ode_algorithm; dt)
    if !(cache.limiter isa Nothing)
        @show cache.limiter.rtol
        Limiters.print_convergence_stats(cache.limiter)
    end
    return sol
end

function total_conservation_error(sol)
    initial_mass = sum(sol[1].c.ρ)
    final_mass = sum(sol[end].c.ρ)
    return abs(final_mass - initial_mass) / initial_mass
end

function tracer_conservation_errors(sol)
    initial_masses = sum(sol[1].c.ρq)
    final_masses = sum(sol[end].c.ρq)
    return abs.(final_masses .- initial_masses) ./ initial_masses
end

# Roughness measured as deviation from mean (TODO: use a low-pass filter instead)
function tracer_roughnesses(sol)
    final_q = sol[end].c.ρq ./ sol[end].c.ρ
    return mean(abs.(final_q .- mean(final_q)))
end

function tracer_ranges(sol)
    final_q = sol[end].c.ρq ./ sol[end].c.ρ
    return maximum(final_q) .- minimum(final_q)
end

@info "Centered Differences"
centered_sol_no_lim = run_deformation_flow(false, nothing, _dt)
centered_sol_with_lim = run_deformation_flow(true, nothing, _dt)
@info "First-Order Upwinding"
upwind1_sol_no_lim = run_deformation_flow(false, upwind1, _dt)
upwind1_sol_with_lim = run_deformation_flow(true, upwind1, _dt)
@info "Third-Order Upwinding"
upwind3_sol_no_lim = run_deformation_flow(false, upwind3, _dt)
upwind3_sol_with_lim = run_deformation_flow(true, upwind3, _dt)
@info "Flux-Corrected Transport"
fct_sol_no_lim = run_deformation_flow(false, FCTZalesak, _dt)
fct_sol_with_lim = run_deformation_flow(true, FCTZalesak, _dt)
@info "Slope-Limited Transport"
tvd_sol_no_lim = run_deformation_flow(false, SlopeLimitedFlux, _dt)
tvd_sol_with_lim = run_deformation_flow(true, SlopeLimitedFlux, _dt)
@info "van Leer Transport"
lvl_sol_no_lim = run_deformation_flow(false, LinVanLeerFlux, _dt)
lvl_sol_with_lim = run_deformation_flow(true, LinVanLeerFlux, _dt)

sols_no_lim = (;
    centered = centered_sol_no_lim,
    upwind1 = upwind1_sol_no_lim,
    upwind3 = upwind3_sol_no_lim,
    fct = fct_sol_no_lim,
    tvd = tvd_sol_no_lim,
    lvl = lvl_sol_no_lim,
)
sols_with_lim = (;
    centered = centered_sol_with_lim,
    upwind1 = upwind1_sol_with_lim,
    upwind3 = upwind3_sol_with_lim,
    fct = fct_sol_with_lim,
    tvd = tvd_sol_with_lim,
    lvl = lvl_sol_with_lim,
)

ρ_errs_no_lim = map(total_conservation_error, sols_no_lim)
ρ_errs_with_lim = map(total_conservation_error, sols_with_lim)
ρq_errs_no_lim = map(tracer_conservation_errors, sols_no_lim)
ρq_errs_with_lim = map(tracer_conservation_errors, sols_with_lim)
roughnesses_no_lim = map(tracer_roughnesses, sols_no_lim)
roughnesses_with_lim = map(tracer_roughnesses, sols_with_lim)
ranges_no_lim = map(tracer_ranges, sols_no_lim)
ranges_with_lim = map(tracer_ranges, sols_with_lim)

# Check that upwinding has no effect on total mass.
for ρ_errs_data in (ρ_errs_no_lim, ρ_errs_with_lim), ρ_err in ρ_errs_data
    @test ρ_err == ρ_errs_no_lim.centered
end

# Check that upwinding has no effect on the constant tracer q5, and that the
# other non-constant tracers are all conserved, accounting for round-off errors.
for ρq_errs_data in (ρq_errs_no_lim, ρq_errs_with_lim), ρq_errs in ρq_errs_data
    @test ρq_errs[5] ≈ ρ_errs_no_lim.centered atol = eps(FT)
    @test all(ρq_errs[1:4] .< 40 * eps(FT))
end

# Check that using a limiter improves the "smoothness" of non-constant tracers.
for (no_lim, with_lim) in zip(roughnesses_no_lim, roughnesses_with_lim)
    @test all(with_lim[1:4] .< no_lim[1:4] .* 0.9999)
end
for (no_lim, with_lim) in zip(ranges_no_lim, ranges_with_lim)
    @test all(with_lim[1:4] .< no_lim[1:4] .* 0.992)
end

# Check that the relative effects of different upwinding schemes are consistent.
for data in (roughnesses_no_lim, roughnesses_with_lim, ranges_no_lim, ranges_with_lim)
    @test all((data.upwind1 .< data.tvd .< data.lvl .< data.fct .< data.upwind3)[1:4])
end

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
path = joinpath(@__DIR__, "output", "deformation_flow")
mkpath(path)

ref_final_q3 = upwind3_sol_with_lim[end].c.ρq.:3 ./ upwind3_sol_with_lim[end].c.ρ
for (lim_suffix, sols) in (("no_lim", sols_no_lim), ("with_lim", sols_with_lim))
    for (name, sol) in pairs(sols)
        final_q3 = sol[end].c.ρq.:3 ./ sol[end].c.ρ
        Plots.png(
            Plots.plot(final_q3, level = 15, clim = (-1, 1)),
            joinpath(path, "q3_day12_$(name)_$(lim_suffix).png"),
        )
        sol === upwind3_sol_with_lim && continue # skip diff plot for reference
        Plots.png(
            Plots.plot(final_q3 .- ref_final_q3, level = 15, clim = (-0.2, 0.2)),
            joinpath(path, "q3_diff_day12_$(name)_$(lim_suffix).png"),
        )
    end
end
