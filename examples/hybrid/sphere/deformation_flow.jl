using ClimaComms
using OrdinaryDiffEq
using Test
using Statistics: mean

using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Fields, Operators, Limiters
using ClimaTimeSteppers

using Logging
using TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const context = ClimaComms.SingletonCommsContext()
# 3D deformation flow (DCMIP 2012 Test 1-1)
# Reference:
# http://www-personal.umich.edu/~cjablono/DCMIP-2012_TestCaseDocument_v1.7.pdf,
# Section 1.1

const FT = Float32                # floating point type
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
z_top = FT(1.2e4)
zelem = 36
helem = 4
npoly = 4
t_end = FT(60 * 60 * 24 * 12) # 12 days of simulation time
dt = FT(60 * 60) # 1 hour timestep
ode_algorithm = ExplicitAlgorithm(SSP33ShuOsher())

# Operators used in increment!
const hdiv = Operators.Divergence()
const hwdiv = Operators.WeakDivergence()
const hgrad = Operators.Gradient()
const If2c = Operators.InterpolateF2C()
const Ic2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const vdivf2c = Operators.DivergenceF2C(
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
const FCTBorisBook = Operators.FCTBorisBook(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
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

function horizontal_tendency!(Yₜ, Y, cache, t)
    (; u, Δₕq) = cache
    coord = Fields.coordinate_field(u)
    @. u = local_velocity(coord, t)
    @. Δₕq = hwdiv(hgrad(Y.c.ρq / Y.c.ρ))
    Spaces.weighted_dss!(Δₕq)
    @. Yₜ.c.ρ = -hdiv(Y.c.ρ * u)
    for n in 1:5 # TODO: update RecursiveApply/Operators to eliminate this loop
        ρq_n = Y.c.ρq.:($n)
        ρqₜ_n = Yₜ.c.ρq.:($n)
        @. ρqₜ_n = -hdiv(ρq_n * u)
    end
    @. Yₜ.c.ρq -= D₄ * hwdiv(Y.c.ρ * hgrad(Δₕq))
end

function vertical_tendency!(Yₜ, Y, cache, t)
    (; q_n, face_u, face_uₕ, face_uᵥ, fct_op) = cache
    face_coord = Fields.coordinate_field(face_u)
    @. face_u = local_velocity(face_coord, t)
    @. face_uₕ = Geometry.project(Geometry.Covariant12Axis(), face_u)
    @. face_uᵥ = Geometry.project(Geometry.Covariant3Axis(), face_u)
    @. Yₜ.c.ρ = -vdivf2c(Ic2f(Y.c.ρ) * face_u)
    for n in 1:5 # TODO: update RecursiveApply/Operators to eliminate this loop
        ρq_n = Y.c.ρq.:($n)
        ρqₜ_n = Yₜ.c.ρq.:($n)
        @. q_n = ρq_n / Y.c.ρ
        @. ρqₜ_n = -vdivf2c(Ic2f(ρq_n) * face_uₕ)
        if isnothing(fct_op)
            @. ρqₜ_n -= vdivf2c(Ic2f(Y.c.ρ) * upwind3(face_uᵥ, q_n))
        elseif fct_op == FCTBorisBook
            @. ρqₜ_n -= vdivf2c(
                Ic2f(Y.c.ρ) * (
                    upwind1(face_uᵥ, q_n) + FCTBorisBook(
                        upwind3(face_uᵥ, q_n) - upwind1(face_uᵥ, q_n),
                        q_n / dt -
                        vdivf2c(Ic2f(Y.c.ρ) * upwind1(face_uᵥ, q_n)) / Y.c.ρ,
                    )
                ),
            )
        elseif fct_op == FCTZalesak
            @. ρqₜ_n -= vdivf2c(
                Ic2f(Y.c.ρ) * (
                    upwind1(face_uᵥ, q_n) + FCTZalesak(
                        upwind3(face_uᵥ, q_n) - upwind1(face_uᵥ, q_n),
                        q_n / dt,
                        q_n / dt -
                        vdivf2c(Ic2f(Y.c.ρ) * upwind1(face_uᵥ, q_n)) / Y.c.ρ,
                    )
                ),
            )
        else
            error("unrecognized FCT operator $fct_op")
        end
    end
end

function lim!(Y, cache, t, Y_ref)
    (; limiter) = cache
    if !isnothing(limiter)
        Limiters.compute_bounds!(limiter, Y_ref.c.ρq, Y_ref.c.ρ)
        Limiters.apply_limiter!(Y.c.ρq, Y.c.ρ, limiter)
    end
end

function dss!(Y, cache, t)
    Spaces.weighted_dss!(Y.c)
end

function run_deformation_flow(use_limiter, fct_op)
    vert_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(z_top);
        boundary_names = (:bottom, :top),
    )
    vert_mesh = Meshes.IntervalMesh(vert_domain, nelems = zelem)
    vert_cent_space = Spaces.CenterFiniteDifferenceSpace(vert_mesh)

    horz_domain = Domains.SphereDomain(R)
    horz_mesh = Meshes.EquiangularCubedSphere(horz_domain, helem)
    horz_topology = Topologies.Topology2D(context, horz_mesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horz_space = Spaces.SpectralElementSpace2D(horz_topology, quad)

    cent_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horz_space, vert_cent_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(cent_space)

    cent_Y = map(Fields.coordinate_field(cent_space)) do coord
        ϕ = coord.lat
        z = coord.z
        zd = z - z_c

        centers = (
            Geometry.LatLongZPoint(ϕ_c, λ_c1, FT(0)),
            Geometry.LatLongZPoint(ϕ_c, λ_c2, FT(0)),
        )
        horz_geometry = horz_space.global_geometry
        rds = map(centers) do center
            Geometry.great_circle_distance(coord, center, horz_geometry)
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
        Δₕq = Fields.Field(NTuple{5, FT}, cent_space),
        q_n = Fields.Field(FT, cent_space),
        face_u = Fields.Field(Geometry.UVWVector{FT}, face_space),
        face_uₕ = Fields.Field(Geometry.Covariant12Vector{FT}, face_space),
        face_uᵥ = Fields.Field(Geometry.Covariant3Vector{FT}, face_space),
        limiter = use_limiter ? Limiters.QuasiMonotoneLimiter(Y.c.ρq) : nothing,
        fct_op,
    )

    problem = ODEProblem(
        ClimaODEFunction(;
            T_lim! = horizontal_tendency!,
            T_exp! = vertical_tendency!,
            lim!,
            dss!,
        ),
        Y,
        (0, t_end),
        cache,
    )
    return solve(problem, ode_algorithm; dt, saveat = t_end / 2)
end

function conservation_errors(sol)
    initial_total_mass = sum(sol.u[1].c.ρ)
    initial_tracer_masses = map(n -> sum(sol.u[1].c.ρq.:($n)), 1:5)
    final_total_mass = sum(sol.u[end].c.ρ)
    final_tracer_masses = map(n -> sum(sol.u[end].c.ρq.:($n)), 1:5)
    return (
        (final_total_mass - initial_total_mass) / initial_total_mass,
        (final_tracer_masses .- initial_tracer_masses) ./ initial_tracer_masses,
    )
end
tracer_roughnesses(sol) =
    map(1:5) do n
        q_n = sol.u[end].c.ρq.:($n) ./ sol.u[end].c.ρ
        mean_q_n = mean(q_n) # TODO: replace the mean with a low-pass filter
        return mean(abs.(q_n .- mean_q_n))
    end
tracer_ranges(sol) =
    map(1:5) do n
        q_n = sol.u[end].c.ρq.:($n) ./ sol.u[end].c.ρ
        return maximum(q_n) - minimum(q_n)
    end

ref_sol = run_deformation_flow(false, nothing)
fct_sol = run_deformation_flow(false, FCTZalesak)
lim_sol = run_deformation_flow(true, nothing)
lim_fct_sol = run_deformation_flow(true, FCTZalesak)

ref_ρ_err, ref_ρq_errs = conservation_errors(ref_sol)
fct_ρ_err, fct_ρq_errs = conservation_errors(fct_sol)
lim_ρ_err, lim_ρq_errs = conservation_errors(lim_sol)
lim_fct_ρ_err, lim_fct_ρq_errs = conservation_errors(lim_fct_sol)

# Check that the conservation errors are not too big.
max_err = 40 * eps(FT)
@test abs(ref_ρ_err) < max_err
@test all(abs.(ref_ρq_errs) .< max_err)
@test all(abs.(fct_ρq_errs) .< max_err)
@test all(abs.(lim_ρq_errs) .< max_err)
@test all(abs.(lim_fct_ρq_errs) .< max_err)

# Check that FCT and the limiter have no effect on ρ.
@test ref_ρ_err == fct_ρ_err == lim_ρ_err == lim_fct_ρ_err

# Check that FCT and the limiter have no effect on the tracer with q = 1, or at
# least no effect up to round-off error.
max_q5_roundoff_err = 2 * eps(FT)
@test ref_ρq_errs[5] ≈ ref_ρ_err atol = max_q5_roundoff_err
@test fct_ρq_errs[5] ≈ ref_ρ_err atol = max_q5_roundoff_err
@test lim_ρq_errs[5] ≈ ref_ρ_err atol = max_q5_roundoff_err
@test lim_fct_ρq_errs[5] ≈ ref_ρ_err atol = max_q5_roundoff_err

# Check that FCT and the limiter improve the "smoothness" of the tracers.
@test all(tracer_roughnesses(fct_sol) .< tracer_roughnesses(ref_sol))
@test all(tracer_roughnesses(lim_sol) .< 0.9 .* tracer_roughnesses(ref_sol))
@test all(tracer_roughnesses(lim_fct_sol) .< 0.8 .* tracer_roughnesses(ref_sol))
@test all(tracer_ranges(fct_sol) .< tracer_ranges(ref_sol))
@test all(tracer_ranges(lim_sol) .< 0.6 .* tracer_ranges(ref_sol))
@test all(tracer_ranges(lim_fct_sol) .< 0.5 .* tracer_ranges(ref_sol))

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
path = joinpath(@__DIR__, "output", "deformation_flow")
mkpath(path)
for (sol, suffix) in (
    (ref_sol, ""),
    (fct_sol, "_fct"),
    (lim_sol, "_lim"),
    (lim_fct_sol, "_lim_fct"),
)
    for (sol_index, day) in ((1, 6), (2, 12))
        Plots.png(
            Plots.plot(
                sol.u[sol_index].c.ρq.:3 ./ sol.u[sol_index].c.ρ,
                level = 15,
                clim = (-1, 1),
            ),
            joinpath(path, "q3_day$day$suffix.png"),
        )
    end
end
