#=
julia --project=.buildkite
ARGS = ["cosine_bells"];
# ARGS = ["gaussian_bells"];
# ARGS = ["slotted_spheres"];
using Revise; include(joinpath("examples", "hybrid", "box", "limiters_advection.jl"))
=#
using ClimaComms
if pkgversion(ClimaComms) >= v"0.6"
    ClimaComms.@import_required_backends
end
using LinearAlgebra
using SciMLBase

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Quadratures,
    Topologies,
    Limiters,
    slab
import ClimaCore.Geometry: ⊗
import ClimaTimeSteppers as CTS

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

"""
    convergence_rate(err, Δh)

Estimate convergence rate given vectors `err` and `Δh`

    err = C Δh^p + H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p

"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

# Function space setup
function hvspace_3D(
    ::Type{FT};
    device,
    context = ClimaComms.SingletonCommsContext(device),
    xlim = (-2π, 2π),
    ylim = (-2π, 2π),
    zlim = (0, 4π),
    xelems = 16,
    yelems = 16,
    zelems = 16,
    Nij = 2,
) where {FT}

    xdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    ydomain = Domains.IntervalDomain(
        Geometry.YPoint{FT}(ylim[1]),
        Geometry.YPoint{FT}(ylim[2]),
        periodic = true,
    )

    horzdomain = Domains.RectangleDomain(xdomain, ydomain)
    horzmesh = Meshes.RectilinearMesh(horzdomain, xelems, yelems)
    horztopology = Topologies.Topology2D(context, horzmesh)

    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(zdomain, nelems = zelems)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)

    quad = Quadratures.GLL{Nij}()
    hspace = Spaces.SpectralElementSpace2D(horztopology, quad)
    cspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, z_space)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    return (cspace, fspace)
end

# Advection problem on a 3D Cartesian domain with bounds-preserving quasimonotone horizontal limiter.
# The initial condition can be set via a command line argument.
# Possible test cases are: cosine_bells (default), gaussian_bells, and slotted_spheres

# Set up physical parameters
Base.@kwdef struct LimAdvectionParams{FT}
    xmin::FT = -2π              # domain x lower bound
    xmax::FT = 2π               # domain x upper bound
    ymin::FT = -2π              # domain y lower bound
    ymax::FT = 2π               # domain y upper bound
    zmin::FT = 0                # domain z lower bound
    zmax::FT = 4π               # domain z upper bound
    ρ₀::FT = 1.0                # air density
    D₄::FT = 0.0                # hyperdiffusion coefficient
    u0::FT = π / 2              # angular velocity
    r0::FT = (xmax - xmin) / 6  # bells radius
    end_time::FT = 2π           # simulation period in seconds
    dt::FT = end_time / 2000
    n_steps::Int = Int(round(end_time / dt))
    flow_center::Geometry.XYZPoint{FT} = Geometry.XYZPoint(
        xmin + (xmax - xmin) / 2,
        ymin + (ymax - ymin) / 2,
        zmin + (zmax - zmin) / 2,
    )
    bell_centers::Tuple{Geometry.XYZPoint{FT}, Geometry.XYZPoint{FT}} = (
        Geometry.XYZPoint(
            xmin + (xmax - xmin) / 4,
            ymin + (ymax - ymin) / 2,
            zmin + (zmax - zmin) / 2,
        ),
        Geometry.XYZPoint(
            xmin + 3 * (xmax - xmin) / 4,
            ymin + (ymax - ymin) / 2,
            zmin + (zmax - zmin) / 2,
        ),
    )
    zelems::Int = 8
end
Base.broadcastable(x::LimAdvectionParams) = tuple(x)

abstract type AbstractTestCase end
struct CosineBells <: AbstractTestCase end
struct GaussianBells <: AbstractTestCase end
struct SlottedSpheres <: AbstractTestCase end

test_cases = Dict(
    "cosine_bells" => CosineBells(),
    "gaussian_bells" => GaussianBells(),
    "slotted_spheres" => SlottedSpheres(),
)
for k in keys(test_cases)
    @eval name(::typeof($(test_cases[k]))) = $k
end
test_case = test_cases[get(ARGS, 1, "cosine_bells")]
# Set up test parameters
const lim_flag = true

function init_state(cspace, fspace, test_case, params)
    (; bell_centers, r0, ρ₀) = params
    # Initialize state
    ρ_init = ρ₀ .* ones(cspace)
    q_init = map(Fields.coordinate_field(cspace)) do coord
        rd = (
            Geometry.euclidean_distance(coord, bell_centers[1]),
            Geometry.euclidean_distance(coord, bell_centers[2]),
        )
        (; x, y, z) = coord
        if test_case isa SlottedSpheres
            if rd[1] <= r0 && abs(x - bell_centers[1].x) >= r0 / 6
                return 1.0
            elseif rd[2] <= r0 && abs(x - bell_centers[2].x) >= r0 / 6
                return 1.0
            elseif rd[1] <= r0 &&
                   abs(x - bell_centers[1].x) < r0 / 6 &&
                   (y - bell_centers[1].y) < -5 * r0 / 12
                return 1.0
            elseif rd[2] <= r0 &&
                   abs(x - bell_centers[2].x) < r0 / 6 &&
                   (y - bell_centers[2].y) > 5 * r0 / 12
                return 1.0
            else
                return 0.1
            end
        elseif test_case isa GaussianBells
            return 0.95 * (exp(-5.0 * (rd[1] / r0)^2) + exp(-5.0 * (rd[2] / r0)^2))
        else # default test case, cosine bells
            if rd[1] < r0
                return 0.1 + 0.9 * (1 / 2) * (1 + cospi(rd[1] / r0))
            elseif rd[2] < r0
                return 0.1 + 0.9 * (1 / 2) * (1 + cospi(rd[2] / r0))
            else
                return 0.1
            end
        end
    end
    return (; q_init, y = Fields.FieldVector(ρ = ρ_init, ρq = ρ_init .* q_init))
end


# Plot variables and auxiliary function
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
device = ClimaComms.device()
dev_suffix = device isa ClimaComms.CUDADevice ? "_gpu" : ""
dirname = "box_advection_limiter_$(name(test_case))$dev_suffix"

FT = Float64;
params = LimAdvectionParams{FT}();

if lim_flag == false
    dirname = "$(dirname)_no_lim"
end
if params.D₄ == 0
    dirname = "$(dirname)_D0"
end

path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

function linkfig(figpath, alt = "")
    rpath = relpath(figpath, joinpath(@__DIR__, "../../.."))
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$rpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

function local_velocity(params, coord, t)
    (; u0, flow_center, end_time, zmax) = params
    Geometry.UVWVector(
        -u0 * (coord.y - flow_center.y) * cospi(t / end_time),
        u0 * (coord.x - flow_center.x) * cospi(t / end_time),
        u0 * sinpi(coord.z / zmax) * cospi(t / end_time),
    )
end

function horizontal_tendency!(yₜ, y, parameters, t)
    (; u, Δₕq, params) = parameters
    (; D₄) = params
    grad = Operators.Gradient()
    wdiv = Operators.WeakDivergence()
    coord = Fields.coordinate_field(axes(u))
    @. u = local_velocity(params, coord, t)
    @. Δₕq = wdiv(grad(y.ρq / y.ρ))
    Spaces.weighted_dss!(Δₕq)
    @. yₜ.ρ = -wdiv(y.ρ * u)
    @. yₜ.ρq = -wdiv(y.ρq * u) - D₄ * wdiv(y.ρ * grad(Δₕq))
end

function vertical_tendency!(yₜ, y, cache, t)
    (; face_u, params) = cache
    FT = Spaces.undertype(axes(y.ρ))
    Ic2f = Operators.InterpolateC2F()
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    )
    upwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    ax12 = (Geometry.Covariant12Axis(),)
    ax3 = (Geometry.Covariant3Axis(),)
    face_coord = Fields.coordinate_field(face_u)
    @. face_u = local_velocity(params, face_coord, t)
    @. yₜ.ρ = -vdivf2c(Ic2f(y.ρ) * face_u)
    @. yₜ.ρq =
        -vdivf2c(Ic2f(y.ρq) * Geometry.project(ax12, face_u)) -
        vdivf2c(Ic2f(y.ρ) * upwind3(Geometry.project(ax3, face_u), y.ρq / y.ρ))
end

function lim!(y, parameters, t, y_ref)
    (; limiter) = parameters
    if lim_flag
        Limiters.compute_bounds!(limiter, y_ref.ρq, y_ref.ρ)
        Limiters.apply_limiter!(y.ρq, y.ρ, limiter)
    end
end

function dss!(y, parameters, t)
    Spaces.weighted_dss!(y.ρ)
    Spaces.weighted_dss!(y.ρq)
end

# Set up spatial discretization
horz_ne_seq = 2 .^ (2, 3, 4, 5)
Δh = zeros(length(horz_ne_seq))
L1err = zeros(length(horz_ne_seq))
L2err = zeros(length(horz_ne_seq))
Linferr = zeros(length(horz_ne_seq))
Nij = 3
(; xmax, xmin, n_steps, end_time, D₄, dt, zelems) = params
# h-refinement study loop
for (k, horz_ne) in enumerate(horz_ne_seq)
    # Set up 3D spatial domain - doubly periodic box
    cspace, fspace = hvspace_3D(
        FT;
        device,
        Nij = Nij,
        xelems = horz_ne,
        yelems = horz_ne,
        zelems = zelems,
    )
    (; q_init, y) = init_state(cspace, fspace, test_case, params)

    # Solve the ODE
    parameters = (
        u = Fields.Field(Geometry.UVWVector{FT}, cspace),
        Δₕq = Fields.Field(FT, cspace),
        face_u = Fields.Field(Geometry.UVWVector{FT}, fspace),
        limiter = Limiters.QuasiMonotoneLimiter(q_init),
        params,
    )
    prob = SciMLBase.ODEProblem(
        CTS.ClimaODEFunction(;
            T_lim! = horizontal_tendency!,
            T_exp! = vertical_tendency!,
            lim!,
            dss!,
        ),
        y,
        (0.0, end_time),
        parameters,
    )
    sol = SciMLBase.solve(
        prob,
        CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
        dt = dt,
        saveat = 0.99 * 80 * dt,
    )

    q_final = sol.u[end].ρq ./ sol.u[end].ρ
    Δh[k] = (xmax - xmin) / horz_ne
    L1err[k] = norm((q_final .- q_init) ./ q_init, 1)
    L2err[k] = norm((q_final .- q_init) ./ q_init)
    Linferr[k] = norm((q_final .- q_init) ./ q_init, Inf)

    @info "Test case: $(name(test_case))"
    @info "With limiter: $(lim_flag)"
    @info "Hyperdiffusion coefficient: D₄ = $(D₄)"
    @info "Number of elements in XYZ domain: $(horz_ne) x $(horz_ne) x $(zelems)"
    @info "Number of quadrature points per horizontal element: $(Nij) x $(Nij) (p = $(Nij-1))"
    @info "Time step dt = $(dt) (s)"
    @info "Tracer concentration norm at t = 0 (s): ", norm(q_init)
    @info "Tracer concentration norm at $(n_steps) time steps, t = $(params.end_time) (s): ",
    norm(q_final)
    @info "L₁ error at $(n_steps) time steps, t = $(end_time) (s): ", L1err[k]
    @info "L₂ error at $(n_steps) time steps, t = $(end_time) (s): ", L2err[k]
    @info "L∞ error at $(n_steps) time steps, t = $(end_time) (s): ", Linferr[k]
end

# Print convergence rate info
conv = convergence_rate(L2err, Δh)
@info "Converge rates for this test case are: ", conv

# Plot the errors
# L₁ error Vs number of elements
Plots.png(
    Plots.plot(
        collect(horz_ne_seq),
        L1err,
        yscale = :log10,
        xlabel = "Nₑ",
        ylabel = "log₁₀(L₁ err)",
        label = "",
    ),
    joinpath(path, "L1error.png"),
)
linkfig(joinpath(path, "L1error.png"), "L₁ error Vs Nₑ")


# L₂ error Vs number of elements
Plots.png(
    Plots.plot(
        collect(horz_ne_seq),
        L2err,
        yscale = :log10,
        xlabel = "Nₑ",
        ylabel = "log₁₀(L₂ err)",
        label = "",
    ),
    joinpath(path, "L2error.png"),
)
linkfig(joinpath(path, "L2error.png"), "L₂ error Vs Nₑ")

# L∞ error Vs number of elements
Plots.png(
    Plots.plot(
        collect(horz_ne_seq),
        Linferr,
        yscale = :log10,
        xlabel = "Nₑ",
        ylabel = "log₁₀(L∞ err)",
        label = "",
    ),
    joinpath(path, "Linferror.png"),
)
linkfig(joinpath(path, "Linferror.png"), "L∞ error Vs Nₑ")
