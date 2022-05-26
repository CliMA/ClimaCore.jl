using LinearAlgebra

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    Limiters,
    slab
import ClimaCore.Geometry: ⊗
using OrdinaryDiffEq: ODEProblem, solve
using DiffEqBase
using ClimaTimeSteppers

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
    FT = Float64;
    xlim = (-2π, 2π),
    ylim = (-2π, 2π),
    zlim = (0, 4π),
    xelems = 16,
    yelems = 16,
    zelems = 16,
    Nij = 2,
)

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
    horztopology = Topologies.Topology2D(horzmesh)

    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(zdomain, nelems = zelems)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    quad = Spaces.Quadratures.GLL{Nij}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (horzspace, hv_center_space, hv_face_space)
end

# Advection problem on a 3D Cartesian domain with bounds-preserving quasimonotone horizontal limiter.
# The initial condition can be set via a command line argument.
# Possible test cases are: cosine_bells (default), gaussian_bells, and slotted_spheres

FT = Float64

# Set up physical parameters
const xmin = -2π              # domain x lower bound
const xmax = 2π               # domain x upper bound
const ymin = -2π              # domain y lower bound
const ymax = 2π               # domain y upper bound
const zmin = 0                # domain z lower bound
const zmax = 4π               # domain z upper bound
const ρ₀ = 1.0                # air density
const D₄ = 0.0                # hyperdiffusion coefficient
const u0 = π / 2              # angular velocity
const r0 = (xmax - xmin) / 6  # bells radius
const end_time = 2π           # simulation period in seconds
const dt = end_time / 800
const n_steps = Int(round(end_time / dt))
const flow_center = Geometry.XYZPoint(
    xmin + (xmax - xmin) / 2,
    ymin + (ymax - ymin) / 2,
    zmin + (zmax - zmin) / 2,
)
const bell_centers = [
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
]
const zelems = 8

# Set up test parameters
const test_name = get(ARGS, 1, "cosine_bells") # default test case to run
const cosine_test_name = "cosine_bells"
const gaussian_test_name = "gaussian_bells"
const cylinder_test_name = "slotted_spheres"
const lim_flag = true

# Plot variables and auxiliary function
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()
dirname = "box_advection_limiter_$(test_name)"

if lim_flag == false
    dirname = "$(dirname)_no_lim"
end
if D₄ == 0
    dirname = "$(dirname)_D0"
end

path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

# Set up spatial discretization
horz_ne_seq = 2 .^ (2, 3, 4, 5)
Δh = zeros(FT, length(horz_ne_seq))
L1err, L2err, Linferr = zeros(FT, length(horz_ne_seq)),
zeros(FT, length(horz_ne_seq)),
zeros(FT, length(horz_ne_seq))
Nij = 3

# h-refinement study loop
for (k, horz_ne) in enumerate(horz_ne_seq)
    # Set up 3D spatial domain - doubly periodic box
    horzspace, hv_center_space, hv_face_space = hvspace_3D(
        FT,
        Nij = Nij,
        xelems = horz_ne,
        yelems = horz_ne,
        zelems = zelems,
    )

    center_coords = Fields.coordinate_field(hv_center_space)
    face_coords = Fields.coordinate_field(hv_face_space)
    Δh[k] = (xmax - xmin) / horz_ne

    # Initialize state
    y0 = map(center_coords) do coord
        x, y, z = coord.x, coord.y, coord.z

        rd = Vector{Float64}(undef, 2)
        for i in 1:2
            rd[i] = Geometry.euclidean_distance(coord, bell_centers[i])
        end

        # Initialize specific tracer concentration
        if test_name == cylinder_test_name
            if rd[1] <= r0 && abs(x - bell_centers[1].x) >= r0 / 6
                q = 1.0
            elseif rd[2] <= r0 && abs(x - bell_centers[2].x) >= r0 / 6
                q = 1.0
            elseif rd[1] <= r0 &&
                   abs(x - bell_centers[1].x) < r0 / 6 &&
                   (y - bell_centers[1].y) < -5 * r0 / 12
                q = 1.0
            elseif rd[2] <= r0 &&
                   abs(x - bell_centers[2].x) < r0 / 6 &&
                   (y - bell_centers[2].y) > 5 * r0 / 12
                q = 1.0
            else
                q = 0.1
            end
        elseif test_name == gaussian_test_name
            q = 0.95 * (exp(-5.0 * (rd[1] / r0)^2) + exp(-5.0 * (rd[2] / r0)^2))
        else # default test case, cosine bells
            if rd[1] < r0
                q = 0.1 + 0.9 * (1 / 2) * (1 + cospi(rd[1] / r0))
            elseif rd[2] < r0
                q = 0.1 + 0.9 * (1 / 2) * (1 + cospi(rd[2] / r0))
            else
                q = 0.1
            end
        end

        # Initialize air density
        ρ = ρ₀

        # Tracer density
        Q = ρ * q
        return (ρ = ρ, ρq = Q)
    end

    y0 = Fields.FieldVector(ρ = y0.ρ, ρq = y0.ρq)

    function f!(dy, y, parameters, t, alpha, beta)

        end_time = parameters.end_time
        center_coords = parameters.center_coords
        face_coords = parameters.face_coords
        zf = face_coords.z
        xc, yc = center_coords.x, center_coords.y

        # Define the flow
        uu = @. -u0 * (yc - flow_center.y) * cospi(t / end_time)
        uv = @. u0 * (xc - flow_center.x) * cospi(t / end_time)
        uw = @. u0 * sinpi(zf / zmax) * cospi(t / end_time)

        cuₕ = Geometry.Covariant12Vector.(Geometry.UVVector.(uu, uv))
        fw = Geometry.Covariant3Vector.(Geometry.WVector.(uw))

        # Set up operators
        # Spectral horizontal operators
        hgrad = Operators.Gradient()
        hdiv = Operators.Divergence()
        hwdiv = Operators.WeakDivergence()
        # Vertical staggered FD operators
        first_order_Ic2f = Operators.InterpolateC2F(
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        first_order_If2c = Operators.InterpolateF2C()
        vdivf2c = Operators.DivergenceF2C(
            top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
            bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
        )
        third_order_upwind_c2f = Operators.Upwind3rdOrderBiasedProductC2F(
            bottom = Operators.ThirdOrderOneSided(),
            top = Operators.ThirdOrderOneSided(),
        )

        fuₕ = first_order_Ic2f.(cuₕ)
        fuvw =
            Geometry.Contravariant123Vector.(fuₕ) .+
            Geometry.Contravariant123Vector.(fw)
        vert_flux_wρ = vdivf2c.(fw .* first_order_Ic2f.(y.ρ))
        vert_flux_wρq =
            vdivf2c.(
                first_order_Ic2f.(y.ρ) .*
                third_order_upwind_c2f.(fuvw, y.ρq ./ y.ρ),
            )

        # Compute vertical velocity by interpolating faces to centers
        cw = first_order_If2c.(fw)        # Covariant3Vector on faces, interpolated to centers
        cuvw =
            Geometry.Covariant123Vector.(cuₕ) .+
            Geometry.Covariant123Vector.(cw)

        ## NOTE: With limiters, the order of the operations 1)-5) below matters!

        # 1) Horizontal Transport:
        # 1.1) Horizontal part of continuity equation:
        @. dy.ρ = beta * dy.ρ - alpha * hwdiv(y.ρ * cuvw)

        # 1.2) Horizontal advection of tracer equations:
        @. dy.ρq = beta * dy.ρq - alpha * hwdiv(y.ρq * cuvw) + alpha * ystar.ρq

        # 2) Apply the limiters:
        if lim_flag
            # First compute the min_q/max_q at the current time step
            Limiters.compute_bounds!(parameters.limiter, y.ρq, y.ρ)
            # Then apply the limiters
            Limiters.apply_limiter!(dy.ρq, dy.ρ, parameters.limiter)
        end

        # 3) Add Hyperdiffusion to the horizontal tracers equation (using weak div)
        # Set up working variable needed for hyperdiffusion
        ystar = similar(y)

        # Compute hyperviscosity for the tracer equation by splitting it in two diffusion calls
        @. ystar.ρq = hwdiv(hgrad(y.ρq / y.ρ))
        Spaces.weighted_dss!(ystar.ρq)
        @. ystar.ρq = -D₄ * hwdiv(y.ρ * hgrad(ystar.ρq))

        # 4) Vertical Transport:
        # 4.1) Vertical part of continuity equation
        @. dy.ρ -= alpha * vert_flux_wρ

        # 4.2) Vertical advection of tracer equation
        @. dy.ρ -= alpha * vdivf2c.(first_order_Ic2f.(y.ρ .* cuₕ))

        # 5) DSS:
        Spaces.weighted_dss!(dy.ρ)
        Spaces.weighted_dss!(dy.ρq)

        return dy
    end

    # Set up RHS function
    ystar = copy(y0)
    parameters = (
        horzspace = horzspace,
        limiter = Limiters.QuasiMonotoneLimiter(y0.ρq, y0.ρ),
        end_time = end_time,
        center_coords = center_coords,
        face_coords = face_coords,
    )
    f!(ystar, y0, parameters, 0.0, dt, 1)

    # Solve the ODE
    prob = ODEProblem(
        IncrementingODEFunction(f!),
        copy(y0),
        (0.0, end_time),
        parameters,
    )
    sol = solve(
        prob,
        SSPRK33ShuOsher(),
        dt = dt,
        saveat = 0.99 * 80 * dt,
        progress = true,
        adaptive = false,
        progress_message = (dt, u, p, t) -> t,
    )
    L1err[k] = norm(
        (sol.u[end].ρq ./ sol.u[end].ρ .- y0.ρq ./ y0.ρ) ./ (y0.ρq ./ y0.ρ),
        1,
    )
    L2err[k] = norm(
        (sol.u[end].ρq ./ sol.u[end].ρ .- y0.ρq ./ y0.ρ) ./ (y0.ρq ./ y0.ρ),
    )
    Linferr[k] = norm(
        (sol.u[end].ρq ./ sol.u[end].ρ .- y0.ρq ./ y0.ρ) ./ (y0.ρq ./ y0.ρ),
        Inf,
    )

    @info "Test case: $(test_name)"
    @info "With limiter: $(lim_flag)"
    @info "Hyperdiffusion coefficient: D₄ = $(D₄)"
    @info "Number of elements in XYZ domain: $(horz_ne) x $(horz_ne) x $(zelems)"
    @info "Number of quadrature points per horizontal element: $(Nij) x $(Nij) (p = $(Nij-1))"
    @info "Time step dt = $(dt) (s)"
    @info "Tracer concentration norm at t = 0 (s): ", norm(y0.ρq ./ y0.ρ)
    @info "Tracer concentration norm at $(n_steps) time steps, t = $(end_time) (s): ",
    norm(sol.u[end].ρq ./ sol.u[end].ρ)
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
linkfig(
    relpath(joinpath(path, "L1error.png"), joinpath(@__DIR__, "../../..")),
    "L₁ error Vs Nₑ",
)


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
linkfig(
    relpath(joinpath(path, "L2error.png"), joinpath(@__DIR__, "../../..")),
    "L₂ error Vs Nₑ",
)

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
linkfig(
    relpath(joinpath(path, "Linferror.png"), joinpath(@__DIR__, "../../..")),
    "L∞ error Vs Nₑ",
)
