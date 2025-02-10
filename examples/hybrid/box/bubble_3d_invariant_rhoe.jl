# Usage:
# julia --project hybrid/box/bubble_3d_invariant_rhoe.jl
# runs the low resolution simulation with Float64
# julia --project hybrid/box/bubble_3d_invariant_rhoe.jl Float64 medium
# rung the medium resolution simulation with Float64
# julia --project hybrid/box/bubble_3d_invariant_rhoe.jl Float64 high
# rung the high resolution simulation with Float64
# julia --project hybrid/box/bubble_3d_invariant_rhoe.jl Float64 custom lxy lz xyelem zelem npoly Δt t_int
# runs the simulation with user-defined resolution

using Test
using Adapt
using ClimaComms
ClimaComms.@import_required_backends
FloatType = eval(Meta.parse(get(ARGS, 1, "Float64")))
using StaticArrays, IntervalSets, LinearAlgebra, SciMLBase
using OrdinaryDiffEqSSPRK: SSPRK33
using DocStringExtensions

import ClimaCore:
    ClimaCore,
    DataLayouts,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    Operators
using ClimaCorePlots, Plots
using ClimaCore.Geometry
using ClimaCore.Spaces.Quadratures
using Logging

using CUDA
CUDA.allowscalar(false)

"""
    SimulationParameters{FT}

Parameters needed for the simulation.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SimulationParameters{FT} # rename to PhysicalParameters
    "Domain length in x and y directions. Here, lx = ly = lxy"
    lxy::FT
    "Domain length in z direction"
    lz::FT
    "Number of elements in x and y directions"
    xyelem::Int
    "Number of elements in z direction"
    zelem::Int
    "Polynomial order"
    npoly::Int
    "time step"
    Δt::FT
    "Integration time (sec)"
    t_int::FT # integration time
end

function SimulationParameters(::Type{FT}, resolution, args...) where {FT}
    @assert resolution ∈ ("low", "medium", "high", "custom")
    domain_extents = (FT(1000), FT(1000))
    if resolution == "high"
        return SimulationParameters{FT}(
            domain_extents...,
            16,
            32,
            3,
            FT(0.01),
            FT(700),
        )
    elseif resolution == "medium"
        return SimulationParameters{FT}(
            domain_extents...,
            8,
            16,
            3,
            FT(0.025),
            FT(700),
        )
    elseif resolution == "custom"
        @assert length(args) == 7 "provide lxy, lz, xyelem, zelem, npoly, Δt and t_int for the custom simulation"
        return SimulationParameters{FT}(args...)
    else # low resolution
        return SimulationParameters{FT}(
            domain_extents...,
            4,
            16,
            3,
            FT(0.05),
            FT(700),
        )
    end
end

"""
    PhysicalParameters{FT}

Physical parameters needed for the simulation.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PhysicalParameters{FT} # rename to PhysicalParameters
    "Mean sea level pressure"
    MSLP::FT = FT(1e5)
    "Gravitational constant"
    grav::FT = FT(9.8)
    "R dry (gas constant / mol mass dry air)"
    R_d::FT = FT(287.058)
    "Heat capacity ratio"
    γ::FT = FT(1.4)
    "Heat capacity at constant pressure"
    C_p::FT = FT(R_d * γ / (γ - 1))
    "Heat capacity at constant volume"
    C_v::FT = FT(R_d / (γ - 1))
    "Triple point temperature"
    T_0::FT = FT(273.16)
end
Adapt.@adapt_structure PhysicalParameters

function hvspace_3D(
    sim_parameters::SimulationParameters{FT},
    comms_ctx,
) where {FT}
    (; lxy, lz, xyelem, zelem, npoly) = sim_parameters
    xlim = (-lxy / 2, lxy / 2)
    ylim = (-lxy / 2, lxy / 2)
    zlim = (0.0, lz)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    verttopo = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(comms_ctx.device),
        vertmesh,
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopo)

    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(ylim[1]) .. Geometry.YPoint{FT}(ylim[2]),
        x1periodic = true,
        x2periodic = true,
    )
    Nv = Meshes.nelements(vertmesh)
    Nf_center, Nf_face = 2, 1 #1 + 3 + 1
    quad = Quadratures.GLL{npoly + 1}()
    horzmesh = Meshes.RectilinearMesh(horzdomain, xyelem, xyelem)
    horztopology = Topologies.Topology2D(comms_ctx, horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space, horztopology)
end

@inline Φ(z, grav) = grav * z

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
function init_dry_rising_bubble_3d(x, y, z, params)
    (; C_p, C_v, MSLP, grav, R_d, T_0) = params
    x_c = 0.0
    y_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0
    θ_c = 0.5
    p_0 = MSLP

    # auxiliary quantities
    r = sqrt((x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation

    θ = θ_b + θ_p # potential temperature
    π_exn = 1.0 - grav * z / C_p / θ # exner function
    T = π_exn * θ # temperature
    p = p_0 * π_exn^(C_p / R_d) # pressure
    ρ = p / R_d / T # density
    e = C_v * (T - T_0) + grav * z
    ρe = ρ * e # total energy

    return (ρ = ρ, ρe = ρe)
end
# Reference: https://doi.org/10.5194/gmd-9-2007-2016 , equation (77).
function compute_κ₄(sim_params::SimulationParameters{FT}) where {FT}
    (; lxy, xyelem, npoly) = sim_params
    quad_points, _ =
        Quadratures.quadrature_points(FT, Quadratures.GLL{npoly + 1}())
    Δx = (lxy / xyelem) * diff(quad_points)[1] / 2
    κ₄ = 1.0e6 * (Δx / lxy)^3.2
    return κ₄
end

function rhs_invariant!(dY, Y, ghost_buffer, t)
    (; C_p, C_v, MSLP, grav, R_d, T_0) = ghost_buffer.params
    (; z, κ₄, cω³, fω¹², fu¹², fu³, cuvw, cE, ce, cI, cT, cp, ch_tot) =
        ghost_buffer
    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe = Y.Yc.ρe

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe

    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.WeakCurl()

    dρ .= 0 .* cρ

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    # fuₕ = Ic2f.(cuₕ)
    cuvw .=
        Geometry.Covariant123Vector.(cuₕ) .+
        Geometry.Covariant123Vector.(If2c.(fw))

    ce .= @. cρe / cρ
    cI .= @. ce - Φ(z, grav) - (norm(cuvw)^2) / 2
    cT .= @. cI / C_v + T_0
    cp .= @. cρ * R_d * cT

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients
    ch_tot = @. ce + cp / cρ
    χe = @. dρe = hwdiv(hgrad(ch_tot)) # we store χe in dρe
    χuₕ = @. duₕ =
        hwgrad(hdiv(cuₕ)) - Geometry.Covariant12Vector(
            hwcurl(Geometry.Covariant3Vector(hcurl(cuₕ))),
        )
    Spaces.weighted_dss_start!(dρe, ghost_buffer.dρe)
    Spaces.weighted_dss_start!(duₕ, ghost_buffer.duₕ)
    Spaces.weighted_dss_internal!(dρe, ghost_buffer.dρe)
    Spaces.weighted_dss_internal!(duₕ, ghost_buffer.duₕ)
    Spaces.weighted_dss_ghost!(dρe, ghost_buffer.dρe)
    Spaces.weighted_dss_ghost!(duₕ, ghost_buffer.duₕ)

    @. dρe = -κ₄ * hwdiv(cρ * hgrad(χe))
    @. duₕ =
        -κ₄ * (
            hwgrad(hdiv(χuₕ)) - Geometry.Covariant12Vector(
                hwcurl(Geometry.Covariant3Vector(hcurl(χuₕ))),
            )
        )

    # 1) Mass conservation

    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuvw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    # curl term
    hcurl = Operators.Curl()
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
    )
    cω³ .= hcurl.(cuₕ) # Contravariant3Vector
    fω¹² .= hcurl.(fw) # Contravariant12Vector
    fω¹² .+= vcurlc2f.(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu¹² .=
        Geometry.Contravariant12Vector.(
            Geometry.Covariant123Vector.(Ic2f.(cuₕ)),
        ) # Contravariant12Vector in 3D
    fu³ .= Geometry.Contravariant3Vector.(Geometry.Covariant123Vector.(fw))
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹² × fu³)

    # Needed for 3D:
    @. duₕ -=
        cω³ × Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))


    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE .= @. (norm(cuvw)^2) / 2 + Φ(z, grav)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) potential temperature

    @. dρe -= hdiv(cuvw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    fcc = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    @. dρ += fcc(fw, cρ)
    @. dρe += fcc(fw, cρe)
    # dYc.ρuₕ += fcc(w, Yc.ρuₕ)

    Spaces.weighted_dss_start!(dY.Yc, ghost_buffer.Yc)
    Spaces.weighted_dss_start!(dY.uₕ, ghost_buffer.uₕ)
    Spaces.weighted_dss_start!(dY.w, ghost_buffer.w)
    Spaces.weighted_dss_internal!(dY.Yc, ghost_buffer.Yc)
    Spaces.weighted_dss_internal!(dY.uₕ, ghost_buffer.uₕ)
    Spaces.weighted_dss_internal!(dY.w, ghost_buffer.w)
    Spaces.weighted_dss_ghost!(dY.Yc, ghost_buffer.Yc)
    Spaces.weighted_dss_ghost!(dY.uₕ, ghost_buffer.uₕ)
    Spaces.weighted_dss_ghost!(dY.w, ghost_buffer.w)
    return dY
end

function bubble_3d_invariant_ρe(ARGS, comms_ctx, ::Type{FT}) where {FT}

    params = PhysicalParameters{FT}()
    resolution = get(ARGS, 2, "low")
    if resolution == "custom"
        args = (
            parse(FT, get(ARGS, 3, "1000")),
            parse(FT, get(ARGS, 4, "1000")),
            parse(Int, get(ARGS, 5, "4")),
            parse(Int, get(ARGS, 6, "16")),
            parse(Int, get(ARGS, 7, "3")),
            parse(FT, get(ARGS, 8, "0.05")),
            parse(FT, get(ARGS, 9, "700.0")),
        )
    else
        args = ()
    end

    logger_stream = ClimaComms.iamroot(comms_ctx) ? stderr : devnull
    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))

    @info "Context information" device = comms_ctx.device context = comms_ctx nprocs =
        ClimaComms.nprocs(comms_ctx) Float_type = FT resolution = resolution

    sim_params = SimulationParameters(FT, resolution, args...)
    (; lxy, lz) = sim_params
    # set up 3D domain - doubly periodic box
    hv_center_space, hv_face_space, horztopology =
        hvspace_3D(sim_params, comms_ctx)

    # initial conditions
    coords = Fields.coordinate_field(hv_center_space)
    face_coords = Fields.coordinate_field(hv_face_space)

    Yc = map(
        coord ->
            init_dry_rising_bubble_3d(coord.x, coord.y, coord.z, params),
        coords,
    )
    uₕ = map(_ -> Geometry.Covariant12Vector(0.0, 0.0), coords)
    w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
    Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

    energy_0 = sum(Y.Yc.ρe)
    mass_0 = sum(Y.Yc.ρ)
    κ₄ = compute_κ₄(sim_params)

    @info "Initial condition" energy_0 mass_0 κ₄
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    cuvw =
        Geometry.Covariant123Vector.(Y.uₕ) .+
        Geometry.Covariant123Vector.(If2c.(Y.w))
    cE = @. (norm(cuvw)^2) / 2 + Φ(coords.z, params.grav)
    ce = @. Y.Yc.ρe / Y.Yc.ρ
    cI = @. ce - Φ(coords.z, params.grav) - (norm(cuvw)^2) / 2
    cT = @. cI / params.C_v + params.T_0
    cp = @. Y.Yc.ρ * params.R_d * cT
    ch_tot = @. ce + cp / Y.Yc.ρ
    ghost_buffer = (
        dρe = Spaces.create_dss_buffer(Yc.ρe),
        duₕ = Spaces.create_dss_buffer(Y.uₕ),
        uₕ = Spaces.create_dss_buffer(Y.uₕ),
        Yc = Spaces.create_dss_buffer(Y.Yc),
        w = Spaces.create_dss_buffer(Y.w),
        params = params,
        z = coords.z,
        κ₄ = compute_κ₄(sim_params),
        fω¹² = Operators.Curl().(Y.w),
        cω³ = Operators.Curl().(Y.uₕ),
        fu¹² = Geometry.Contravariant12Vector.(
            Geometry.Covariant123Vector.(Ic2f.(Y.uₕ))
        ),
        fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant123Vector.(Y.w)),
        cuvw = cuvw,
        cE = cE,
        ce = ce,
        cI = cI,
        cT = cT,
        cp = cp,
        ch_tot = ch_tot,
    )

    dYdt = similar(Y)
    rhs_invariant!(dYdt, Y, ghost_buffer, 0.0)
    # run!
    Δt = sim_params.Δt
    prob = ODEProblem(rhs_invariant!, Y, (0.0, sim_params.t_int), ghost_buffer)
    integrator = SciMLBase.init(
        prob,
        SSPRK33(),
        dt = Δt,
        saveat = [0.0:10.0:(sim_params.t_int)..., sim_params.t_int],
        progress = true,
        progress_message = (dt, u, p, t) -> t,
        internalnorm = (u, t) -> norm(u),
    )

    if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
        throw(:exit_profile)
    end

    t_diff = @elapsed sol_invariant = SciMLBase.solve!(integrator)

    if ClimaComms.iamroot(comms_ctx)
        println("Walltime = $t_diff seconds")
    end

    Es = FT[]
    Mass = FT[]
    for sol_step in sol_invariant.u
        Es_step = sum(sol_step.Yc.ρe)
        Mass_step = sum(sol_step.Yc.ρ)
        if ClimaComms.iamroot(comms_ctx)
            push!(Es, Es_step)
            push!(Mass, Mass_step)
        end
    end

    @info "summary" Es[end] Mass[end]
    #-----------------------------------

    ENV["GKSwstype"] = "nul"
    Plots.GRBackend()

    dir =
        comms_ctx.device isa ClimaComms.AbstractCPUDevice ?
        "bubble_3d_invariant_rhoe" : "gpu_bubble_3d_invariant_rhoe"
    path = joinpath(@__DIR__, "output", dir)
    mkpath(path)

    if ClimaComms.iamroot(comms_ctx)
        # post-processing
        Plots.png(
            Plots.plot((Es .- energy_0) ./ energy_0),
            joinpath(path, "energy_" * resolution * "_res.png"),
        )
        Plots.png(
            Plots.plot((Mass .- mass_0) ./ mass_0),
            joinpath(path, "mass_" * resolution * "_res.png"),
        )

        function linkfig(figpath, alt = "")
            # buildkite-agent upload figpath
            # link figure in logs if we are running on CI
            if get(ENV, "BUILDKITE", "") == "true"
                artifact_url = "artifact://$figpath"
                print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
            end
        end

        linkfig(
            relpath(
                joinpath(path, "energy_" * resolution * "_resolution.png"),
                joinpath(@__DIR__, "../.."),
            ),
            "Total Energy",
        )
        linkfig(
            relpath(
                joinpath(path, "mass_" * resolution * "_resolution.png"),
                joinpath(@__DIR__, "../.."),
            ),
            "Mass",
        )
        if comms_ctx isa ClimaComms.SingletonCommsContext
            cpu_hv_center_space, cpu_hv_face_space, cpu_horztopology =
                hvspace_3D(
                    sim_params,
                    ClimaComms.SingletonCommsContext(
                        ClimaComms.CPUSingleThreaded(),
                    ),
                )

            plotfield = ones(cpu_hv_center_space)

            copyto!(parent(plotfield), parent(sol_invariant.u[30].Yc.ρe))

            png(
                plot(plotfield, slice = (:, 0.0, :)),
                joinpath(path, "slice_" * resolution * "_res.png"),
            )
        end
    end
    return sol_invariant
end

comms_ctx = ClimaComms.context()
ClimaComms.init(comms_ctx)
sol_invariant = bubble_3d_invariant_ρe(ARGS, comms_ctx, FloatType)
