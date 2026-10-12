# Stratified flow over terrain: the model behind the interactive demo.
#
# A dry, compressible, nonhydrostatic atmosphere on a 2D x–z slice, discretized
# on ClimaCore's hybrid space (spectral elements in x, finite differences in z)
# with a terrain-following coordinate. The dynamics, diffusion, sponge and
# free-slip surface constraint are those of
# `examples/hybrid/plane/schar_mountain.jl`; this file wraps them so that the
# terrain, the wind, the stratification and the diffusivities are inputs, the
# state can be rebuilt on a new terrain, and the integrator can be advanced a
# few steps at a time from a GUI loop.
#
# Nothing here depends on Makie: `run_offline.jl` drives it headlessly and
# `gui.jl` drives it interactively.
module TerrainFlow

using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    ClimaCore,
    Domains,
    Fields,
    Geometry,
    Hypsography,
    Meshes,
    Operators,
    Quadratures,
    Remapping,
    Spaces,
    Topologies
import ClimaCore.Utilities: half
import ClimaTimeSteppers as CTS

# ============================================================================
# Physical constants
# ============================================================================

const MSLP = 1e5              # mean sea level pressure (Pa)
const grav = 9.8              # gravitational acceleration (m/s²)
const R_d = 287.058           # gas constant for dry air (J/kg/K)
const γ = 1.4                 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1)     # heat capacity at constant volume
const T_0 = 273.16            # triple point temperature (K)
const θ₀ = 280.0              # surface potential temperature (K)
const c_sound = 340.0         # sound speed bound used for the time step (m/s)

geopotential(z) = grav * z

# ============================================================================
# Configuration
# ============================================================================

"""
    GridConfig(; Lx, Lz, nx, nz, np)

Domain and resolution. Changing any of these rebuilds the spaces.

  - `Lx`, `Lz`: domain width (periodic in x) and height (m).
  - `nx`, `nz`: number of spectral elements in x and finite-difference levels in z.
  - `np`: polynomial degree of the elements (`np + 1` nodes per element).
"""
Base.@kwdef struct GridConfig
    Lx::Float64 = 40e3
    Lz::Float64 = 15e3
    nx::Int = 30
    nz::Int = 40
    np::Int = 3
end

# Measured single-threaded on one CPU core (2026-10-09): `coarse` costs about
# 11 ms per 0.55 s step, so the model runs ~50x faster than real time; `fine`
# about 30 ms per 0.37 s step, ~12x real time.
const GRID_PRESETS = (
    coarse = GridConfig(nx = 30, nz = 40),
    fine = GridConfig(nx = 40, nz = 60),
    finest = GridConfig(nx = 60, nz = 80, np = 4),
)

"""
Nominal horizontal node spacing (m).
"""
Δx(cfg::GridConfig) = cfg.Lx / cfg.nx / cfg.np
"""
Vertical level spacing (m).
"""
Δz(cfg::GridConfig) = cfg.Lz / cfg.nz

"""
    Params(; u₀, N, κ₂, κ₄, sponge_α, z_sponge, x_sponge)

Physical parameters. `u₀` (background wind, m/s) and `N` (Brunt–Väisälä
frequency, 1/s) set the initial state, so changing them takes effect on the
next reset; the diffusivities and sponge strength are read by the tendency on
every step and can be changed live.

The default diffusivities are the values of the Schär example (`κ₂ = 75 m²/s`,
`κ₄ = 2e7 m⁴/s` at a 940 m node spacing) scaled to the grid so that the
damping rate at the grid scale stays the same: `κ₂ ∝ Δx²`, `κ₄ ∝ Δx⁴`.
"""
Base.@kwdef mutable struct Params
    u₀::Float64 = 10.0
    N::Float64 = 0.01
    κ₂::Float64 = 75.0
    κ₄::Float64 = 2e7
    sponge_α::Float64 = 0.5    # peak sponge relaxation rate (1/s)
    z_sponge::Float64 = 0.6    # top sponge starts at this fraction of Lz
    x_sponge::Float64 = 0.15   # lateral sponges span this fraction of Lx at each edge
end

function default_params(cfg::GridConfig; kwargs...)
    scale = Δx(cfg) / 940.0
    return Params(; κ₂ = 75.0 * scale^2, κ₄ = 2e7 * scale^4, kwargs...)
end

"""
    froude_parameter(p::Params, h)

The nondimensional mountain height `N h / u₀` that selects the flow regime:
below about 0.5 linear mountain waves, around 1 wave breaking, above about 1.5
upstream blocking.
"""
froude_parameter(p::Params, h) = p.N * h / p.u₀

# ============================================================================
# Terrain
# ============================================================================
#
# A terrain is any callable `h(x)` returning the surface height (m) at
# horizontal position `x` (m). Presets are closures; a drawn profile is a
# `SampledTerrain`, which linearly interpolates a profile sampled on a uniform
# grid over `[-Lx/2, Lx/2]`.

flat_terrain() = x -> 0.0

"""
Witch of Agnesi hill: height `h₀`, half-width `a`, centred at `x₀`.
"""
agnesi_terrain(; h₀ = 1000.0, a = 2000.0, x₀ = 0.0) =
    x -> h₀ / (1 + ((x - x₀) / a)^2)

"""
Schär ridge: a Gaussian envelope of half-width `a` modulating cos² of wavelength `λ`.
"""
schar_terrain(; h₀ = 250.0, a = 5000.0, λ = 4000.0, x₀ = 0.0) =
    x -> begin
        ξ = x - x₀
        abs(ξ) <= a ? h₀ * exp(-(ξ / a)^2) * cospi(ξ / λ)^2 : 0.0
    end

"""
Flat-topped plateau of height `h₀`, half-width `a`, with `w`-wide cos² ramps.
"""
plateau_terrain(; h₀ = 800.0, a = 3000.0, w = 1500.0, x₀ = 0.0) =
    x -> begin
        ξ = abs(x - x₀)
        ξ <= a - w ? h₀ :
        ξ >= a ? 0.0 : h₀ * cospi((ξ - (a - w)) / w / 2)^2
    end

"""
Two Agnesi hills separated by `d`.
"""
twin_terrain(; h₀ = 800.0, a = 1500.0, d = 8000.0) =
    x -> h₀ / (1 + ((x - d / 2) / a)^2) + h₀ / (1 + ((x + d / 2) / a)^2)

const TERRAIN_PRESETS = (
    flat = flat_terrain,
    agnesi = agnesi_terrain,
    schar = schar_terrain,
    plateau = plateau_terrain,
    twin = twin_terrain,
)

"""
    SampledTerrain(xs, hs)

Surface height sampled at the uniformly spaced positions `xs`; evaluated by
linear interpolation and clamped to the end values outside `xs`. This is the
form a profile drawn in the GUI takes.
"""
struct SampledTerrain{V <: AbstractVector{Float64}}
    xs::V
    hs::Vector{Float64}
    function SampledTerrain(xs::AbstractVector, hs::AbstractVector)
        length(xs) == length(hs) ||
            throw(ArgumentError("xs and hs must have the same length"))
        new{typeof(xs)}(xs, Vector{Float64}(hs))
    end
end

SampledTerrain(cfg::GridConfig, h; n = 4 * cfg.nx * cfg.np) =
    (xs = range(-cfg.Lx / 2, cfg.Lx / 2; length = n); SampledTerrain(xs, h.(xs)))

function (t::SampledTerrain)(x)
    xs, hs = t.xs, t.hs
    x <= first(xs) && return first(hs)
    x >= last(xs) && return last(hs)
    dx = step(xs)
    i = clamp(floor(Int, (x - first(xs)) / dx) + 1, 1, length(xs) - 1)
    α = (x - xs[i]) / dx
    return (1 - α) * hs[i] + α * hs[i + 1]
end

"""
    prepare_terrain(cfg, terrain; max_slope = 0.5, smoothing = 2)

Sample `terrain` on a uniform grid over the domain, limit its slope to
`max_slope` (keeping peak heights where possible) and apply `smoothing`
passes of a 3-point filter. The terrain-following coordinate degrades with
steep or jagged surfaces, so every terrain, drawn or preset, goes through
this before the spaces are built.
"""
function prepare_terrain(cfg::GridConfig, terrain; max_slope = 0.5, smoothing = 2)
    n = 4 * cfg.nx * cfg.np
    xs = range(-cfg.Lx / 2, cfg.Lx / 2; length = n)
    hs = max.(Float64.(terrain.(xs)), 0.0)
    dx = step(xs)
    # Slope limiter: sweep both ways so no neighbour differs by more than
    # `max_slope * dx`. Lowers the steep side rather than raising the valley.
    for _ in 1:2
        for i in 2:n
            hs[i] = min(hs[i], hs[i - 1] + max_slope * dx)
        end
        for i in (n - 1):-1:1
            hs[i] = min(hs[i], hs[i + 1] + max_slope * dx)
        end
    end
    for _ in 1:smoothing
        hs .= [
            (hs[max(i - 1, 1)] + 2hs[i] + hs[min(i + 1, n)]) / 4 for i in 1:n
        ]
    end
    return SampledTerrain(xs, hs)
end

# ============================================================================
# Spaces
# ============================================================================

"""
    build_spaces(cfg::GridConfig, terrain; smooth = true, context)

Center and face extruded spaces on a terrain-following mesh whose bottom
surface is `terrain(x)`. With `smooth = true` the sampled surface is diffused
before the mesh is warped, which keeps sharp drawn profiles from producing an
overly distorted coordinate.
"""
function build_spaces(
    cfg::GridConfig,
    terrain;
    smooth = true,
    context = ClimaComms.context(),
)
    FT = Float64
    device = ClimaComms.device(context)

    xdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(-cfg.Lx / 2),
        Geometry.XPoint{FT}(cfg.Lx / 2),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(xdomain, nelems = cfg.nx)
    horztopology = Topologies.IntervalTopology(device, horzmesh)
    horzspace = Spaces.SpectralElementSpace1D(
        horztopology,
        Quadratures.GLL{cfg.np + 1}(),
    )

    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(cfg.Lz);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(zdomain, nelems = cfg.nz)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(device, vertmesh)

    z_surface = Geometry.ZPoint.(
        FT.(terrain.(Geometry.component.(Fields.coordinate_field(horzspace), 1))),
    )
    if smooth
        # A few diffusion steps with κ Δt / Δx² ≈ 0.1 per step.
        dt = 0.1 * Δx(cfg)^2 / 1e8
        Hypsography.diffuse_surface_elevation!(z_surface; κ = 1e8, dt, maxiter = 4)
    end
    face_space = Spaces.ExtrudedFiniteDifferenceSpace(
        horzspace,
        vert_face_space,
        Hypsography.LinearAdaption(z_surface),
    )
    center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(face_space)
    return (center_space, face_space)
end

# ============================================================================
# Initial state
# ============================================================================

"""
Hydrostatic state with uniform stratification `N`: `(ρ, ρe, ρq)` at height `z`.
"""
function stratified_state(z, p::Params)
    N, g = p.N, grav
    θ = θ₀ * exp(N^2 * z / g)
    π_exner = 1 + g^2 / N^2 / C_p / θ₀ * (exp(-N^2 * z / g) - 1)
    T = π_exner * θ
    # p = p₀ π^(cp/R) and p = ρ R_d T = ρ R_d π θ, so ρ carries π^(cp/R - 1),
    # and cp - R_d = cv.
    ρ = MSLP / (R_d * θ) * π_exner^(C_v / R_d)
    e = C_v * (T - T_0) + geopotential(z) + p.u₀^2 / 2
    return (ρ = ρ, ρe = ρ * e, ρq = ρ * dye(z))
end

"""
Passive dye in horizontal stripes, so material lines show as deformed bands.
"""
dye(z; period = 1000.0) = cospi(z / period)^2

function initial_state(center_space, face_space, p::Params)
    coords = Fields.coordinate_field(center_space)
    face_coords = Fields.coordinate_field(face_space)
    Yc = map(coord -> stratified_state(coord.z, p), coords)
    # The conversion to covariant components needs the local geometry, which
    # broadcasting over a field supplies.
    uₕ_local = map(_ -> Geometry.UWVector(p.u₀, 0.0), coords)
    uₕ = Geometry.Covariant1Vector.(uₕ_local)
    w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
    Y = Fields.FieldVector(; Yc, uₕ, w)
    Spaces.weighted_dss!(Y.Yc)
    Spaces.weighted_dss!(Y.uₕ)
    Spaces.weighted_dss!(Y.w)
    return Y
end

# ============================================================================
# Sponges
# ============================================================================

function rayleigh_sponge_z(z, cfg::GridConfig, p::Params)
    z_sponge = p.z_sponge * cfg.Lz
    z < z_sponge && return 0.0
    r = (z - z_sponge) / (cfg.Lz - z_sponge)
    return p.sponge_α * sinpi(0.5 * r)^2
end

function rayleigh_sponge_x(x, cfg::GridConfig, p::Params)
    width = p.x_sponge * cfg.Lx
    x_start = cfg.Lx / 2 - width
    abs(x) < x_start && return 0.0
    r = (abs(x) - x_start) / width
    return p.sponge_α * sinpi(0.5 * r)^2
end

sponge_profile(coord, cfg, p) =
    max(rayleigh_sponge_z(coord.z, cfg, p), rayleigh_sponge_x(coord.x, cfg, p))

# ============================================================================
# Model
# ============================================================================

"""
    Model(cfg, p, terrain; context)

Spaces, state, scratch fields and integrator for one terrain. The integrator
is a ClimaTimeSteppers SSP33 scheme advanced by [`step!`](@ref). The sponge
and the initial state are stored so the tendency and the diagnostics can refer
to them; `p.u₀` and `p.N` changes need [`reset!`](@ref).
"""
mutable struct Model{CS, FS, Y, C, P, I}
    cfg::GridConfig
    p::Params
    terrain::Any
    center_space::CS
    face_space::FS
    Y::Y
    cache::C
    params::P
    integrator::I
    dt::Float64
end

function build_cache(center_space, face_space, cfg, p, Y)
    coords = Fields.coordinate_field(center_space)
    face_coords = Fields.coordinate_field(face_space)
    u_init = copy(Y.uₕ)
    β = sponge_profile.(coords, Ref(cfg), Ref(p))
    βf = sponge_profile.(face_coords, Ref(cfg), Ref(p))
    # Background fields for anomaly diagnostics.
    θ_init = potential_temperature(Y, coords)
    p_init = pressure(Y, coords)
    return (; coords, face_coords, u_init, β, βf, θ_init, p_init)
end

"""
Acoustic time step: half the grid spacing over the sound speed.
"""
time_step(cfg::GridConfig) = 0.5 * min(Δx(cfg), Δz(cfg)) / c_sound

function Model(
    cfg::GridConfig = GridConfig(),
    p::Params = default_params(cfg),
    terrain = TERRAIN_PRESETS.agnesi();
    context = ClimaComms.context(),
)
    terrain = prepare_terrain(cfg, terrain)
    center_space, face_space = build_spaces(cfg, terrain; context)
    Y = initial_state(center_space, face_space, p)
    cache = build_cache(center_space, face_space, cfg, p, Y)
    params = (; cfg, p, cache)
    project_surface_w!(Y, face_space)
    dt = time_step(cfg)
    integrator = make_integrator(Y, params, dt)
    return Model(
        cfg,
        p,
        terrain,
        center_space,
        face_space,
        Y,
        cache,
        params,
        integrator,
        dt,
    )
end

function make_integrator(Y, params, dt)
    prob = CTS.ODEProblem(
        CTS.ClimaODEFunction(; T_exp! = tendency!, dss!),
        Y,
        (0.0, Inf),
        params,
    )
    return CTS.init(prob, CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()); dt)
end

"""
    reset!(model; terrain = model.terrain)

Return the model to the initial state on `terrain` (rebuilding the spaces if
the terrain changed) with the current `model.p`. `model.terrain` always holds
the slope-limited `SampledTerrain` actually used by the grid.
"""
function reset!(model::Model; terrain = model.terrain)
    if terrain !== model.terrain
        terrain = prepare_terrain(model.cfg, terrain)
        model.center_space, model.face_space =
            build_spaces(model.cfg, terrain)
        model.terrain = terrain
    end
    model.Y = initial_state(model.center_space, model.face_space, model.p)
    model.cache =
        build_cache(model.center_space, model.face_space, model.cfg, model.p, model.Y)
    model.params = (; cfg = model.cfg, p = model.p, cache = model.cache)
    project_surface_w!(model.Y, model.face_space)
    model.integrator = make_integrator(model.Y, model.params, model.dt)
    return model
end

"""
Advance the model by `n` time steps.
"""
function step!(model::Model, n::Integer = 1)
    for _ in 1:n
        CTS.step!(model.integrator)
    end
    return model
end

current_time(model::Model) = model.integrator.t

# ============================================================================
# Surface constraint and DSS
# ============================================================================

# The terrain follows the lower boundary, so the flow there must be tangent to
# it: contravariant u³ = g³¹u₁ + g³³u₃ = 0, that is u₃ = -g³¹u₁/g³³. This is a
# constraint on the state, applied after every stage next to the DSS.
function project_surface_w!(Y, face_space)
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    face_geometry =
        Fields.level(Fields.local_geometry_field(face_space), half)
    u₁_bc = Geometry.contravariant3.(
        Fields.level(Ic2f.(Y.uₕ), half),
        face_geometry,
    )
    g33 = face_geometry.gⁱʲ.components.data.:9
    u₃_bc = Geometry.Covariant3Vector.(-1 .* u₁_bc ./ g33)
    apply_boundary_w =
        Operators.SetBoundaryOperator(bottom = Operators.SetValue(u₃_bc))
    @. Y.w = apply_boundary_w(Y.w)
    return Y
end

_dss!(x::Fields.Field) = Spaces.weighted_dss!(x)
_dss!(::Any) = nothing
function dss!(Y, params, t)
    foreach(_dss!, Fields._values(Y))
    project_surface_w!(Y, axes(Y.w))
    return nothing
end

# ============================================================================
# Tendency
# ============================================================================

function tendency!(dY, Y, params, t)
    (; p, cache) = params
    (; coords, face_coords, u_init, β, βf) = cache

    cρ = Y.Yc.ρ   # scalar on centers
    fw = Y.w      # Covariant3Vector on faces
    cuₕ = Y.uₕ    # Covariant1Vector on centers
    cρe = Y.Yc.ρe
    cρq = Y.Yc.ρq

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    dρq = dY.Yc.ρq
    z = coords.z

    hdiv = Operators.Divergence()
    hwdiv = Operators.Divergence{Operators.WeakForm}()
    hgrad = Operators.Gradient()
    hwgrad = Operators.Gradient{Operators.WeakForm}()
    hcurl = Operators.Curl()

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    dρ .= 0 .* cρ

    cw = If2c.(fw)
    cuw = @. Geometry.Covariant13Vector(cuₕ) + Geometry.Covariant13Vector(cw)
    fuw = @. Ic2f(cuw)

    ce = @. cρe / cρ
    cq = @. cρq / cρ
    cI = @. ce - geopotential(z) - (norm(cuw)^2) / 2
    cT = @. cI / C_v + T_0
    cp = @. cρ * R_d * cT
    h_tot = @. ce + cp / cρ # total enthalpy at cell centers

    # Hyperdiffusion: two Laplacian passes with a DSS in between. The
    # intermediates are stored in the tendency fields.
    χe = @. dρe = hwdiv(hgrad(h_tot))
    χq = @. dρq = hwdiv(hgrad(cq))
    χuₕ = @. duₕ = hwgrad(hdiv(cuₕ))
    Spaces.weighted_dss!(dρe)
    Spaces.weighted_dss!(duₕ)
    Spaces.weighted_dss!(dρq)
    κ₄ = p.κ₄
    @. dρe = -κ₄ * hwdiv(cρ * hgrad(χe))
    @. dρq = 0 * hwdiv(cρ * hgrad(χq))
    @. duₕ = -κ₄ * (hwgrad(hdiv(χuₕ)))

    # Mass
    dw .= fw .* 0
    dρ .-= hwdiv.(cρ .* (cuw))
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # Momentum (vector-invariant form)
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
        top = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
    )
    fω¹ = hcurl.(fw)
    fω¹ .+= vcurlc2f.(cuₕ)

    fu¹ = @. Geometry.project(Geometry.Contravariant1Axis(), fuw)
    fu³ = @. Geometry.project(Geometry.Contravariant3Axis(), fuw)

    @. dw -= fω¹ × fu¹ # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹ × fu³)
    @. duₕ -= hgrad(cp) / cρ

    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    cE = @. (norm(cuw)^2) / 2 + geopotential(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    # Total energy and dye
    @. dρe -= hwdiv(cuw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))
    @. dρq -= hwdiv(cuw * (cρq))
    @. dρq -= vdivf2c(fw * Ic2f(cρq))
    @. dρq -= vdivf2c(Ic2f(cuₕ * (cρq)))

    # Uniform second-order diffusion
    ∂c = Operators.GradientF2C()
    fρ = @. Ic2f(cρ)
    κ₂ = p.κ₂

    ᶠ∇ᵥuₕ = @. vgradc2f(cuₕ.components.data.:1)
    ᶜ∇ᵥw = @. ∂c(fw.components.data.:1)
    ᶠ∇ᵥh_tot = @. vgradc2f(h_tot)
    ᶜ∇ₕuₕ = @. hgrad(cuₕ.components.data.:1)
    ᶠ∇ₕw = @. hgrad(fw.components.data.:1)
    ᶜ∇ₕh_tot = @. hgrad(h_tot)

    κ₂∇ᵥw = @. κ₂ * ᶜ∇ᵥw
    vκ₂∇²w = Operators.divergence_c2f_dirichlet(
        κ₂∇ᵥw;
        bottom = Geometry.WVector(0.0),
        top = Geometry.WVector(0.0),
    )

    dfw = dY.w.components.data.:1
    dcu = dY.uₕ.components.data.:1
    @. dcu += hwdiv(κ₂ * ᶜ∇ₕuₕ)
    @. dcu += vdivf2c(κ₂ * ᶠ∇ᵥuₕ)
    @. dfw += hwdiv(κ₂ * ᶠ∇ₕw)
    @. dfw += vκ₂∇²w
    @. dρe += hwdiv(cρ * κ₂ * ᶜ∇ₕh_tot)
    @. dρe += vdivf2c(fρ * κ₂ * ᶠ∇ᵥh_tot)

    # Sponges: relax the wind to its initial value, damp w.
    @. duₕ -= β * (cuₕ - u_init)
    @. dw -= βf * fw

    # `w` at the surface is fixed by the terrain constraint, so the vertical
    # momentum equation must not drive it off that constraint.
    apply_boundary_dw = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Covariant3Vector(0.0)),
    )
    @. dw = apply_boundary_dw(dw)

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)
    return dY
end

# ============================================================================
# Diagnostics (all returned as scalar fields on cell centers)
# ============================================================================

function center_velocity(Y)
    If2c = Operators.InterpolateF2C()
    return @. Geometry.Covariant13Vector(Y.uₕ) +
              Geometry.Covariant13Vector(If2c(Y.w))
end

horizontal_velocity(Y) =
    Geometry.project.(Ref(Geometry.UAxis()), center_velocity(Y)).components.data.:1
vertical_velocity(Y) =
    Geometry.project.(Ref(Geometry.WAxis()), center_velocity(Y)).components.data.:1

function temperature(Y, coords)
    cuw = center_velocity(Y)
    return @. (Y.Yc.ρe / Y.Yc.ρ - geopotential(coords.z) - norm(cuw)^2 / 2) /
              C_v + T_0
end
pressure(Y, coords) = @. Y.Yc.ρ * R_d * $(temperature(Y, coords))
potential_temperature(Y, coords) =
    @. $(temperature(Y, coords)) * (MSLP / $(pressure(Y, coords)))^(R_d / C_p)

"""
    diagnose(model, name::Symbol)

A scalar center field for plotting. `name` is one of `DIAGNOSTICS`.
"""
function diagnose(model::Model, name::Symbol)
    Y, coords = model.Y, model.cache.coords
    name === :w && return vertical_velocity(Y)
    name === :u && return horizontal_velocity(Y)
    name === :u′ && return horizontal_velocity(Y) .- model.p.u₀
    name === :θ && return potential_temperature(Y, coords)
    name === :θ′ && return potential_temperature(Y, coords) .- model.cache.θ_init
    name === :p′ && return pressure(Y, coords) .- model.cache.p_init
    name === :q && return Y.Yc.ρq ./ Y.Yc.ρ
    throw(ArgumentError("unknown diagnostic $name; expected one of $DIAGNOSTICS"))
end

const DIAGNOSTICS = (
    w = "vertical velocity (m/s)",
    u = "horizontal velocity (m/s)",
    u′ = "horizontal velocity anomaly (m/s)",
    θ = "potential temperature (K)",
    θ′ = "potential temperature anomaly (K)",
    p′ = "pressure anomaly (Pa)",
    q = "dye",
)

# ============================================================================
# Tracers: passive particles advected by the resolved wind
# ============================================================================
#
# The wind is remapped once per frame onto a regular grid in (x, reference z)
# with `Remapping.Remapper`, and each particle then samples that grid
# bilinearly. Particles live in physical (x, z); the reference height for the
# lookup follows the `LinearAdaption` mapping of the grid.

"""
    Tracers(model; nx = 32, nz = 10, z_max = 0.6 * model.cfg.Lz)

`nx × nz` particles on a regular grid up to `z_max`, lifted above the terrain,
plus the remapper and scratch used to advect them with [`advance!`](@ref).
"""
mutable struct Tracers{R, X <: AbstractRange, Z <: AbstractRange}
    x::Vector{Float64}
    z::Vector{Float64}
    remapper::R
    xs::X
    zs::Z
    uw::Array{Float64, 3}
end

function Tracers(model::Model; nx = 32, nz = 10, z_max = 0.6 * model.cfg.Lz)
    cfg = model.cfg
    # Target grid: two points per node horizontally, two per level vertically,
    # kept inside the domain so no target falls outside an element.
    xs = range(-cfg.Lx / 2, cfg.Lx / 2; length = 2 * cfg.nx * cfg.np + 1)
    xs = range(xs[1] + 1e-3, xs[end] - 1e-3; length = length(xs))
    zs = range(Δz(cfg) / 2, cfg.Lz - Δz(cfg) / 2; length = 2 * cfg.nz)
    remapper = Remapping.Remapper(
        model.center_space;
        target_hcoords = Geometry.XPoint.(collect(xs)),
        target_zcoords = Geometry.ZPoint.(collect(zs)),
        buffer_length = 2,
    )
    uw = zeros(length(xs), length(zs), 2)
    x = Float64[]
    z = Float64[]
    for xi in range(-cfg.Lx / 2, cfg.Lx / 2; length = nx + 1)[1:nx],
        zj in range(z_max / nz, z_max; length = nz)

        push!(x, xi)
        push!(z, max(zj, model.terrain(xi) + 100.0))
    end
    return Tracers(x, z, remapper, xs, zs, uw)
end

reference_z(model::Model, x, z) =
    (h = model.terrain(x); (z - h) / (1 - h / model.cfg.Lz))

"""
Bilinear sample of the remapped wind at physical `(x, z)`: returns `(u, w)`.
"""
function sample_wind(tr::Tracers, model::Model, x, z)
    xs, zs, uw = tr.xs, tr.zs, tr.uw
    zr = reference_z(model, x, z)
    fx = clamp((x - first(xs)) / step(xs), 0.0, length(xs) - 1 - 1e-9)
    fz = clamp((zr - first(zs)) / step(zs), 0.0, length(zs) - 1 - 1e-9)
    i, j = floor(Int, fx) + 1, floor(Int, fz) + 1
    a, b = fx - (i - 1), fz - (j - 1)
    lerp(k) =
        (1 - a) * (1 - b) * uw[i, j, k] +
        a * (1 - b) * uw[i + 1, j, k] +
        (1 - a) * b * uw[i, j + 1, k] +
        a * b * uw[i + 1, j + 1, k]
    return lerp(1), lerp(2)
end

"""
    advance!(tr::Tracers, model, Δt)

Move the particles through the current wind by `Δt` seconds with a midpoint
step; periodic in `x`, kept above the terrain and below the top.
"""
function advance!(tr::Tracers, model::Model, Δt)
    Remapping.interpolate!(
        tr.uw,
        tr.remapper,
        [horizontal_velocity(model.Y), vertical_velocity(model.Y)],
    )
    Lx, Lz = model.cfg.Lx, model.cfg.Lz
    wrap(x) = x - Lx * round(x / Lx)
    for i in eachindex(tr.x)
        x, z = tr.x[i], tr.z[i]
        u1, w1 = sample_wind(tr, model, x, z)
        xm, zm = wrap(x + Δt / 2 * u1), z + Δt / 2 * w1
        u2, w2 = sample_wind(tr, model, xm, zm)
        x = wrap(x + Δt * u2)
        z = clamp(z + Δt * w2, model.terrain(x) + 10.0, Lz - 10.0)
        tr.x[i], tr.z[i] = x, z
    end
    return tr
end

total_mass(model::Model) = sum(model.Y.Yc.ρ)
total_energy(model::Model) = sum(model.Y.Yc.ρe)

end # module
