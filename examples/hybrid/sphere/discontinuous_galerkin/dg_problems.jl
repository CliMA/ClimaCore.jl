#=
REPL-friendly problem constructors for the DG-FD sphere examples.

The underlying drivers configure themselves from ENV at include time and
define constants, so they cannot be re-included in a session. These
constructors wrap that mechanism: each `run_problem` call sets the relevant
ENV entries, evaluates the driver into a FRESH anonymous module (so consts
never clash and successive runs with different settings work in one REPL
session), and restores the previous ENV afterwards (no leakage between runs
— stale ENV settings have caused real debugging pain).

Usage:
    julia --project=.buildkite
    julia> include("examples/hybrid/sphere/dg_problems.jl")
    julia> prob = BaroclinicWaveFDDG(; stepper = :hevi, dt = 60.0,
                                       t_end = 10 * 86400.0, κ₄ = 0.0,
                                       filter_Nc = 0)
    julia> result = run_problem(prob)
    julia> result.sol.u[end].Yc.ρ      # solution snapshots
    julia> result.model.Δt             # everything the driver defined

`nothing` for a keyword means "use the driver's default" (e.g. κ₄ = nothing
→ the SIPG-cap/10 default; dt = nothing → 4 s explicit / ~182 s hevi — pass
dt explicitly for hevi runs).
=#

"""
    BaroclinicWaveFDDG(; kwargs...)

Flux-form FDDG baroclinic wave / balanced flow
(`baroclinic_wave_fddg_fluxform.jl`): full (ρ, ρe, ρu⃗-Cartesian, ρw)
system, KEP Kennedy-Gruber flux differencing, explicit SSPRK33 or HEVI
(ARS343 + Newton with the flux-form Jacobian).

Keywords (defaults in parentheses):
- `helem` (4), `npoly` (4), `zelem` (10), `zmax` (30e3): resolution
- `stepper` (`:hevi`): `:hevi` or `:explicit`
- `dt` (60.0 for `:hevi`, `nothing` → 4.0 for `:explicit`): timestep [s]
- `t_end` (86400.0): simulation length [s]
- `perturb` (true): JW perturbation on/off (off = balanced-flow test)
- `κ₄` (`nothing` → the driver's SIPG-cap/10 default): biharmonic
  coefficient. Pass `κ₄ = 0.0, filter_Nc = 0` explicitly for the pure-KEP
  configuration (a stability demonstration — long perturbed runs at fine
  vertical resolution need the dissipation; see the 2026-07-26 failures).
- `filter_Nc` (0 = OFF; keep it off for this driver): the tendency cutoff
  filter is a projection applied AFTER the KEP fluxes — the KE pairing is
  bilinear with the state outside the projection, so filtering voids the
  kinetic-energy-compatibility telescoping (and exact conservation) that
  this scheme's stability rests on. Measured (helem=16, zelem=30, dt=90,
  κ₄=cap/10, Rusanov): filter_Nc=4 crashes at day 2.5 via the top-level
  drain; filter_Nc=0 runs on healthy.
- `interface_flux` (`:rusanov`): `:rusanov` (uniform |u|+c dissipation) or
  `:roe` (wave-selective; entropy/shear jumps damped at |u_n| ≈ 0 — far
  less spurious forcing of balanced jets, the Souza et al. interface)
- `dt_save` (21600.0): solution snapshot / movie-frame interval [s]
- `ndiag` (10): monitor print interval in steps
- `plots` (true): write v/u/p/T PNGs + MP4s to the output directory
"""
Base.@kwdef struct BaroclinicWaveFDDG{FT <: AbstractFloat}
    helem::Int = 4
    npoly::Int = 4
    zelem::Int = 10
    zmax::FT = 30e3
    stepper::Symbol = :hevi
    dt::Union{Nothing, FT} = stepper == :hevi ? 60.0 : nothing
    t_end::FT = 86400.0
    perturb::Bool = true
    κ₄::Union{Nothing, FT} = nothing
    filter_Nc::Union{Nothing, Int} = 0   # filtering voids the KEP property
    interface_flux::Symbol = :rusanov
    # (dz_bottom, dz_top) [m] for a stretched vertical grid; nothing = uniform
    zstretch::Union{Nothing, Tuple{FT, FT}} = nothing
    sponge_τ::FT = 1200.0
    sponge_uh::Bool = false
    dt_save::FT = 21600.0
    ndiag::Int = 10
    plots::Bool = true
end

"""
    HeldSuarezFDDG(; kwargs...)

Held–Suarez (1994) forced dry dynamical core on the flux-form FDDG driver:
identical dynamics/keywords to [`BaroclinicWaveFDDG`](@ref) plus the HS
Rayleigh low-level drag and Newtonian temperature relaxation as additive
tendencies (the KEP core is untouched — the drag is a sign-definite KE
sink). Starts from the balanced zonal jet; `perturb = true` (default) adds
the JW perturbation to break hemispheric symmetry and speed up eddy onset.

Held–Suarez-specific keywords:
- `t_end` (10 days): the canonical HS climatology needs ≳ 200-day spinup +
  a long average — short runs give qualitative diagnostics only
- `hs_spinup` (`nothing` → `t_end / 2`) [s]: snapshots with t ≥ hs_spinup
  enter the time & zonal mean u(φ,z) / T(φ,z) diagnostics
  (`u_zonal_mean.png`, `T_zonal_mean.png`)
- `dt_save` (86400.0): 1-day snapshots — sets both the movie frame rate
  (lower than the baroclinic-wave default) and the averaging sample rate

Output goes to `output/held_suarez_fddg_fluxform/` (movies + end-state PNGs
as in the baroclinic wave case, plus the zonal-mean panels).
"""
Base.@kwdef struct HeldSuarezFDDG{FT <: AbstractFloat}
    helem::Int = 4
    npoly::Int = 4
    zelem::Int = 10
    zmax::FT = 30e3
    stepper::Symbol = :hevi
    dt::Union{Nothing, FT} = stepper == :hevi ? 60.0 : nothing
    t_end::FT = 10 * 86400.0
    perturb::Bool = true
    κ₄::Union{Nothing, FT} = nothing
    filter_Nc::Union{Nothing, Int} = 0   # filtering voids the KEP property
    interface_flux::Symbol = :rusanov
    zstretch::Union{Nothing, Tuple{FT, FT}} = nothing
    sponge_τ::FT = 1200.0
    sponge_uh::Bool = false
    hs_spinup::Union{Nothing, FT} = nothing   # nothing → t_end / 2
    dt_save::FT = 86400.0
    ndiag::Int = 10
    plots::Bool = true
end

driver_file(::BaroclinicWaveFDDG) = "baroclinic_wave_fddg_fluxform.jl"
driver_file(::HeldSuarezFDDG) = "baroclinic_wave_fddg_fluxform.jl"

float_type(::BaroclinicWaveFDDG{FT}) where {FT} = FT
float_type(::HeldSuarezFDDG{FT}) where {FT} = FT

function env_settings(p::Union{BaroclinicWaveFDDG, HeldSuarezFDDG})
    env = Dict(
        "HELEM" => string(p.helem),
        "NPOLY" => string(p.npoly),
        "ZELEM" => string(p.zelem),
        "ZMAX" => string(p.zmax),
        "STEPPER" => string(p.stepper),
        "FLOAT_TYPE" => string(float_type(p)),
        "T_END" => string(p.t_end),
        "PERTURB" => p.perturb ? "1" : "0",
        "DT_SAVE" => string(p.dt_save),
        "NDIAG" => string(p.ndiag),
        "PLOTS" => p.plots ? "1" : "0",
        "INTERFACE_FLUX" => string(p.interface_flux),
        "SPONGE_TAU" => string(p.sponge_τ),
        "SPONGE_UH" => p.sponge_uh ? "1" : "0",
    )
    p.dt === nothing || (env["DT"] = string(p.dt))
    p.κ₄ === nothing || (env["KAPPA4"] = string(p.κ₄))
    p.filter_Nc === nothing || (env["FILTER"] = string(p.filter_Nc))
    p.zstretch === nothing ||
        (env["ZSTRETCH"] = string(p.zstretch[1], ",", p.zstretch[2]))
    if p isa HeldSuarezFDDG
        env["HELD_SUAREZ"] = "1"
        p.hs_spinup === nothing || (env["HS_SPINUP"] = string(p.hs_spinup))
    end
    return env
end

# every key any of the drivers reads — all are scoped (restored after the run)
const _DG_ENV_KEYS = [
    "HELEM", "NPOLY", "ZELEM", "ZMAX", "DT", "T_END", "STEPPER", "PERTURB",
    "KAPPA4", "FILTER", "DT_SAVE", "NDIAG", "PLOTS",
    "FLOAT_TYPE", "INTERFACE_FLUX", "ZSTRETCH", "SPONGE_TAU", "SPONGE_UH",
    "STATE_FILTER_ALPHA", "STATE_FILTER_KC", "STATE_FILTER_S",
    "HELD_SUAREZ", "HS_SPINUP",
]

"""
    DGRunResult

Return type of [`run_problem`](@ref): access the solution as `result.sol`
and the driver's module (every constant/function it defined) as
`result.model`. Displays as a one-line summary — the underlying solution
types are enormous (especially on GPU) and useless to print.
"""
struct DGRunResult
    sol::Any
    model::Module
end

function Base.show(io::IO, r::DGRunResult)
    t_end = try
        string(getfield(r, :sol).t[end])
    catch
        "?"
    end
    n = try
        length(getfield(r, :sol).u)
    catch
        "?"
    end
    print(
        io,
        "DGRunResult(t_end = $t_end s, $n snapshots; access via .sol, .model)",
    )
end
Base.show(io::IO, ::MIME"text/plain", r::DGRunResult) = show(io, r)

"""
    run_problem(p) -> DGRunResult

Run a DG problem defined by a constructor struct. Evaluates the driver in a
fresh anonymous module, so successive calls with different settings work
within one REPL session. Returns a [`DGRunResult`](@ref): the solution
object (`result.sol`) and the module itself (`result.model`, e.g.
`result.model.Δt`, `result.model.rhs_fddg!`).
"""
function run_problem(p)
    env = env_settings(p)
    saved = Dict(k => get(ENV, k, nothing) for k in _DG_ENV_KEYS)
    for k in _DG_ENV_KEYS
        haskey(env, k) ? (ENV[k] = env[k]) : delete!(ENV, k)
    end
    try
        mod = Module(gensym(:DGRun))
        # anonymous modules lack the `include` convenience the drivers use
        Core.eval(
            mod,
            :(include(path) = Base.include(@__MODULE__, path)),
        )
        Base.include(mod, joinpath(@__DIR__, driver_file(p)))
        return DGRunResult(mod.sol, mod)
    finally
        for (k, v) in saved
            v === nothing ? delete!(ENV, k) : (ENV[k] = v)
        end
    end
end
