#=
Baroclinic wave / balanced flow on the cubed sphere with the full system in
flux form: (ρ, ρe, ρu⃗) with momentum in global Cartesian components, all
horizontal terms discretized with Kennedy-Gruber flux-differencing volume
fluxes + KG-Rusanov interfaces (Souza et al. 2023, JAMES,
doi:10.1029/2022MS003527). The constant Cartesian basis makes component-wise
flux differencing kinetic-energy-preserving with no curvature source terms —
the volume terms cannot spuriously produce kinetic energy (TODO: explore 
whether vector-invariant form can retain this structural property)

Shallow-atmosphere: the Cartesian center momentum is kept tangential by
projecting its tendency against r̂ (the discarded radial component is the
u²/r curvature term shallow-atmosphere momentum also neglects); ρw (faces)
carries the radial momentum exactly as in the plane flux-form drivers.

Pressure gradient: Exner-perturbation form (Yatunin et al. 2026) — advective
KG momentum flux + −ρ cp_d(θ ∇Π' + θ' ∇Π_ref), well-balanced over topography.
Time integration: explicit SSPRK33 (DT default 4 s) or IMEX-HEVI (ARS343 +
Newton; implicit vertical acoustics with the Exner Jacobian in
fddg_fluxform_jacobian.jl; DT default 60 s at helem=4).

Env: HELEM, NPOLY, ZELEM, ZMAX, DT, T_END, FILTER, PERTURB, NDIAG,
     HELD_SUAREZ, HS_SPINUP
Run: PERTURB=0 DT=4 T_END=3600 julia --project=.buildkite \
         examples/hybrid/sphere/baroclinic_wave_fddg_fluxform.jl
=#

# FLOAT_TYPE = Float64 (default) | Float32 — set by the problem
# constructors' type parameter (e.g. BaroclinicWaveFDDG{Float32}(...))
const FT = get(ENV, "FLOAT_TYPE", "Float64") == "Float32" ? Float32 : Float64
# HELD_SUAREZ=1 (the HeldSuarezFDDG constructor) adds the Held–Suarez (1994)
# forcing as pointwise additive tendencies — the FDDG/KEP dynamical core is
# untouched. Adds zonal-time-mean u/T diagnostics to the plot outputs.
const apply_held_suarez = get(ENV, "HELD_SUAREZ", "0") == "1"
const is_balanced_flow = get(ENV, "PERTURB", "1") == "0"
const t_end_default = 86400.0

# STEPPER = explicit (SSPRK33, Δt default 4 s, vertical-acoustic-limited) or
# hevi (ARS343 + Newton on the flux-form vertical acoustic subsystem via
# fddg_fluxform_jacobian.jl; Δt = 60 s works at helem = 4). Note the model
# file's hevi Δt default (~182 s at helem=4) exceeds the horizontal
# advective range — pass DT explicitly for HEVI runs.

# Shared grid, constants, ICs, operators (If, Ic, vdivf2c, VanLeer, ᶠgradᵥ,
# filter, sponge). The vector-invariant rhs!/run_simulation defined there
# are simply not used.
include("sphere_dg_fd_moist_model.jl")

import ClimaCore.Geometry: ⊗
import ClimaCore.Limiters

# Moist HEVI: the implicit vertical acoustics (ρ, ρe, ρw) use the moist
# saturation-adjusted pressure in the residual; the analytic dry ∂p Jacobian
# (fddg_fluxform_jacobian.jl) serves as an APPROXIMATE preconditioner (its O(q_tot)
# error affects only Newton/linear-solve convergence, not accuracy). Moisture
# transport (ρq_tot) + microphysics stay explicit. Tracer ρq_tot rides the
# implicit solver as an explicit identity block (like ρu1..3).
using Printf

# The tendency cutoff filter is a projection applied AFTER the KEP fluxes;
# the kinetic-energy pairing is bilinear with the state outside the
# projection, so filtering voids the KEP telescoping (and the exact
# conservation) this scheme's stability rests on. Measured: helem=16,
# zelem=30, dt=90 crashes at day 2.5 with FILTER=4 and runs on healthy with
# FILTER=0. Rely on the interface (Roe/Rusanov) dissipation instead.
filter_Nc > 0 && @warn(
    "FILTER=$filter_Nc voids the KEP property of the flux-differencing " *
    "scheme and has been measured to DESTABILIZE stressed runs; " *
    "use FILTER=0 (rely on the interface dissipation) with this driver.",
)

# PGF = exner (default) | conservative | conservative_pert — the momentum
# pressure-gradient formulation, gated for the A/B comparison:
#   exner            : non-conservative Exner-perturbation PGF (Yatunin et al.
#                      2026) with advective (pressure-stripped) KG volume +
#                      interface. Well-balanced; momentum NOT conserved.
#   conservative     : full p carried in the flux (KG volume + conservative
#                      interface). Momentum conserved; NOT well-balanced over
#                      terrain (full-p PGF cancellation error).
#   conservative_pert: stratified conservative — perturbation pressure p' = p −
#                      p_ref in the momentum flux (full p in energy), buoyancy
#                      −(ρ−ρ_ref)g in ρw. Momentum conserved AND well-balanced
#                      over terrain (differences the small p').
# Cells: B=exner+roe, B'=conservative+roe, B''=conservative_pert+roe,
#        A=conservative+lmars, A'=conservative_pert+lmars.
const pgf = Symbol(lowercase(get(ENV, "PGF", "exner")))
pgf in (:exner, :conservative, :conservative_pert) ||
    error("PGF must be exner, conservative, or conservative_pert")
const is_conservative = pgf in (:conservative, :conservative_pert)

# INTERFACE_FLUX = rusanov (default) | roe | lmars. Rusanov damps every wave
# family at λ=|u|+c (over-dissipates stationary jumps); Roe damps entropy/shear
# at |û_n| (Souza et al. 2023); LMARS is a low-Mach two-wave Riemann solver
# (acoustic ∝ρc, advective ∝|u*|) — wave-selective like Roe but cheaper and with
# no sqrt(γp/ρ). With a conservative PGF LMARS carries p* in the momentum flux;
# with the Exner PGF it uses the advective variant (contact u* upwinding, no p*).
const interface_flux =
    Symbol(lowercase(get(ENV, "INTERFACE_FLUX", "rusanov")))
interface_flux in (:rusanov, :roe, :lmars, :es) ||
    error("INTERFACE_FLUX must be rusanov, roe, lmars, or es")
# es = Lax-Friedrichs dissipation in ENTROPY variables (½λĤ⟦w⟧, Ĥ=∂U/∂w SPD):
# paired with the Ranocha EC volume flux it gives a provable discrete entropy
# inequality. Requires the conservative PGF (full Euler dissipation).
(interface_flux == :es && !is_conservative) &&
    error("INTERFACE_FLUX=es requires PGF=conservative or conservative_pert")

# VOLUME_FLUX = kg (default) | ranocha. Kennedy-Gruber is KEP + pressure-
# equilibrium-preserving; Ranocha is additionally entropy-conservative (Tadmor),
# so with a dissipative interface it yields a discrete entropy inequality. Only
# for the conservative PGF (the advective/Exner volume flux carries no pressure).
const volume_flux = Symbol(lowercase(get(ENV, "VOLUME_FLUX", "kg")))
volume_flux in (:kg, :ranocha, :waruszewski) ||
    error("VOLUME_FLUX must be kg, ranocha, or waruszewski")
(volume_flux in (:ranocha, :waruszewski) && !is_conservative) && error(
    "VOLUME_FLUX=$volume_flux requires PGF=conservative or conservative_pert",
)
(volume_flux in (:ranocha, :waruszewski) && interface_flux == :lmars) &&
    error("VOLUME_FLUX=$volume_flux requires INTERFACE_FLUX=rusanov, roe, or es")
# Waruszewski (2022) is the well-balanced ENTROPY-CONSERVATIVE flux: it handles
# the geopotential as a non-conservative fluctuation term ½ρ̂⟦φ⟧ (no reference
# split), so it is EC AND machine-precision well-balanced over terrain at once.
# Pair with PGF=conservative_pert for the vertical ρw perturbation (vertical WB).

# Volume + interface fluxes for the (ρ,ρe,ρu⃗) system. The conservative
# formulations share a flux family; they differ only in the momentum pressure
# `pm` (= p or p') set in the tendency. Ranocha swaps in the entropy-conservative
# central pair (volume + interface central both use the log-mean flux).
const cartesian_volume_fn =
    !is_conservative ? Operators.kennedy_gruber_cartesian_advective_flux :
    volume_flux == :waruszewski ? Operators.waruszewski_cartesian_flux :
    volume_flux == :ranocha ? Operators.ranocha_cartesian_flux :
    Operators.kennedy_gruber_cartesian_flux
const cartesian_interface_fn =
    !is_conservative ?
    (
        interface_flux == :lmars ? Operators.lmars_cartesian_advective :
        interface_flux == :roe ?
        Operators.kennedy_gruber_roe_cartesian_advective :
        Operators.kennedy_gruber_rusanov_cartesian_advective
    ) :
    volume_flux == :waruszewski ?
    (
        interface_flux == :es ? Operators.waruszewski_es_cartesian :
        interface_flux == :roe ? Operators.waruszewski_roe_cartesian :
        Operators.waruszewski_rusanov_cartesian
    ) :
    volume_flux == :ranocha ?
    (
        interface_flux == :es ? Operators.ranocha_es_cartesian :
        interface_flux == :roe ? Operators.ranocha_roe_cartesian :
        Operators.ranocha_rusanov_cartesian
    ) :
    (
        interface_flux == :es ? Operators.kennedy_gruber_es_cartesian :
        interface_flux == :lmars ? Operators.lmars_cartesian :
        interface_flux == :roe ? Operators.kennedy_gruber_roe_cartesian :
        Operators.kennedy_gruber_rusanov_cartesian
    )

# Horizontal moisture-tracer INTERFACE flux (the KG two-point volume term is kept
# either way, as for momentum). With INTERFACE_FLUX=lmars the tracer is upwinded at
# the SAME contact velocity as the LMARS mass flux (constancy-preserving); es/roe
# have no dedicated tracer variant here, so they fall back to KG-central + Rusanov.
const tracer_interface_fn =
    interface_flux == :lmars ? Operators.lmars_tracer :
    Operators.kennedy_gruber_rusanov_tracer

# ---------------------------------------------------------------------------
# Cartesian basis fields come from sphere_dg_fd_model.jl (eE*, eN*, eR*).
# Tangential projections of the Cartesian unit vectors (state-independent):
# ê_c ⋅ ê_E and ê_c ⋅ ê_N are the Cartesian components of ê_E, ê_N.
const E1 = @. Geometry.UVVector(eE1, eN1)
const E2 = @. Geometry.UVVector(eE2, eN2)
const E3 = @. Geometry.UVVector(eE3, eN3)

# Geographic horizontal gradient of the reference Exner Π_ref (time-invariant;
# nonzero only over topography, where terrain-following levels tilt). Completed
# DG gradient = strong hgrad + central lifting, in the local orthonormal frame.
const ᶜgΠ_ref = let
    lift = Operators.lifting_correction(
        Operators.central_gradient_lift,
        Geometry.UVVector{FT},
        ᶜΠ_ref,
    )
    @. Geometry.UVVector(hgrad(ᶜΠ_ref)) + lift
end

# ---------------------------------------------------------------------------
# Initial state: convert the (discretely balanced) vector-invariant IC
# ---------------------------------------------------------------------------
Y0 = initial_state(
    Fields.local_geometry_field(hv_center_space),
    Fields.local_geometry_field(hv_face_space),
)
let uv0 = Geometry.UVVector.(Y0.uₕ)
    uE0 = uv0.components.data.:1
    uN0 = uv0.components.data.:2
    ρ0 = Y0.Yc.ρ
    global Yc0 = map(
        (ρi, ρei, ρqti, mE, mN) -> (;
            ρ = ρi,
            ρe = ρei,
            ρq_tot = ρqti,
            ρu1 = mE, # filled below
            ρu2 = mN,
            ρu3 = FT(0),
        ),
        ρ0,
        Y0.Yc.ρe,
        Y0.Yc.ρq_tot,
        ρ0 .* uE0,
        ρ0 .* uN0,
    )
    # u_c = u_E (ê_E)_c + u_N (ê_N)_c
    global Yc0 = map(
        (y, e1E, e1N, e2E, e2N, e3E, e3N) -> (;
            ρ = y.ρ,
            ρe = y.ρe,
            ρq_tot = y.ρq_tot,
            ρu1 = y.ρu1 * e1E + y.ρu2 * e1N,
            ρu2 = y.ρu1 * e2E + y.ρu2 * e2N,
            ρu3 = y.ρu1 * e3E + y.ρu2 * e3N,
        ),
        Yc0,
        eE1, eN1, eE2, eN2, eE3, eN3,
    )
end
# ρw is stored in Covariant3 basis so the HEVI Jacobian
# (fddg_fluxform_jacobian.jl) can use the proven MatrixFields machinery (g³³
# pairings) directly.
ρw0 = map(_ -> C3(FT(0)), fcoords)
Y = Fields.FieldVector(Yc = Yc0, ρw = ρw0)

const mass_0 = sum(Y.Yc.ρ)
const energy_0 = sum(Y.Yc.ρe)
const water_0 = sum(Y.Yc.ρq_tot)

# ---------------------------------------------------------------------------
# RHS
# ---------------------------------------------------------------------------
const vvdivc2f = Operators.DivergenceC2F(
    bottom = Operators.SetDivergence(Geometry.WVector(FT(0))),
    top = Operators.SetDivergence(Geometry.WVector(FT(0))),
)

# Shared tendency core: `vertical_transport = true` gives the full tendency;
# `false` gives the HEVI explicit part (everything except the central
# vertical mass/energy fluxes and the ρw pressure-gradient + buoyancy terms,
# which live in implicit_tendency_fddg!). The (VanLeer − central) energy
# correction and the full momentum-component vertical transport stay
# explicit, so the HEVI total equals the fully explicit path.
# --- Report-only diagnostic scan (DEBUG_SCAN=1) -----------------------------
# Prints the min/max of the prognostics ENTERING the tendency, plus a finiteness
# flag, so the first field + time to go bad is visible immediately before a device
# DomainError (ρ≤0 ⇒ dynamics / waruszewski ln_mean poison; ρq_tot<0 ⇒ moisture).
# Because CUDA kernel exceptions are async, the LAST printed scan is the culprit
# state and the crash surfaces at the next scan's reduction. GPU-safe (whole-array
# reductions, no scalar indexing) and STRICTLY read-only — no clamping of the state.
const debug_scan = get(ENV, "DEBUG_SCAN", "0") == "1"
@inline _dmin(f) = minimum(parent(f))
@inline _dmax(f) = maximum(parent(f))
@inline _dbad(f) = any(x -> !isfinite(x), parent(f))
function debug_scan_fddg!(Y, t, tag)
    debug_scan || return nothing
    Yc = Y.Yc
    finite = !(
        _dbad(Yc.ρ) || _dbad(Yc.ρe) || _dbad(Yc.ρq_tot) ||
        _dbad(Yc.ρu1) || _dbad(Yc.ρu2) || _dbad(Yc.ρu3) || _dbad(Y.ρw)
    )
    @info "scan[$tag]" t minρ = _dmin(Yc.ρ) minρe = _dmin(Yc.ρe) minρq =
        _dmin(Yc.ρq_tot) maxρq = _dmax(Yc.ρq_tot) finite
    return nothing
end

function compute_tendency_fddg!(dY, Y, t, vertical_transport)
    debug_scan_fddg!(Y, t, vertical_transport ? :rhs : :remaining)
    Yc = Y.Yc
    ρw = Y.ρw
    dYc = dY.Yc
    dρw = dY.ρw
    ρ = Yc.ρ
    ρe = Yc.ρe
    ρw_w = @. Geometry.WVector(ρw)
    lgeom_c = Fields.local_geometry_field(hv_center_space)
    lgeom_f = Fields.local_geometry_field(hv_face_space)

    # Velocities: tangential-project the state (guards roundoff drift)
    u1r = @. Yc.ρu1 / ρ
    u2r = @. Yc.ρu2 / ρ
    u3r = @. Yc.ρu3 / ρ
    ur = @. u1r * eR1 + u2r * eR2 + u3r * eR3
    u1 = @. u1r - ur * eR1
    u2 = @. u2r - ur * eR2
    u3 = @. u3r - ur * eR3
    uE = @. u1 * eE1 + u2 * eE2 + u3 * eE3
    uN = @. u1 * eN1 + u2 * eN2 + u3 * eN3
    uv = @. Geometry.UVVector(uE, uN)
    w_c = @. Ic(ρw_w).components.data.:1 / ρ

    K = @. (uE^2 + uN^2 + w_c^2) / 2
    e = @. ρe / ρ
    # Moist saturation-adjusted thermodynamics (Thermodynamics.jl): internal
    # energy per mass e_int = ρe/ρ − K − Φ, total specific humidity
    # q_tot = ρq_tot/ρ ⇒ equilibrium T, moist pressure p = ρ R_m T, and the
    # condensate partition (q_liq, q_ice) used by the 0-moment microphysics.
    q_tot = @. Yc.ρq_tot / ρ
    e_int = @. e - K - ᶜΦ
    # Robust (non-throwing) dynamics pressure/temperature (ClimaAtmos T_min
    # mechanism — see moist_p_dyn): safe on the implicit solver's transient
    # iterates, used identically in the implicit tendency so the HEVI split is
    # exact. Condensate for microphysics from the floored T via the closed-form
    # condensate_partition (T > 0 ⇒ non-throwing) — saturation_adjustment is kept
    # out of the tendencies entirely.
    dyn = @. moist_p_dyn(ρ, e_int, q_tot)
    p = dyn.p
    T_air = dyn.T
    cond = @. condensate_partition_tuple(T_air, ρ, q_tot)
    q_liq = cond.q_liq
    q_ice = cond.q_ice
    h_tot = @. (ρe + p) / ρ
    λ = @. sqrt(uE^2 + uN^2) + sqrt(γ * p / ρ)
    # LMARS reference impedance uses a REFERENCE sound speed c_ref = √(γ R_d
    # T_ref) (consistent with the isothermal reference state), not the local
    # √(γp/ρ). This is LMARS's low-Mach design — a fixed reference impedance,
    # independent of the local p — so it is robust by construction WITHOUT any
    # unphysical clamp on pressure (a genuinely negative p still surfaces via λ
    # above, as it should, rather than being silently floored).
    c = @. sqrt(γ * R_d * T_ref) + zero(p)
    # Momentum pressure carried in the conservative flux: full p (conservative)
    # or perturbation p' = p − p_ref (conservative_pert, well-balanced over
    # terrain). Unused by the advective flux (PGF=exner).
    pm = pgf == :conservative_pert ? (@. p - ᶜp_ref) : p
    # Exner function / potential-temperature deviations — only needed (and only
    # well-defined: (p/p₀)^κ requires p>0) for PGF=exner. Computing them for the
    # conservative path would gratuitously DomainError on a transient p<0.
    if pgf == :exner
        Π = @. (p / p_0)^κ_gas
        θ = @. p / (ρ * R_d) / Π
        Πp = @. Π - ᶜΠ_ref
        θp = @. θ - ᶜθ_ref
    end

    # --- Horizontal: FDDG volume + interface (flux gated by PGF/INTERFACE_FLUX) ---
    y = map(
        (ρi, ρei, ei, pi, pmi, uvi, u1i, u2i, u3i, E1i, E2i, E3i, λi, ci, φi, qi) ->
            (;
                ρ = ρi, ρe = ρei, e = ei, p = pi, pm = pmi, uv = uvi,
                u1 = u1i, u2 = u2i, u3 = u3i,
                E1 = E1i, E2 = E2i, E3 = E3i, λ = λi, c = ci, φ = φi, q = qi,
            ),
        ρ, ρe, e, p, pm, uv, u1, u2, u3, E1, E2, E3, λ, c, ᶜΦ, q_tot,
    )
    dy_mw = map(
        _ -> (ρ = FT(0), ρe = FT(0), ρu1 = FT(0), ρu2 = FT(0), ρu3 = FT(0)),
        ρ,
    )
    Operators.add_flux_differencing_divergence!(cartesian_volume_fn, dy_mw, y)
    Operators.add_numerical_flux_internal!(cartesian_interface_fn, dy_mw, y)
    @. dYc.ρ = dy_mw.ρ / lgeom_c.WJ
    @. dYc.ρe = dy_mw.ρe / lgeom_c.WJ
    @. dYc.ρu1 = dy_mw.ρu1 / lgeom_c.WJ
    @. dYc.ρu2 = dy_mw.ρu2 / lgeom_c.WJ
    @. dYc.ρu3 = dy_mw.ρu3 / lgeom_c.WJ

    # --- Horizontal moisture tracer transport: ρq_tot advected by the SAME KG
    #     mass flux as continuity (passive scalar), with a Rusanov penalty. ---
    dy_q = map(_ -> (ρq = FT(0),), ρ)
    Operators.add_flux_differencing_divergence!(
        Operators.kennedy_gruber_tracer_flux,
        dy_q,
        y,
    )
    Operators.add_numerical_flux_internal!(
        tracer_interface_fn,
        dy_q,
        y,
    )
    @. dYc.ρq_tot = dy_q.ρq / lgeom_c.WJ

    # --- Horizontal Exner-perturbation pressure-gradient force (PGF=exner only;
    #     Yatunin et al. 2026), Cartesian components: −ρ c_pd (θ ∇ₕΠ' + θ' ∇ₕΠ_ref).
    #     DG gradient = strong hgrad + central lifting (no DSS), geographic →
    #     Cartesian. ∇ₕ of a z-only field vanishes on level surfaces, so at rest
    #     (θ'=Π'=0) this is exactly zero — well-balanced over topography, unlike
    #     the metric-defective conservative pressure flux. Under PGF=conservative
    #     the pressure is carried in cartesian_volume_fn/cartesian_interface_fn. ---
    if pgf == :exner
        liftΠp = Operators.lifting_correction(
            Operators.central_gradient_lift,
            Geometry.UVVector{FT},
            Πp,
        )
        gΠp = @. Geometry.UVVector(hgrad(Πp)) + liftΠp
        pgfE =
            @. -cp_d *
               (θ * gΠp.components.data.:1 + θp * ᶜgΠ_ref.components.data.:1)
        pgfN =
            @. -cp_d *
               (θ * gΠp.components.data.:2 + θp * ᶜgΠ_ref.components.data.:2)
        @. dYc.ρu1 += ρ * (pgfE * eE1 + pgfN * eN1)
        @. dYc.ρu2 += ρ * (pgfE * eE2 + pgfN * eN2)
        @. dYc.ρu3 += ρ * (pgfE * eE3 + pgfN * eN3)
    end

    # --- Vertical FD (plane flux-form pattern; implicit under HEVI) ---
    if vertical_transport
        @. dYc.ρ -= vdivf2c(ρw_w)
        @. dYc.ρe -= vdivf2c(VanLeer(ρw_w, h_tot, Δt))
    else
        # mass flux is fully implicit (linear); energy gets the explicit
        # (VanLeer − central) correction so the HEVI total is Lin-VanLeer
        @. dYc.ρe -=
            vdivf2c(VanLeer(ρw_w, h_tot, Δt)) - vdivf2c(ρw_w * If(h_tot))
    end
    @. dYc.ρu1 -= vdivf2c(VanLeer(ρw_w, u1, Δt))
    @. dYc.ρu2 -= vdivf2c(VanLeer(ρw_w, u2, Δt))
    @. dYc.ρu3 -= vdivf2c(VanLeer(ρw_w, u3, Δt))
    # Vertical moisture transport: monotone Lin-VanLeer of q_tot on the mass flux.
    @. dYc.ρq_tot -= vdivf2c(VanLeer(ρw_w, q_tot, Δt))

    # --- 0-moment microphysics (CloudMicrophysics.jl): instantaneous removal of
    #     condensate above the threshold as precipitation. S ≤ 0 [1/s] is the
    #     specific-humidity sink; it removes mass (ρq_tot) and the internal +
    #     potential energy carried by the precipitated condensate (ρe). The
    #     removed water is partitioned liquid/ice by (q_liq, q_ice); its specific
    #     internal energy is Iₗ = cv_l(T−T_0) for liquid, Iᵢ = cv_i(T−T_0) −
    #     e_int_i0 for ice (Thermodynamics reference), plus geopotential Φ. ---
    S_qt = @. M0M.remove_precipitation(cm_params.precip, q_liq, q_ice)
    q_c = @. q_liq + q_ice
    f_liq = @. ifelse(q_c > 0, q_liq / q_c, one(q_c))
    I_c = @. f_liq * cv_l * (T_air - T_0_td) +
       (1 - f_liq) * (cv_i * (T_air - T_0_td) - e_int_i0)
    @. dYc.ρq_tot += ρ * S_qt
    @. dYc.ρe += ρ * S_qt * (I_c + ᶜΦ)

    # --- Coriolis: −2Ω ẑ×u⃗, exact in the constant Cartesian frame ---
    @. dYc.ρu1 += 2 * Ω * ρ * u2
    @. dYc.ρu2 -= 2 * Ω * ρ * u1

    # --- Held–Suarez forcing (HS94): Rayleigh low-level drag on the
    #     tangential wind + Newtonian relaxation of T to the analytic
    #     equilibrium. Pointwise and additive — the drag is a sign-definite
    #     KE sink, so the KEP advective core is untouched (constants from
    #     sphere_dg_fd_model.jl, same as the vector-invariant HS driver). ---
    if apply_held_suarez
        φ = @. deg2rad(ccoords.lat)
        σ = @. p / p_0
        height_factor = @. max(0, (σ - σ_b) / (1 - σ_b))
        ΔρT = @. (k_a + (k_s - k_a) * height_factor * cos(φ)^4) *
           ρ *
           (
               p / (ρ * R_d) - max(
                   T_min,
                   (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) *
                   σ^(R_d / cp_d),
               )
           )
        @. dYc.ρu1 -= (k_f * height_factor) * ρ * u1
        @. dYc.ρu2 -= (k_f * height_factor) * ρ * u2
        @. dYc.ρu3 -= (k_f * height_factor) * ρ * u3
        @. dYc.ρe -= ΔρT * cv_d
    end

    # --- Tangential projection of the momentum tendency (shallow atm.) ---
    dmr = @. dYc.ρu1 * eR1 + dYc.ρu2 * eR2 + dYc.ρu3 * eR3
    @. dYc.ρu1 -= dmr * eR1
    @. dYc.ρu2 -= dmr * eR2
    @. dYc.ρu3 -= dmr * eR3

    # --- ρw: pressure gradient + buoyancy (discretely balanced pair,
    #     implicit under HEVI), vertical advection, horizontal DG
    #     advection, sponge ---
    w = @. ρw_w / If(ρ)
    if vertical_transport
        if pgf == :exner
            # Exner-perturbation PGF + gravity (Yatunin et al. 2026): gravity
            # absorbed into the reference term, so at rest (Π'=θ'=0) exactly 0.
            @. dρw = Bw(
                -If(ρ) *
                cp_d *
                (If(θ) * ᶠgradᵥ(Πp) + If(θp) * ᶠgradᵥ(ᶜΠ_ref)) -
                C3(vvdivc2f(Ic(ρw_w ⊗ w)), lgeom_f),
            )
        elseif pgf == :conservative
            # Conservative full-p pressure gradient + gravity (balanced pair;
            # relies on the discrete-hydrostatic IC, i.e. REBALANCE=1).
            @. dρw = Bw(
                -(ᶠgradᵥ(p) + If(ρ) * ᶠgradᵥ(ᶜΦ)) -
                C3(vvdivc2f(Ic(ρw_w ⊗ w)), lgeom_f),
            )
        else
            # Stratified conservative: −∂_ξ³ p' − (ρ−ρ_ref) g (buoyancy pair),
            # pm = p'. Differences the small p' ⇒ well-balanced over terrain.
            @. dρw = Bw(
                -(ᶠgradᵥ(pm) + If(ρ - ᶜρ_ref) * ᶠgradᵥ(ᶜΦ)) -
                C3(vvdivc2f(Ic(ρw_w ⊗ w)), lgeom_f),
            )
        end
    else
        @. dρw = Bw(-C3(vvdivc2f(Ic(ρw_w ⊗ w)), lgeom_f))
    end
    ρw_sc = @. ρw_w.components.data.:1
    uvf = @. If(uv)
    λf = interface_flux == :roe ? (@. If(sqrt(uE^2 + uN^2))) : (@. If(λ))
    y_f = map((h, uvi, λi) -> (; h = h, uv = uvi, λ = λi), ρw_sc, uvf, λf)
    dρw_mw = @. hwdiv(uvf * ρw_sc) * (-(lgeom_f.WJ))
    Operators.add_numerical_flux_internal!(
        Operators.kennedy_gruber_rusanov_height,
        dρw_mw,
        y_f,
    )
    @. dρw += C3(Geometry.WVector(dρw_mw / lgeom_f.WJ), lgeom_f)
    @. dρw -= ᶠβ_sponge * ρw
    if sponge_uh
        @. dYc.ρu1 -= ᶜβ_sponge * Yc.ρu1
        @. dYc.ρu2 -= ᶜβ_sponge * Yc.ρu2
        @. dYc.ρu3 -= ᶜβ_sponge * Yc.ρu3
    end

    # --- Cutoff filter on the tendencies ---
    if filter_Nc > 0
        M = Quadratures.cutoff_filter_matrix(
            FT,
            Spaces.quadrature_style(hv_center_space),
            filter_Nc,
        )
        for f in (dYc.ρ, dYc.ρe, dYc.ρu1, dYc.ρu2, dYc.ρu3, dρw)
            data = Fields.field_values(f)
            Operators.tensor_product!(data, data, M)
        end
        @. dρw = Bw(dρw)
    end
    return dY
end

rhs_fddg!(dY, Y, p, t) = compute_tendency_fddg!(dY, Y, t, true)
remaining_tendency_fddg!(dY, Y, p, t) = compute_tendency_fddg!(dY, Y, t, false)

# HEVI implicit part: central vertical mass/energy fluxes + ρw pressure
# gradient and buoyancy (the discretely balanced pair). Linear in ρw given
# frozen h_tot; Jacobian in fddg_fluxform_jacobian.jl.
function implicit_tendency_fddg!(dY, Y, p, t)
    Yc = Y.Yc
    ρ = Yc.ρ
    ρe = Yc.ρe
    ρw_w = @. Geometry.WVector(Y.ρw)

    uE = @. (Yc.ρu1 * eE1 + Yc.ρu2 * eE2 + Yc.ρu3 * eE3) / ρ
    uN = @. (Yc.ρu1 * eN1 + Yc.ρu2 * eN2 + Yc.ρu3 * eN3) / ρ
    w_c = @. Ic(ρw_w).components.data.:1 / ρ
    K = @. (uE^2 + uN^2 + w_c^2) / 2
    # Robust moist pressure (same moist_p_dyn as the explicit path ⇒ HEVI split
    # stays exact AND the implicit solver's transient iterates never trigger a
    # saturation-adjustment DomainError — the ClimaAtmos frozen/floored-thermo
    # mechanism). The analytic dry ∂p in the Jacobian is the preconditioner.
    q_tot = @. Yc.ρq_tot / ρ
    e_int = @. ρe / ρ - K - ᶜΦ
    p_thermo = (@. moist_p_dyn(ρ, e_int, q_tot)).p
    h_tot = @. (ρe + p_thermo) / ρ
    pm = pgf == :conservative_pert ? (@. p_thermo - ᶜp_ref) : p_thermo
    if pgf == :exner
        Π = @. (p_thermo / p_0)^κ_gas
        θ = @. p_thermo / (ρ * R_d) / Π
        Πp = @. Π - ᶜΠ_ref
        θp = @. θ - ᶜθ_ref
    end

    @. dY.Yc.ρ = -vdivf2c(ρw_w)
    @. dY.Yc.ρe = -vdivf2c(ρw_w * If(h_tot))
    dY.Yc.ρu1 .= FT(0)
    dY.Yc.ρu2 .= FT(0)
    dY.Yc.ρu3 .= FT(0)
    dY.Yc.ρq_tot .= FT(0)   # moisture transport is explicit (remaining_tendency)
    if pgf == :exner
        @. dY.ρw =
            Bw(-If(ρ) * cp_d * (If(θ) * ᶠgradᵥ(Πp) + If(θp) * ᶠgradᵥ(ᶜΠ_ref)))
    elseif pgf == :conservative
        @. dY.ρw = Bw(-(ᶠgradᵥ(p_thermo) + If(ρ) * ᶠgradᵥ(ᶜΦ)))
    else
        @. dY.ρw = Bw(-(ᶠgradᵥ(pm) + If(ρ - ᶜρ_ref) * ᶠgradᵥ(ᶜΦ)))
    end
    return dY
end

include("fddg_fluxform_jacobian.jl")

# ---------------------------------------------------------------------------
# Time integration (explicit SSPRK33) with a step monitor
# ---------------------------------------------------------------------------
@info "Momentum scheme" pgf volume_flux interface_flux stepper
dY = similar(Y)
rhs_fddg!(dY, Y, nothing, FT(0))
@info "Initial RHS" max_dρ = maximum(abs, parent(dY.Yc.ρ)) max_dρe =
    maximum(abs, parent(dY.Yc.ρe)) max_dρu = max(
    maximum(abs, parent(dY.Yc.ρu1)),
    maximum(abs, parent(dY.Yc.ρu2)),
    maximum(abs, parent(dY.Yc.ρu3)),
) max_dρw = maximum(abs, parent(dY.ρw))

const ndiag = parse(Int, get(ENV, "NDIAG", "150"))

function diag_str(Y, t)
    ρ = Y.Yc.ρ
    ρw_w = @. Geometry.WVector(Y.ρw)
    uE = @. (Y.Yc.ρu1 * eE1 + Y.Yc.ρu2 * eE2 + Y.Yc.ρu3 * eE3) / ρ
    uN = @. (Y.Yc.ρu1 * eN1 + Y.Yc.ρu2 * eN2 + Y.Yc.ρu3 * eN3) / ρ
    w_c = @. Ic(ρw_w).components.data.:1 / ρ
    K = @. (uE^2 + uN^2 + w_c^2) / 2
    q_tot = @. Y.Yc.ρq_tot / ρ
    e_int = @. Y.Yc.ρe / ρ - K - ᶜΦ
    # Use the non-throwing dynamics thermo (moist_p_dyn / condensate_partition_tuple)
    # for the monitor too: saturation_adjustment has exp/log/^ that throw a
    # DomainError on an extreme transient column and can kill an otherwise-healthy
    # run *from the diagnostic*. moist_p_dyn floors T > 0, so condensate_partition
    # is safe.
    dyn = @. moist_p_dyn(ρ, e_int, q_tot)
    p = dyn.p
    cond = @. condensate_partition_tuple(dyn.T, ρ, q_tot)
    q_c = @. cond.q_liq + cond.q_ice
    total_water = Fields.sum(Y.Yc.ρq_tot)
    @sprintf(
        "t=%8.0f  max|w|=%.3e  max|v|=%.3e  min p=%.3e  min ρ=%.3e  max q_tot=%.4e  max q_cond=%.3e  ∫ρq=%.6e",
        t,
        maximum(abs, parent(ρw_w)) / maximum(parent(ρ)),
        maximum(abs, parent(uN)),
        minimum(parent(p)),
        minimum(parent(ρ)),
        maximum(parent(q_tot)),
        maximum(parent(q_c)),
        total_water,
    )
end

saveat_grid =
    collect(FT(0):min(t_end, parse(FT, get(ENV, "DT_SAVE", "21600"))):t_end)

# Diagnostic monitor at a fixed simulated-time interval (CTS-native callback;
# CTS 0.10 does not accept SciMLBase's DiscreteCallback).
monitor = CTS.Callbacks.EveryXSimulationTime(
    integrator -> println(diag_str(integrator.u, integrator.t)),
    max(Δt, ndiag * Δt);
    atinit = true,
)
callback = CTS.CallbackSet(monitor)

# --- Bound-preserving moisture limiter (Guba et al. 2014, QuasiMonotone "OP1") ---
# The horizontal DG ρq_tot transport (KG/LMARS two-point + interface flux) is not
# sign-preserving at GLL nodes: over steep terrain it Gibbs-undershoots q_tot < 0,
# which then poisons the moist saturation/condensate thermodynamics (the source of
# the transient sqrt/log/^ DomainErrors). QuasiMonotoneLimiter redistributes tracer
# mass *within each element* (conservatively) to keep nodal q_tot inside its
# element+neighbor [min,max] — a monotone projection, NOT an unphysical clamp of the
# prognostic. It is applied once per RK stage through ClimaTimeSteppers' `lim!` hook
# (imex_ark.jl calls `f.lim!(U, p, t, u_ref)` each stage i≠1), so every stage's
# tendency sees an in-bounds q_tot before the thermo runs. Bounds are taken from the
# stage reference state `u_ref`; the projection is applied to the incremented `U`.
const use_tracer_limiter = get(ENV, "TRACER_LIMITER", "1") == "1"
const tracer_limiter = Limiters.QuasiMonotoneLimiter(Y.Yc.ρq_tot)
function lim_fddg!(U, p, t, u_ref)
    use_tracer_limiter || return nothing
    Limiters.compute_bounds!(tracer_limiter, u_ref.Yc.ρq_tot, u_ref.Yc.ρ)
    # POSITIVITY: QuasiMonotone by itself only enforces the element+neighbor
    # [min,max]; once a whole region undershoots, its lower bound is itself < 0 and
    # q_tot stays negative → R_m = R_d+(R_v−R_d)q_tot flips sign → p = ρR_mT < 0 →
    # sqrt(γp/ρ) / ln_mean(ρ/2p) DomainError (diagnosed on CPU). Floor the lower
    # bound at 0 so the limiter is bound-preserving WITH a physical q_tot ≥ 0 floor
    # (standard positivity limiter, mass-conserving when the element mean ≥ 0 — not a
    # pointwise clamp of the prognostic). q_bounds_nbr stores [min,max] on axis 2;
    # index 1 = min. selectdim keeps this rank/device-agnostic (GPU broadcast-safe).
    q_min = selectdim(parent(tracer_limiter.q_bounds_nbr), 2, 1)
    before = debug_scan ? minimum(parent(U.Yc.ρq_tot)) : 0.0
    boundmin_pre = debug_scan ? minimum(q_min) : 0.0
    @. q_min = max(q_min, 0)
    Limiters.apply_limiter!(U.Yc.ρq_tot, U.Yc.ρ, tracer_limiter)
    debug_scan && @info "lim!" t minρq_before = before minρq_after =
        minimum(parent(U.Yc.ρq_tot)) boundmin_pre boundmin_post = minimum(q_min)
    return nothing
end

if stepper == "hevi"
    # Split-consistency check: rhs == implicit + remaining (exact when the
    # tendency filter is off; with the filter on, the implicit part is
    # unfiltered — same convention as the vector-invariant model).
    let dY1 = similar(Y), dY2 = similar(Y), dY3 = similar(Y)
        rhs_fddg!(dY1, Y, nothing, FT(0))
        implicit_tendency_fddg!(dY2, Y, nothing, FT(0))
        remaining_tendency_fddg!(dY3, Y, nothing, FT(0))
        r(f) = maximum(
            abs,
            parent(getproperty(dY1, f)) .- parent(getproperty(dY2, f)) .-
            parent(getproperty(dY3, f)),
        )
        @info "HEVI split check (exact only when FILTER=0)" split_Yc =
            r(:Yc) split_ρw = r(:ρw)
    end
    jacobian = FDDGImplicitEquationJacobian(Y)
    ode_function = CTS.ClimaODEFunction(;
        T_imp! = CTS.ODEFunction(
            implicit_tendency_fddg!;
            jac_prototype = jacobian,
            Wfact = fddg_implicit_equation_jacobian!,
        ),
        # NB: the explicit tendency is passed as T_lim! (NOT T_exp!): ClimaTimeSteppers
        # only calls the `lim!` hook when a T_lim! is present (has_T_lim ⇔ _has_lim,
        # set iff T_lim! is given; imex_ark.jl gates `f.lim!` on has_T_lim). Numerically
        # identical to T_exp! (same explicit tableau coeffs) but activates the per-stage
        # QuasiMonotone positivity limiter on ρq_tot.
        T_lim! = remaining_tendency_fddg!,
        lim! = lim_fddg!,
    )
    ode_algo =
        CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 2))
else
    ode_function = CTS.ClimaODEFunction(; T_lim! = rhs_fddg!, lim! = lim_fddg!)
    ode_algo = CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher())
end

prob = CTS.ODEProblem(ode_function, Y, (FT(0), t_end), nothing)
integrator = CTS.init(
    prob,
    ode_algo;
    dt = Δt,
    saveat = saveat_grid,
    callback = callback,
    adaptive = false,
)
sol = CTS.solve!(integrator)

@info (
    apply_held_suarez ?
    "Conservation (energy is forced under Held–Suarez; mass exact)" :
    "Conservation"
) mass_rel = (sum(sol.u[end].Yc.ρ) - mass_0) / mass_0 energy_rel =
    (sum(sol.u[end].Yc.ρe) - energy_0) / energy_0 water_change =
    sum(sol.u[end].Yc.ρq_tot) - water_0 water_rel =
    water_0 > 0 ? (sum(sol.u[end].Yc.ρq_tot) - water_0) / water_0 : FT(0)
if is_balanced_flow && !apply_held_suarez
    Yend = sol.u[end]
    uN_end = @. (
        Yend.Yc.ρu1 * eN1 + Yend.Yc.ρu2 * eN2 + Yend.Yc.ρu3 * eN3
    ) / Yend.Yc.ρ
    @info "Balanced-flow drift" max_v = maximum(abs, parent(uN_end)) max_ρw =
        maximum(abs, parent(Geometry.WVector.(Yend.ρw)))
end

# ---------------------------------------------------------------------------
# Plots (v/u at level 3, p/T at level 1; PLOTS=0 disables). Uses CairoMakie +
# the ClimaCoreMakie extension; both must be in the active environment (see the
# note in the header — add them to .buildkite or run from an env that has
# them). Import is inside the guard so PLOTS=0 runs need neither. GPU-safe:
# each plotted field is moved to the CPU with `ClimaCore.to_cpu`.
# ---------------------------------------------------------------------------
if get(ENV, "PLOTS", "1") != "0"
import CairoMakie, ClimaCoreMakie
output_dir = joinpath(
    @__DIR__,
    "output",
    apply_held_suarez ? "held_suarez_fddg_fluxform" :
    "moist_baroclinic_wave_fddg_fluxform",
)
mkpath(output_dir)

# Form the diagnostics on the device (all fields share spaces there), then
# move each scalar result to the CPU for the plot recipes.
function plot_fields_cpu(Yi)
    Yc = Yi.Yc
    ρ = Yc.ρ
    uE = @. (Yc.ρu1 * eE1 + Yc.ρu2 * eE2 + Yc.ρu3 * eE3) / ρ
    uN = @. (Yc.ρu1 * eN1 + Yc.ρu2 * eN2 + Yc.ρu3 * eN3) / ρ
    w_c = @. Ic(Geometry.WVector(Yi.ρw)).components.data.:1 / ρ
    K = @. (uE^2 + uN^2 + w_c^2) / 2
    q_tot = @. Yc.ρq_tot / ρ
    e_int = @. Yc.ρe / ρ - K - ᶜΦ
    # Non-throwing thermo for plotting (see diag_str): saturation_adjustment can
    # DomainError on an extreme column at output time.
    dyn = @. moist_p_dyn(ρ, e_int, q_tot)
    p = dyn.p
    T = dyn.T
    return (;
        u = ClimaCore.to_cpu(uE),
        v = ClimaCore.to_cpu(uN),
        p = ClimaCore.to_cpu(p),
        T = ClimaCore.to_cpu(T),
        q_tot = ClimaCore.to_cpu(q_tot),
    )
end

# ClimaCoreMakie.fieldheatmap plots a 2D field, so slice the requested
# horizontal level out of the extruded field first. On the cubed sphere the
# level's coordinates are LatLongPoints, so this renders a long–lat map.
function save_level_heatmap(path, field, lev; colorrange = nothing)
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel = "long [deg]", ylabel = "lat [deg]")
    kw = isnothing(colorrange) ? (;) : (; colorrange)
    plt = ClimaCoreMakie.fieldheatmap!(ax, Fields.level(field, lev); kw...)
    CairoMakie.Colorbar(fig[1, 2], plt)
    CairoMakie.save(path, fig)
    return nothing
end

function save_level_animation(
    path,
    states,
    to_field,
    lev;
    colorrange = nothing,
    framerate = 5,
)
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel = "long [deg]", ylabel = "lat [deg]")
    frame = CairoMakie.Observable(Fields.level(to_field(first(states)), lev))
    kw = isnothing(colorrange) ? (;) : (; colorrange)
    plt = ClimaCoreMakie.fieldheatmap!(ax, frame; kw...)
    CairoMakie.Colorbar(fig[1, 2], plt)
    CairoMakie.record(fig, path, states; framerate) do Yi
        frame[] = Fields.level(to_field(Yi), lev)
    end
    return nothing
end

let f_end = plot_fields_cpu(sol.u[end])
    save_level_heatmap(
        joinpath(output_dir, "v_end.png"),
        f_end.v,
        3;
        colorrange = (-6, 6),
    )
    save_level_heatmap(joinpath(output_dir, "p_sfc_end.png"), f_end.p, 1)
end
if length(sol.u) > 2
    for (name, getfield_fn, lev, colorrange) in (
        ("v", f -> f.v, 3, (-6, 6)),
        ("u", f -> f.u, 3, nothing),
        ("p_sfc", f -> f.p, 1, nothing),
        ("T_sfc", f -> f.T, 1, nothing),
    )
        save_level_animation(
            joinpath(output_dir, "$name.mp4"),
            sol.u,
            Yi -> getfield_fn(plot_fields_cpu(Yi)),
            lev;
            colorrange,
        )
    end
end

# --- Canonical Held–Suarez diagnostics: time & zonal mean u(φ, z) and
#     T(φ, z), quadrature-weighted (WJ) latitude binning of the saved
#     snapshots past the spinup time. HS_SPINUP [s] defaults to t_end/2;
#     Held & Suarez (1994) use a 200-day spinup and a long-time mean, so
#     treat short-run output as qualitative. ---
if apply_held_suarez && length(sol.u) > 1
    hs_spinup = parse(FT, get(ENV, "HS_SPINUP", string(t_end / 2)))
    avg_idx = [i for (i, ti) in enumerate(sol.t) if ti >= hs_spinup]
    isempty(avg_idx) && (avg_idx = [length(sol.u)])
    nbins = 45
    edges = range(FT(-90), FT(90); length = nbins + 1)
    lat_centers = collect((edges[1:(end - 1)] .+ edges[2:end]) ./ 2)
    lat_p = parent(ClimaCore.to_cpu(ccoords.lat))
    wj_p = parent(
        ClimaCore.to_cpu(Fields.local_geometry_field(hv_center_space).WJ),
    )
    z_p = parent(ClimaCore.to_cpu(ccoords.z))
    Nv = size(lat_p, 1)
    z_km = [
        sum(view(z_p, v, :, :, :, :)) / length(view(z_p, v, :, :, :, :)) /
        1e3 for v in 1:Nv
    ]
    usum = zeros(Nv, nbins)
    Tsum = zeros(Nv, nbins)
    wsum = zeros(Nv, nbins)
    for i in avg_idx
        # `local`: this runs at top level, where loop-body assignments that
        # shadow globals (T_p, b, ...) are ambiguous soft scope
        local fi = plot_fields_cpu(sol.u[i])
        local u_p = parent(fi.u)
        local T_p = parent(fi.T)
        for I in CartesianIndices(lat_p)
            local b = clamp(searchsortedlast(edges, lat_p[I]), 1, nbins)
            local w = wj_p[I]
            usum[I[1], b] += w * u_p[I]
            Tsum[I[1], b] += w * T_p[I]
            wsum[I[1], b] += w
        end
    end
    ubar = usum ./ wsum   # empty bins → NaN → contour gaps
    Tbar = Tsum ./ wsum
    day_str(i) = string(round(sol.t[i] / 86400; digits = 1))
    span = "days $(day_str(avg_idx[1]))–$(day_str(avg_idx[end]))"
    # These are plain (lat × z) matrices, not ClimaCore fields, so use
    # CairoMakie's contourf directly. Makie expects z of size
    # (length(x), length(y)) = (nbins, Nv), hence the permutedims.
    let fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(
            fig[1, 1];
            xlabel = "latitude [deg]",
            ylabel = "z [km]",
            title = "zonal-mean u [m/s], $span",
        )
        cf = CairoMakie.contourf!(
            ax,
            lat_centers,
            z_km,
            permutedims(ubar);
            colormap = :balance,
        )
        CairoMakie.Colorbar(fig[1, 2], cf)
        CairoMakie.save(joinpath(output_dir, "u_zonal_mean.png"), fig)
    end
    let fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(
            fig[1, 1];
            xlabel = "latitude [deg]",
            ylabel = "z [km]",
            title = "zonal-mean T [K], $span",
        )
        cf = CairoMakie.contourf!(
            ax,
            lat_centers,
            z_km,
            permutedims(Tbar);
            colormap = :thermal,
        )
        CairoMakie.Colorbar(fig[1, 2], cf)
        CairoMakie.save(joinpath(output_dir, "T_zonal_mean.png"), fig)
    end
    @info "Held–Suarez zonal-mean diagnostics" averaged_snapshots =
        length(avg_idx) window = span
end
@info "Output written to $output_dir"
end # PLOTS

# script value: keep REPL `include(...)` from displaying the (enormous)
# solution object; access it as `sol` / via run_problem's DGRunResult
nothing
