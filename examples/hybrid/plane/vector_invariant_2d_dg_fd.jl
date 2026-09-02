#=
Vector-invariant compressible Euler on an extruded x–z plane with
  • discontinuous Galerkin (DG) horizontal spectral elements (no DSS)
  • finite-difference vertical staggering (Atmos-like)

Companion to the flux-form files `density_current_2d_dg_fd.jl` /
`rising_bubble_2d_dg_fd.jl`: same meshes, EOS, wall treatment and test cases,
but the momentum equations are integrated in vector-invariant form:

  ∂t u = −∂x K − (1/ρ) ∂x p′            − ω_y w   (centers)
  ∂t w = −∂z K − (1/ρ) ∂z p′ − (ρ−ρ_bg)g/ρ + ω_y u   (faces)
  ω_y  = ∂z u − ∂x w                                  (faces)

with (ρ, ρe) advanced in flux form. On the flat plane the prognostic
velocities are stored as orthonormal scalars: `u` on centers, `w` on faces.

Horizontal DG treatment:
  • (ρ, ρe): WeakDivergence volume + central/Rusanov interface flux
    (`add_numerical_flux_interior!`).
  • Non-conservative gradients ∂x q and the curl ∂x w: element-local strong
    Gradient completed by the *symmetric* central face lifting
    (`add_lifting_flux_interior!`), the DG analog of the CG hgrad + DSS.
  • Velocity jumps [[u]], [[w]]: λ-scaled interface penalties (λ = |u| + c),
    applied through the same lifting (each side relaxes toward its neighbor).
Vertical FD: mass/energy via face mass flux (Lin–van Leer for h_tot);
reflecting slip walls (w = 0, ∂z p′ = 0, ∂z u = 0 ⇒ ω_y = 0 at walls).
Stabilization: Nc=3 cutoff filter on tendencies; optional κ₂ (LDG horizontal +
FD vertical, velocity & enthalpy only — never ρ) for the Straka case.

Run:
  CASE=dc  julia --project=.buildkite examples/hybrid/plane/vector_invariant_2d_dg_fd.jl
  CASE=rtb julia --project=.buildkite examples/hybrid/plane/vector_invariant_2d_dg_fd.jl

Environment: CASE=dc|rtb, HELEM, VELEM, NPOLY, DT, T_END, KAPPA2, FILTER
=#

using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies
import ClimaCore.Geometry: ⊗

using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const FT = Float64

# ---------------------------------------------------------------------------
# Case selection: Straka density current (dc) or rising thermal bubble (rtb)
# ---------------------------------------------------------------------------
const case = get(ENV, "CASE", "dc")
function case_defaults(case)
    if case == "dc"
        # xlim, zlim, helem, velem, dt, t_end, κ₂
        return ((-25600.0, 25600.0), (0.0, 6400.0), 64, 64, 0.0125, 900.0, 75.0)
    else
        return ((0.0, 20000.0), (0.0, 10000.0), 40, 40, 0.05, 1000.0, 0.0)
    end
end
const (xlim0, zlim0, helem0, velem0, dt0, t_end0, κ₂0) = case_defaults(case)

function hvspace_2D(xlim, zlim, helem, velem, npoly)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    context = ClimaComms.context()
    device = ClimaComms.device(context)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(device, horzmesh)
    quad = Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

const helem = parse(Int, get(ENV, "HELEM", string(helem0)))
const velem = parse(Int, get(ENV, "VELEM", string(velem0)))
const npoly = parse(Int, get(ENV, "NPOLY", "3"))
const Δt = parse(FT, get(ENV, "DT", string(dt0)))
hv_center_space, hv_face_space = hvspace_2D(xlim0, zlim0, helem, velem, npoly)

# ---------------------------------------------------------------------------
# Physics (ρe / total energy)
# ---------------------------------------------------------------------------
const MSLP = 1e5
const grav = 9.8
const R_d = 287.058
const γ = 1.4
const C_p = R_d * γ / (γ - 1)
const θ₀ = 300.0

Φ(z) = grav * z

background_pressure(z) = MSLP * (1 - Φ(z) / (C_p * θ₀))^(C_p / R_d)
background_density(z) =
    background_pressure(z) / (R_d * θ₀ * (1 - Φ(z) / (C_p * θ₀)))

pressure(ρ, ρe, K, z) = (γ - 1) * (ρe - ρ * K - ρ * Φ(z))

function init_state(x, z)
    if case == "dc"
        x_c, z_c, r_c = 0.0, 3000.0, 1.0
        x_r, z_r = 4000.0, 2000.0
        θ_c = -15.0
        r = sqrt((x - x_c)^2 / x_r^2 + (z - z_c)^2 / z_r^2)
        θ_p = r < r_c ? 0.5 * θ_c * (1 + cospi(r / r_c)) : 0.0
        θ = θ₀ + θ_p
        π_exn = 1 - Φ(z) / (C_p * θ)
        T = π_exn * θ
        p = MSLP * π_exn^(C_p / R_d)
        ρ = p / (R_d * T)
        ρe = p / (γ - 1) + ρ * Φ(z)
        return (ρ = ρ, ρe = ρe)
    else
        xc, zc, xr, zr = 10000.0, 2000.0, 2000.0, 2000.0
        θ_pert = 2.0
        L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
        δθ = L ≤ 1 ? 0.5 * θ_pert * (1 + cospi(L)) : 0.0
        p_bg = background_pressure(z)
        Π = (p_bg / MSLP)^(R_d / C_p)
        T = (θ₀ + δθ) * Π
        ρ = p_bg / (R_d * T)
        ρe = p_bg / (γ - 1) + ρ * Φ(z)
        return (ρ = ρ, ρe = ρe)
    end
end

# ---------------------------------------------------------------------------
# Initial state: (ρ, ρe) centers; u scalar centers; w scalar faces
# ---------------------------------------------------------------------------
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_state(coord.x, coord.z), coords)
Y = Fields.FieldVector(
    Yc = Yc,
    u = map(_ -> FT(0), coords),
    w = map(_ -> FT(0), face_coords),
)

const mass_0 = sum(Y.Yc.ρ)
const energy_0 = sum(Y.Yc.ρe)

# ---------------------------------------------------------------------------
# Horizontal DG interface functions
# ---------------------------------------------------------------------------

# Flux-form interface flux for the (ρ, ρe) subsystem: central + Rusanov.
# The state carries u (orthonormal), full p (energy flux) and λ = |u| + c.
function rusanov_scalars(normal, (y⁻,), (y⁺,))
    n = normal.u
    λ = max(y⁻.λ, y⁺.λ)
    Fρ = ((y⁻.ρ * y⁻.u + y⁺.ρ * y⁺.u) / 2) * n
    Fρe = ((y⁻.u * (y⁻.ρe + y⁻.p) + y⁺.u * (y⁺.ρe + y⁺.p)) / 2) * n
    return (
        ρ = Fρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = Fρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
    )
end

# Symmetric central lifting completing the strong-form DG x-derivative:
# both sides add ((q* − q_side) n̂_side) with central q*.
grad_lift(normal, (q⁻,), (q⁺,)) = ((q⁺ - q⁻) / 2) * normal.u

# λ-scaled interface penalty (each side relaxes toward its neighbor).
diss_lift(normal, (q⁻, λ⁻), (q⁺, λ⁺)) = max(λ⁻, λ⁺) / 2 * (q⁺ - q⁻)

"""
DG x-derivative of a scalar Field `q` (center or face space): element-local
strong Gradient + symmetric central face lifting. Returns a scalar Field.
"""
function dg_ddx(q)
    hgrad = Operators.Gradient()
    lgeom = Fields.local_geometry_field(axes(q))
    r = similar(q)
    @. r =
        Geometry.transform(Geometry.UAxis(), hgrad(q)).components.data.:1 *
        lgeom.WJ
    Operators.add_lifting_flux_interior!(grad_lift, r, q)
    return r ./ lgeom.WJ
end

"""
λ-weighted interface jump dissipation for a scalar Field `q`; returns the
tendency contribution (already WJ-normalized).
"""
function dg_jump_dissipation(q, λ)
    lgeom = Fields.local_geometry_field(axes(q))
    r = similar(q)
    r .= 0
    Operators.add_lifting_flux_interior!(diss_lift, r, q, λ)
    return r ./ lgeom.WJ
end

# LDG / SIPG Laplacian tendency: provided by ClimaCore
# (`Operators.ldg_laplacian_tendency`).
const ldg_laplacian_tendency = Operators.ldg_laplacian_tendency

const κ₂ = parse(FT, get(ENV, "KAPPA2", string(κ₂0)))

# ---------------------------------------------------------------------------
# RHS
# ---------------------------------------------------------------------------
function rhs!(dY, Y, _, t)
    ρ = Y.Yc.ρ
    ρe = Y.Yc.ρe
    u = Y.u
    w = Y.w
    dρ = dY.Yc.ρ
    dρe = dY.Yc.ρe
    du = dY.u
    dw = dY.w
    z = coords.z

    hwdiv = Operators.WeakDivergence()
    lgeom_c = Fields.local_geometry_field(axes(ρ))

    vdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    VanLeer = Operators.LinVanLeerC2F(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
        constraint = Operators.MonotoneLocalExtrema(),
    )
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Ic = Operators.InterpolateF2C()
    # Reflecting slip walls: ∂z q|wall = 0 for q ∈ {p′, u, K, h_tot};
    # ω_y|wall = 0 follows from ∂z u = 0 and w = 0.
    ∂f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
        top = Operators.SetGradient(Geometry.WVector(FT(0))),
    )
    ∂c = Operators.GradientF2C()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(FT(0)),
        top = Operators.SetValue(FT(0)),
    )

    # --- Diagnostics ---
    w_c = @. Ic(w)
    K = @. (u^2 + w_c^2) / 2
    p = @. pressure(ρ, ρe, K, z)
    p′ = @. p - background_pressure(z)
    ρ_bg = @. background_density(z)
    h_tot = @. (ρe + p) / ρ
    ρ_floor = @. max(ρ, ρ_bg / 10)
    c_snd = @. sqrt(γ * max(p, background_pressure(z) / 10) / ρ_floor)
    λ_c = @. abs(u) + c_snd
    λ_f = @. If(λ_c)
    u_f = @. If(u)
    ρ_f = @. If(ρ)

    # --- (ρ, ρe): horizontal flux-form DG ---
    y = map(
        (ρi, ρei, ui, pi, λi) -> (; ρ = ρi, ρe = ρei, u = ui, p = pi, λ = λi),
        ρ,
        ρe,
        u,
        p,
        λ_c,
    )
    Fh = map(
        yi -> (
            ρ = Geometry.UVector(yi.ρ * yi.u),
            ρe = Geometry.UVector(yi.u * (yi.ρe + yi.p)),
        ),
        y,
    )
    dy_mw = @. hwdiv(Fh) * (-(lgeom_c.WJ))
    Operators.add_numerical_flux_interior!(rusanov_scalars, dy_mw, y)
    @. dρ = dy_mw.ρ / lgeom_c.WJ
    @. dρe = dy_mw.ρe / lgeom_c.WJ

    # --- (ρ, ρe): vertical FD ---
    w_vec = @. Geometry.WVector(w)
    @. dρ -= vdivf2c(ρ_f * w_vec)
    @. dρe -= vdivf2c(ρ_f * VanLeer(w_vec, h_tot, Δt))

    # --- Vorticity ω_y = ∂z u − ∂x w (faces) ---
    ∂zu = @. Geometry.transform(Geometry.WAxis(), ∂f(u)).components.data.:1
    ∂xw = dg_ddx(w)
    ω_y = @. ∂zu - ∂xw

    # --- Horizontal momentum (centers): −∂x K − (1/ρ)∂x p′ − ω_y w ---
    ∂xK = dg_ddx(K)
    ∂xp = dg_ddx(p′)
    @. du = -∂xK - ∂xp / ρ - Ic(ω_y * w)
    du .+= dg_jump_dissipation(u, λ_c)

    # --- Vertical momentum (faces): −∂z K − (1/ρ)∂z p′ + buoyancy + ω_y u ---
    ∂zK = @. Geometry.transform(Geometry.WAxis(), ∂f(K)).components.data.:1
    ∂zp = @. Geometry.transform(Geometry.WAxis(), ∂f(p′)).components.data.:1
    buoy = @. If(-(ρ - ρ_bg) * grav) / ρ_f
    @. dw = -∂zK - ∂zp / ρ_f + buoy + ω_y * u_f
    dw .+= dg_jump_dissipation(w, λ_f)
    @. dw = B(dw)

    # --- Optional κ₂ diffusion (velocity + enthalpy only; never ρ) ---
    if κ₂ != 0
        τ_κ₂ = Operators.ldg_penalty_parameter(κ₂, axes(ρ))
        τ_κ₂f = Operators.ldg_penalty_parameter(κ₂, axes(w))
        du .+= ldg_laplacian_tendency(u, nothing, κ₂, τ_κ₂)
        dw .+= ldg_laplacian_tendency(w, nothing, κ₂, τ_κ₂f)
        dρe .+= ldg_laplacian_tendency(h_tot, ρ, κ₂, τ_κ₂)

        vdivc2f_diff = Operators.DivergenceC2F(
            bottom = Operators.SetDivergence(FT(0)),
            top = Operators.SetDivergence(FT(0)),
        )
        @. du += vdivf2c(κ₂ * ∂f(u))
        @. dw += vdivc2f_diff(κ₂ * ∂c(w))
        @. dρe += vdivf2c(κ₂ * (ρ_f * ∂f(h_tot)))
        @. dw = B(dw)
    end

    # --- Element-local cutoff filter (Nc=3) on all tendencies ---
    if parse(Int, get(ENV, "FILTER", "3")) > 0
        M = Quadratures.cutoff_filter_matrix(
            FT,
            Spaces.quadrature_style(axes(ρ)),
            parse(Int, get(ENV, "FILTER", "3")),
        )
        for f in (dρ, dρe, du, dw)
            data = Fields.field_values(f)
            Operators.tensor_product!(data, data, M)
        end
        @. dw = B(dw)
    end

    return dY
end

# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
dY = similar(Y)
rhs!(dY, Y, nothing, 0.0)
@info "Initial RHS" max_dρ = maximum(abs, parent(dY.Yc.ρ)) max_du =
    maximum(abs, parent(dY.u)) max_dw = maximum(abs, parent(dY.w))

const t_end = parse(FT, get(ENV, "T_END", string(t_end0)))
prob = ODEProblem(rhs!, Y, (0.0, t_end))

sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = collect(0.0:parse(FT, get(ENV, "SAVEAT", "50.0")):t_end),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "vector_invariant_dg_fd_$(case)"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

u_ok = filter(
    u -> all(isfinite, parent(Fields.field_values(u.Yc.ρ))),
    sol.u,
)
if isempty(u_ok)
    @error "No finite solution frames to plot" retcode = sol.retcode
else
    function potential_temperature(ρ, ρe, u, w, z)
        K = (u^2 + w^2) / 2
        p = pressure(ρ, ρe, K, z)
        T = p / (ρ * R_d)
        return T * (MSLP / p)^((γ - 1) / γ)
    end

    Ic_out = Operators.InterpolateF2C()
    anim = Plots.@animate for s in u_ok
        w_c = @. Ic_out(s.w)
        θ = @. potential_temperature(s.Yc.ρ, s.Yc.ρe, s.u, w_c, coords.z)
        Plots.plot(θ .- θ₀; title = "θ′ (K)", colorbar = true)
    end
    Plots.mp4(anim, joinpath(path, "theta.mp4"), fps = 10)

    anim = Plots.@animate for s in u_ok
        Plots.plot(s.w; title = "w (m/s)", colorbar = true)
    end
    Plots.mp4(anim, joinpath(path, "w.mp4"), fps = 10)

    Es = [sum(s.Yc.ρe) for s in u_ok]
    Mass = [sum(s.Yc.ρ) for s in u_ok]
    Plots.png(
        Plots.plot((Es .- energy_0) ./ energy_0; ylabel = "Δ∫ρe / ∫ρe₀"),
        joinpath(path, "energy_budget.png"),
    )
    Plots.png(
        Plots.plot((Mass .- mass_0) ./ mass_0; ylabel = "Δ∫ρ / ∫ρ₀"),
        joinpath(path, "mass_budget.png"),
    )
end

@info "Vector-invariant DG+FD ($case) finished" path = path retcode =
    sol.retcode
