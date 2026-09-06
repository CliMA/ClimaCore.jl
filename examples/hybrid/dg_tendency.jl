# Flux-form horizontal tendency for the discontinuous-Galerkin (DG) form of the
# staggered nonhydrostatic model.
#
# A DG element sees its neighbours only through a numerical flux, and a
# numerical flux is a flux of a conserved quantity, so the DG form carries
# horizontal momentum `ρuₕ` and writes its equation in flux form. The CG form
# carries velocity `uₕ` in the vector-invariant form, which needs no interface
# flux because DSS makes the field continuous.
#
# What the two forms share: the vertical finite differences, the implicit
# split, the Jacobian, the initial condition, and the vertical momentum
# equation, which stays vector-invariant here (a face lift supplies the
# element coupling its one horizontal derivative needs).
#
# On a curved space `∇·(ρu⊗u)` carries Christoffel terms that a spectral
# divergence omits. Both horizontal assemblies below remove them the same way,
# by carrying the momentum in the global Cartesian basis, where the connection
# vanishes: `Operators.cartesian_tensor_divergence!` rotates the momentum axis
# of the flux tensor, and the two-point fluxes carry Cartesian momentum
# components directly. The vertical flux divergence drops the same terms and is
# rotated the same way.
#
# `DG_FLUX` selects the horizontal assembly (see `dg_flux_scheme`). The
# two-point volume fluxes and the Roe interface flux are ported from the
# `as/ranocha-flux` branch, which developed them against this model.

import LinearAlgebra
import ClimaCore: Fields, Geometry, Grids, Operators, Spaces
import ClimaCore.Geometry: ⊗

##
## Physical fluxes
##

# One definition of each flux feeds both the element-local weak-form volume
# term and the interface numerical flux, which is what makes the two mutually
# consistent. `u` is the full 3D velocity in the local orthonormal basis: the
# horizontal divergence contracts its first two contravariant components, and
# the vertical one its third.
dg_thermo_flux(ρ, ρe, u, pres) = (; ρ = ρ * u, ρe = (ρe + pres) * u)

# Momentum flux `ρu⊗u + p𝟙`, transport axis first. Carrying the pressure in
# the flux keeps the horizontal pressure gradient conservative, which is what
# lets the interface flux close the momentum budget across an element face.
dg_momentum_flux(ρ, u, pres) = (ρ * u) ⊗ u + pres * LinearAlgebra.I

# The same without pressure, for the vertical flux divergence: the vertical
# pressure gradient belongs to the `w` equation, which the implicit solver
# owns. On a grid with topography that leaves out the part of the horizontal
# pressure gradient that the vertical coordinate carries, the same gap the CG
# form has in its `gradₕ(ᶜp)`.
dg_momentum_transport(ρ, u) = (ρ * u) ⊗ u

# Fastest signal speed, sound plus advection, setting the size of the
# interface penalty below.
dg_wavespeed(ρ, u, pres) = sqrt(γ * pres / ρ) + norm(u)

# Coriolis on the horizontal momentum, `-f k̂ × ρuₕ`, in the local (u, v)
# basis — the traditional approximation, as in the CG form.
dg_coriolis(f, ρuₕ) = Geometry.UVVector(
    f * ρuₕ.components.data.:2,
    -f * ρuₕ.components.data.:1,
)

##
## Interface fluxes for the weak-form assembly
##
# These two go with `wdivₕ` and `cartesian_tensor_divergence!`, which complete
# the mass/energy and the momentum interfaces separately. The
# flux-differencing assembly further down completes the whole state at once and
# uses the fluxes in the section after it instead.

# Rusanov (local Lax-Friedrichs): the two sides' fluxes averaged, plus a jump
# penalty scaled by the fastest signal speed. The penalty is what makes a DG
# transport scheme stable — a central flux adds no dissipation and the
# grid-scale energy the flow feeds it has nowhere to go.
function dg_thermo_numflux(normal, (y⁻, u⁻, p⁻, λ⁻), (y⁺, u⁺, p⁺, λ⁺))
    F⁻ = dg_thermo_flux(y⁻.ρ, y⁻.ρe, u⁻, p⁻)
    F⁺ = dg_thermo_flux(y⁺.ρ, y⁺.ρe, u⁺, p⁺)
    λ = max(λ⁻, λ⁺)
    return (;
        ρ = ((F⁻.ρ + F⁺.ρ) / 2)' * normal + λ / 2 * (y⁻.ρ - y⁺.ρ),
        ρe = ((F⁻.ρe + F⁺.ρe) / 2)' * normal + λ / 2 * (y⁻.ρe - y⁺.ρe),
    )
end

# The momentum interface flux `cartesian_tensor_divergence!` calls. Its first
# face argument is the flux tensor with the momentum axis already rotated into
# the global Cartesian basis, so the penalty is taken on the Cartesian momentum
# `ρ * uc` built from the two arguments passed alongside it: both sides of a
# face are then subtracted in one basis.
dg_momentum_numflux(normal, (T⁻, ρ⁻, uc⁻, λ⁻), (T⁺, ρ⁺, uc⁺, λ⁺)) =
    ((T⁻ + T⁺) / 2)' * normal +
    (max(λ⁻, λ⁺) / 2) * (ρ⁻ * uc⁻ - ρ⁺ * uc⁺)

##
## The vertical momentum equation
##

# `curlₕ(ᶠw)` is the one horizontal derivative left in the vector-invariant
# vertical momentum equation, and its element-local value on a DG space misses
# the coupling across faces. The strong-form correction is
# `(1/J) ∮ n̂ × (w* - w)` with a central `w*`, and `n̂ × ê₃ = (n_v, -n_u)` for a
# horizontal `n̂`, which is the whole of it.
dg_w_curl_lift(normal, (w⁻,), (w⁺,)) =
    ((w⁺ - w⁻) / 2) *
    Geometry.UVVector(normal.components.data.:2, -normal.components.data.:1)

##
## Two-point volume fluxes
##

# `add_flux_differencing_divergence!` replaces the weak-form volume term with a
# collocation derivative of a symmetric two-point flux (Souza et al. 2023).
# Which two-point flux is used fixes the scheme's conservation properties, and
# it is what removes the aliasing error of the quadratic products — the job
# `Operators.SplitDivergence` does for the CG form.
#
# Both fluxes below carry the momentum in global Cartesian components, so that
# the basis the flux differencing works in is spatially constant and no
# curvature source terms appear. That is the same reason
# `cartesian_tensor_divergence!` rotates the momentum axis.
#
# The reference implementation on `as/ranocha-flux` carries a separate
# momentum pressure `pm`, so that a stratified split `p - p_ref` can make the
# scheme well balanced over topography. This case has neither topography nor a
# reference state, so `pm ≡ p` and the field is dropped.

"""
    dg_kennedy_gruber_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux for `(ρ, ρe, ρu⃗)` with Cartesian momentum:
arithmetic means of the primitives, each node's own metric vector. Kinetic
energy- and pressure-equilibrium-preserving, but not entropy-conservative.
"""
function dg_kennedy_gruber_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    return (;
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn + p̄ * Ē1n,
        ρu2 = ρ̄ * ū2 * ūn + p̄ * Ē2n,
        ρu3 = ρ̄ * ū3 * ūn + p̄ * Ē3n,
    )
end

"""
    dg_log_mean(x, y)

Numerically stable logarithmic mean `(x - y) / (log x - log y)` (Ismail & Roe
2009), the building block of the entropy-conservative flux: it is what makes
Tadmor's condition hold exactly. Switches to the Taylor series in
`f² = ((x-y)/(x+y))²` near `x == y`, where the quotient is `0/0`.
"""
@inline function dg_log_mean(x, y)
    ε = oftype(x, 1e-4)
    f² = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
    return f² < ε ?
           (x + y) / (2 + f² * (2 / 3 + f² * (2 / 5 + f² * 2 / 7))) :
           (y - x) / log(y / x)
end

"""
    dg_ranocha_flux(nvec_a, nvec_b, y_a, y_b)

Ranocha (2018, 2020) two-point flux, the entropy-conservative counterpart of
[`dg_kennedy_gruber_flux`](@ref): simultaneously entropy-conservative, kinetic
energy-preserving and pressure-equilibrium-preserving, so paired with a
dissipative interface flux it gives a discrete entropy inequality that
Kennedy-Gruber cannot. It differs in three places — the mass flux uses the
logarithmic mean of `ρ`, the internal energy `1/((γ-1)(ρ/p)ˡⁿ)`, and the
pressure work the cross term `½(pₐ u_{n,b} + p_b u_{n,a})`. The geopotential,
which `ρe` carries and which is single-valued at a shared node, rides along as
a passive potential.
"""
function dg_ranocha_flux(nvec_a, nvec_b, y_a, y_b)
    γd = oftype(y_a.ρ, γ)
    ρln = dg_log_mean(y_a.ρ, y_b.ρ)
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    mn = ρln * ūn
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    p̄ = (y_a.p + y_b.p) / 2
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    e_int = 1 / (dg_log_mean(y_a.ρ / y_a.p, y_b.ρ / y_b.p) * (γd - 1))
    K̃ = (y_a.u1 * y_b.u1 + y_a.u2 * y_b.u2 + y_a.u3 * y_b.u3) / 2
    una = y_a.uv' * nvec_a
    unb = y_b.uv' * nvec_b
    pv = (y_a.p * unb + y_b.p * una) / 2
    Φa = dg_geopotential(y_a, γd)
    Φb = dg_geopotential(y_b, γd)
    return (;
        ρ = mn,
        ρe = mn * (K̃ + e_int + (Φa + Φb) / 2) + pv,
        ρu1 = mn * ū1 + p̄ * Ē1n,
        ρu2 = mn * ū2 + p̄ * Ē2n,
        ρu3 = mn * ū3 + p̄ * Ē3n,
    )
end

# `ρe` here is total energy including the geopotential, so `Φ = e - e_int - K`
# recovers it from the state without carrying it separately.
@inline dg_geopotential(y, γd) =
    y.e - y.p / ((γd - 1) * y.ρ) - (y.u1^2 + y.u2^2 + y.u3^2) / 2

##
## Interface fluxes for the flux-differencing assembly
##
# These complete the whole state at once, so each returns all five components.
# Both are built on the Kennedy-Gruber central flux, which at `nvec_a ==
# nvec_b` and a single state is the consistent physical flux — so they pair
# with either two-point volume flux.

"""
    dg_rusanov(normal, argvals⁻, argvals⁺)

Rusanov (local Lax-Friedrichs) interface flux: the Kennedy-Gruber central flux
plus a jump penalty at the fastest signal speed `λ = c + |u|`, uniformly across
the acoustic, entropy and shear waves.
"""
function dg_rusanov(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = dg_kennedy_gruber_flux(normal, normal, y⁻, y⁺)
    return (;
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
        ρu1 = F.ρu1 - λ / 2 * (y⁺.ρ * y⁺.u1 - y⁻.ρ * y⁻.u1),
        ρu2 = F.ρu2 - λ / 2 * (y⁺.ρ * y⁺.u2 - y⁻.ρ * y⁻.u2),
        ρu3 = F.ρu3 - λ / 2 * (y⁺.ρ * y⁺.u3 - y⁻.ρ * y⁻.u3),
    )
end

"""
    dg_roe(normal, argvals⁻, argvals⁺)

Roe interface flux: the Kennedy-Gruber central flux plus wave-selective
dissipation. Acoustic waves are damped at `|ûₙ ± ĉ|`, but entropy and shear
jumps only at `max(|ûₙ|, ĉ/20)`, so a balanced jet — whose contact and shear
jumps sit at `uₙ ≈ 0` — feels a small fraction of what Rusanov's uniform
`|u| + c` applies to it.

The floor on the entropy speed is not optional: with pure `|ûₙ|`, density
jumps in a near-stagnant column are undamped and its minimum density can
drain unchecked. The energy eigen-component uses `B = Ĥ - ĉ²/(γ-1)`, which
absorbs the geopotential and vertical-kinetic parts of `ρe` without needing
them separately, `Φ` being single-valued at a face.
"""
function dg_roe(normal, (y⁻,), (y⁺,))
    F = dg_kennedy_gruber_flux(normal, normal, y⁻, y⁺)
    γd = oftype(y⁻.ρ, γ)
    # the face normal in Cartesian components; `Eᶜ` is single-valued here
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    # Roe-averaged state
    s⁻ = sqrt(y⁻.ρ)
    s⁺ = sqrt(y⁺.ρ)
    ρ̂ = s⁻ * s⁺
    a⁻ = s⁻ / (s⁻ + s⁺)
    a⁺ = 1 - a⁻
    û1 = a⁻ * y⁻.u1 + a⁺ * y⁺.u1
    û2 = a⁻ * y⁻.u2 + a⁺ * y⁺.u2
    û3 = a⁻ * y⁻.u3 + a⁺ * y⁺.u3
    Ĥ = a⁻ * (y⁻.e + y⁻.p / y⁻.ρ) + a⁺ * (y⁺.e + y⁺.p / y⁺.ρ)
    ĉ = a⁻ * sqrt(γd * y⁻.p / y⁻.ρ) + a⁺ * sqrt(γd * y⁺.p / y⁺.ρ)
    ûn = û1 * n1 + û2 * n2 + û3 * n3
    # jumps and wave amplitudes
    Δρ = y⁺.ρ - y⁻.ρ
    Δp = y⁺.p - y⁻.p
    Δu1 = y⁺.u1 - y⁻.u1
    Δu2 = y⁺.u2 - y⁻.u2
    Δu3 = y⁺.u3 - y⁻.u3
    Δun = Δu1 * n1 + Δu2 * n2 + Δu3 * n3
    α₊ = (Δp + ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₋ = (Δp - ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₀ = Δρ - Δp / ĉ^2
    s₊ = abs(ûn + ĉ)
    s₋ = abs(ûn - ĉ)
    s₀ = max(abs(ûn), ĉ / 20)   # the Harten-type floor of the docstring
    Δut1 = Δu1 - Δun * n1
    Δut2 = Δu2 - Δun * n2
    Δut3 = Δu3 - Δun * n3
    B = Ĥ - ĉ^2 / (γd - 1)
    Dρ = s₊ * α₊ + s₋ * α₋ + s₀ * α₀
    Dρu1 =
        s₊ * α₊ * (û1 + ĉ * n1) + s₋ * α₋ * (û1 - ĉ * n1) +
        s₀ * (α₀ * û1 + ρ̂ * Δut1)
    Dρu2 =
        s₊ * α₊ * (û2 + ĉ * n2) + s₋ * α₋ * (û2 - ĉ * n2) +
        s₀ * (α₀ * û2 + ρ̂ * Δut2)
    Dρu3 =
        s₊ * α₊ * (û3 + ĉ * n3) + s₋ * α₋ * (û3 - ĉ * n3) +
        s₀ * (α₀ * û3 + ρ̂ * Δut3)
    Dρe =
        s₊ * α₊ * (Ĥ + ĉ * ûn) + s₋ * α₋ * (Ĥ - ĉ * ûn) +
        s₀ * (α₀ * B + ρ̂ * (û1 * Δut1 + û2 * Δut2 + û3 * Δut3))
    return (;
        ρ = F.ρ - Dρ / 2,
        ρe = F.ρe - Dρe / 2,
        ρu1 = F.ρu1 - Dρu1 / 2,
        ρu2 = F.ρu2 - Dρu2 / 2,
        ρu3 = F.ρu3 - Dρu3 / 2,
    )
end

# The Ranocha central flux with the dissipation of one of the two interface
# fluxes above, recovered as `F_dissipative - F_KG_central` (one extra
# Kennedy-Gruber evaluation). This keeps the tested wave-selective penalties
# verbatim while making the volume/interface central pair
# entropy-conservative. The dissipation is in conserved rather than entropy
# variables, so this is entropy-stable in the sense of an EC volume flux plus
# a positive dissipation, not a certified entropy-variable dissipation matrix.
for (ranocha, dissipative) in
    ((:dg_ranocha_rusanov, :dg_rusanov), (:dg_ranocha_roe, :dg_roe))
    @eval function $ranocha(normal, (y⁻,), (y⁺,))
        Fr = dg_ranocha_flux(normal, normal, y⁻, y⁺)
        Fkg = dg_kennedy_gruber_flux(normal, normal, y⁻, y⁺)
        Fd = $dissipative(normal, (y⁻,), (y⁺,))
        return (;
            ρ = Fr.ρ + (Fd.ρ - Fkg.ρ),
            ρe = Fr.ρe + (Fd.ρe - Fkg.ρe),
            ρu1 = Fr.ρu1 + (Fd.ρu1 - Fkg.ρu1),
            ρu2 = Fr.ρu2 + (Fd.ρu2 - Fkg.ρu2),
            ρu3 = Fr.ρu3 + (Fd.ρu3 - Fkg.ρu3),
        )
    end
end

##
## Flux schemes
##

# `DG_FLUX` selects the horizontal assembly; see `dg_flux_scheme` below.
const dg_flux_name = get(ENV, "DG_FLUX", "kg-roe")

"""
    dg_flux_scheme(name)

The horizontal assembly named by `DG_FLUX`, as
`(; volume2pt, numflux)`.

  - `"rusanov"`: the weak-form volume divergence — `wdivₕ` for mass and
    energy, `cartesian_tensor_divergence!` for the momentum — with a Rusanov
    interface flux on each. `volume2pt` is `nothing`, which is what selects
    this assembly.
  - `"kg-rusanov"`, `"kg-roe"`, `"ranocha-rusanov"`, `"ranocha-roe"`:
    flux differencing with the named two-point volume flux and interface flux,
    over the whole state at once.

The weak form is the plain reading of the equations; flux differencing is what
removes the aliasing error of the quadratic products, which is why the CG form
of this model uses `Operators.SplitDivergence` for its own volume terms.
"""
function dg_flux_scheme(name)
    name == "rusanov" &&
        return (; volume2pt = nothing, numflux = dg_thermo_numflux)
    volume2pt, numflux = if name == "kg-rusanov"
        dg_kennedy_gruber_flux, dg_rusanov
    elseif name == "kg-roe"
        dg_kennedy_gruber_flux, dg_roe
    elseif name == "ranocha-rusanov"
        dg_ranocha_flux, dg_ranocha_rusanov
    elseif name == "ranocha-roe"
        dg_ranocha_flux, dg_ranocha_roe
    else
        error("DG_FLUX must be one of \"rusanov\", \"kg-rusanov\", \
               \"kg-roe\", \"ranocha-rusanov\", \"ranocha-roe\"; got \
               $(repr(name))")
    end
    return (; volume2pt, numflux)
end

##
## Cache
##

dg_cache(ᶜlocal_geometry, ᶠlocal_geometry, ᶜf) = dg_cache(
    Spaces.discretization(axes(ᶜlocal_geometry)),
    ᶜlocal_geometry,
    ᶠlocal_geometry,
    ᶜf,
)

dg_cache(::Grids.CG, ᶜlocal_geometry, ᶠlocal_geometry, ᶜf) = (;)

function dg_cache(::Grids.DG, ᶜlocal_geometry, ᶠlocal_geometry, ᶜf)
    UV = Geometry.UVVector{FT}
    UVW = Geometry.UVWVector{FT}
    space = axes(ᶜlocal_geometry)
    scheme = dg_flux_scheme(dg_flux_name)
    # The flux tensor's type, taken from the flux itself so the scratch and
    # the boundary value cannot drift from what the broadcast produces. The
    # vertical flux divergence uses it whichever horizontal assembly runs.
    Tensor = typeof(dg_momentum_transport(zero(FT), zero(UVW)))
    return (;
        ᶜfscalar = ᶜf,
        ᶜuₕ = similar(ᶜlocal_geometry, UV),
        ᶜu = similar(ᶜlocal_geometry, UVW),
        ᶜuc = similar(ᶜlocal_geometry, UVW),
        ᶜλ = similar(ᶜlocal_geometry, FT),
        dg_horizontal_cache(scheme.volume2pt, ᶜlocal_geometry, scheme)...,
        ᶠu = similar(ᶠlocal_geometry, UVW),
        ᶠTc = similar(ᶠlocal_geometry, Tensor),
        ᶠwvec = similar(ᶠlocal_geometry, Geometry.WVector{FT}),
        ᶠwlift = similar(ᶠlocal_geometry, UV),
        # No flux of momentum through the top or the bottom of the domain.
        ᶜdivᵥT = Operators.DivergenceF2C(
            top = Operators.SetValue(zero(Tensor)),
            bottom = Operators.SetValue(zero(Tensor)),
        ),
    )
end

# Scratch for the weak-form assembly: the flux tensor and its Cartesian
# rotation, the divergence, and a two-component tendency for mass and energy.
function dg_horizontal_cache(::Nothing, ᶜlocal_geometry, scheme)
    UVW = Geometry.UVWVector{FT}
    Tensor = typeof(dg_momentum_transport(zero(FT), zero(UVW)))
    ᶜdYt = similar(ᶜlocal_geometry, NamedTuple{(:ρ, :ρe), Tuple{FT, FT}})
    ᶜdivT = similar(ᶜlocal_geometry, UVW)
    return (;
        ᶜT = similar(ᶜlocal_geometry, Tensor),
        ᶜTc = similar(ᶜlocal_geometry, Tensor),
        ᶜdivT,
        ᶜdYt,
        ᶜthermo_completion =
        Operators.tendency_completion(ᶜdYt; numflux = scheme.numflux),
        ᶜmomentum_completion =
        Operators.tendency_completion(ᶜdivT; numflux = dg_momentum_numflux),
        volume2pt = nothing,
    )
end

# Scratch for the flux-differencing assembly: the node state its fluxes read,
# and the mass-weighted residual they accumulate into.
function dg_horizontal_cache(volume2pt::V, ᶜlocal_geometry, scheme) where {V}
    UV = Geometry.UVVector{FT}
    ᶜfluxstate = similar(
        ᶜlocal_geometry,
        NamedTuple{
            (:ρ, :ρe, :e, :p, :λ, :uv, :u1, :u2, :u3, :E1, :E2, :E3),
            Tuple{FT, FT, FT, FT, FT, UV, FT, FT, FT, UV, UV, UV},
        },
    )
    # `E1`, `E2`, `E3` — the tangential projections of the global Cartesian
    # unit vectors — depend on position but not on the state, so they are
    # filled once here and the tendency leaves them alone.
    space = axes(ᶜlocal_geometry)
    geometry = Spaces.global_geometry(space)
    coords = Fields.coordinate_field(space)
    for (Ec, ê) in (
        (ᶜfluxstate.E1, Geometry.Cartesian123Vector(FT(1), FT(0), FT(0))),
        (ᶜfluxstate.E2, Geometry.Cartesian123Vector(FT(0), FT(1), FT(0))),
        (ᶜfluxstate.E3, Geometry.Cartesian123Vector(FT(0), FT(0), FT(1))),
    )
        # A closure over the constant vector, so that the broadcast sees only
        # fields: dotting the constructors instead would put StaticArrays'
        # broadcast style into a Field broadcast.
        tangent(geom, coord) = Geometry.project(
            Geometry.UVAxis(),
            Geometry.LocalVector(ê, geom, coord),
        )
        Ec .= tangent.(Ref(geometry), coords)
    end
    numflux = scheme.numflux
    return (;
        ᶜfluxstate,
        ᶜresidual = similar(
            ᶜlocal_geometry,
            NamedTuple{
                (:ρ, :ρe, :ρu1, :ρu2, :ρu3),
                Tuple{FT, FT, FT, FT, FT},
            },
        ),
        volume2pt,
        numflux,
    )
end

##
## Tendency
##

function dg_remaining_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜp, ᶠω¹², ᶠu¹²) = p
    (; ᶜfscalar, ᶜuₕ, ᶜu, ᶜuc, ᶜλ) = p
    (; ᶠu, ᶠTc, ᶠwvec, ᶠwlift, ᶜdivᵥT) = p
    ᶜspace = axes(Y.c)
    ᶠspace = axes(Y.f)
    geometry = Spaces.global_geometry(ᶜspace)
    ᶜcoords = Fields.coordinate_field(ᶜspace)
    ᶠcoords = Fields.coordinate_field(ᶠspace)
    ᶠWJ = Fields.local_geometry_field(ᶠspace).WJ

    @. ᶜuₕ = Y.c.ρuₕ / ᶜρ
    @. ᶜu = Geometry.UVWVector(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw)))
    @. ᶜuc = Geometry.CartesianVector(ᶜu, geometry, ᶜcoords)
    @. ᶜK = norm_sqr(ᶜu) / 2
    @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)
    @. ᶜλ = dg_wavespeed(ᶜρ, ᶜu, ᶜp)

    # The horizontal flux divergence of mass, energy and momentum, by whichever
    # assembly `DG_FLUX` selected.
    dg_horizontal_tendency!(p.volume2pt, Yₜ, Y, p, geometry, ᶜcoords)

    # The vertical halves, which are the finite differences the CG form uses
    # (the `w` half of the mass and energy flux is implicit).
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))
    # The vertical momentum flux carries the same rotation to Cartesian as the
    # horizontal one, and is rotated back separately — which agrees with
    # rotating their sum, the rotation being linear. It is projected onto the
    # horizontal axis for the same reason as the horizontal term: its `w`
    # component is the curvature term of a 3D momentum equation, and the state
    # carries horizontal momentum only.
    @. ᶠu = Geometry.UVWVector(C123(ᶠinterp(ᶜuₕ)) + C123(ᶠw))
    @. ᶠTc = Geometry.CartesianTensor(
        dg_momentum_transport(ᶠinterp(ᶜρ), ᶠu),
        geometry,
        ᶠcoords,
    )
    @. Yₜ.c.ρuₕ -= Geometry.project(
        Geometry.UVAxis(),
        Geometry.LocalVector(ᶜdivᵥT(ᶠTc), geometry, ᶜcoords),
    )

    @. Yₜ.c.ρuₕ += dg_coriolis(ᶜfscalar, Y.c.ρuₕ)
    # `Φ` is continuous, so its horizontal gradient needs no face lift. It
    # vanishes on a grid without topography, where `z` is a function of the
    # vertical coordinate alone.
    @. Yₜ.c.ρuₕ -= ᶜρ * Geometry.project(Geometry.UVAxis(), gradₕ(ᶜΦ))

    # Vertical momentum: the vector-invariant equation of the CG form, with a
    # face lift completing the one horizontal derivative it takes. The lift
    # acts on the jump in the physical `w`, so the covariant component is
    # converted first; `.components.data.:1` of the result is a view, which is
    # what the face loop reads.
    @. ᶠwvec = Geometry.WVector(ᶠw)
    fill!(parent(ᶠwlift), zero(FT))
    Operators.add_lifting_flux_interior!(
        dg_w_curl_lift,
        ᶠwlift,
        ᶠwvec.components.data.:1,
    )
    @. ᶠω¹² = curlₕ(ᶠw)
    @. ᶠω¹² += CT12(ᶠwlift / ᶠWJ)
    @. ᶠω¹² += ᶠcurlᵥ(C12(ᶜuₕ))
    @. ᶠu¹² = CT12(ᶠinterp(ᶜuₕ))
    @. Yₜ.f.w -= ᶠω¹² × ᶠu¹²

    return Yₜ
end

##
## The two horizontal assemblies
##

# Weak form: one volume divergence per equation, each completed by its own
# interface flux. The momentum runs through `cartesian_tensor_divergence!`,
# which rotates the momentum axis of the flux tensor to Cartesian, completes
# the interfaces there, and rotates the result back.
function dg_horizontal_tendency!(::Nothing, Yₜ, Y, p, geometry, ᶜcoords)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    (; ᶜp, ᶜu, ᶜuc, ᶜλ, ᶜT, ᶜTc, ᶜdivT, ᶜdYt) = p
    (; ᶜthermo_completion, ᶜmomentum_completion) = p

    @. ᶜdYt = -wdivₕ(dg_thermo_flux(ᶜρ, ᶜρe, ᶜu, ᶜp))
    Operators.complete_tendency!(ᶜthermo_completion, ᶜdYt, Y.c, ᶜu, ᶜp, ᶜλ)
    @. Yₜ.c.ρ += ᶜdYt.ρ
    @. Yₜ.c.ρe += ᶜdYt.ρe

    @. ᶜT = dg_momentum_flux(ᶜρ, ᶜu, ᶜp)
    Operators.cartesian_tensor_divergence!(
        ᶜdivT,
        ᶜTc,
        ᶜT,
        ᶜmomentum_completion,
        ᶜρ,
        ᶜuc,
        ᶜλ,
    )
    @. Yₜ.c.ρuₕ -= Geometry.project(Geometry.UVAxis(), ᶜdivT)
    return Yₜ
end

# Flux differencing: one volume term and one interface flux over the whole
# state, both reading the node state `ᶜfluxstate` and both carrying the
# momentum in Cartesian components. `add_flux_differencing_divergence!` is
# stored in weak-equivalent form, so it stands in for `wdivₕ(F) * (-WJ)` and
# composes with the interface flux exactly as the weak form does; the residual
# it accumulates is mass-weighted, and dividing by `WJ` gives the tendency.
function dg_horizontal_tendency!(volume2pt::V, Yₜ, Y, p, geometry, ᶜcoords) where {V}
    (; ᶜp, ᶜu, ᶜuc, ᶜλ, ᶜfluxstate, ᶜresidual, numflux) = p
    ᶜWJ = Fields.local_geometry_field(axes(Y.c)).WJ

    @. ᶜfluxstate.ρ = Y.c.ρ
    @. ᶜfluxstate.ρe = Y.c.ρe
    @. ᶜfluxstate.e = Y.c.ρe / Y.c.ρ
    @. ᶜfluxstate.p = ᶜp
    @. ᶜfluxstate.λ = ᶜλ
    @. ᶜfluxstate.uv = Geometry.project(Geometry.UVAxis(), ᶜu)
    @. ᶜfluxstate.u1 = ᶜuc.components.data.:1
    @. ᶜfluxstate.u2 = ᶜuc.components.data.:2
    @. ᶜfluxstate.u3 = ᶜuc.components.data.:3

    fill!(parent(ᶜresidual), zero(FT))
    Operators.add_flux_differencing_divergence!(
        volume2pt,
        ᶜresidual,
        ᶜfluxstate,
    )
    Operators.add_numerical_flux_interior!(numflux, ᶜresidual, ᶜfluxstate)

    @. Yₜ.c.ρ += ᶜresidual.ρ / ᶜWJ
    @. Yₜ.c.ρe += ᶜresidual.ρe / ᶜWJ
    @. Yₜ.c.ρuₕ += dg_cartesian_momentum_tendency(
        ᶜresidual,
        ᶜWJ,
        geometry,
        ᶜcoords,
    )
    return Yₜ
end

# The Cartesian momentum residual, unweighted, back in the local frame and
# projected onto the horizontal.
dg_cartesian_momentum_tendency(r, WJ, geometry, coord) = Geometry.project(
    Geometry.UVAxis(),
    Geometry.LocalVector(
        Geometry.UVWVector(r.ρu1, r.ρu2, r.ρu3) / WJ,
        geometry,
        coord,
    ),
)
