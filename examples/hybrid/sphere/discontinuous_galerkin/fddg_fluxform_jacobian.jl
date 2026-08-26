#=
Column-wise analytic Jacobian of the HEVI implicit tendency for the FLUX-FORM
FDDG sphere driver (state: Yc = (ρ, ρe, ρu1, ρu2, ρu3) at centers, ρw as a
Covariant3 face field). Self-contained: the vertical operator matrices and
the ρ/ρe field-name constants used below are defined here, built on the FD
operators (If, vdivf2c, ᶠgradᵥ) and thermodynamic constants (R_d, cv_d,
T_tri) from the model file.

Implicit tendency (vertical acoustic subsystem; everything else explicit).
The ρw pressure-gradient + gravity is in Exner-perturbation form (Yatunin et
al. 2026): with Π = (p/p₀)^κ, θ = T/Π, Π' = Π − Π_ref, θ' = θ − θ_ref,
    ᶜρₜ  = −ᶜdivᵥ(ᶠρw)                       [linear in ρw]
    ᶜρeₜ = −ᶜdivᵥ(ᶠρw · ᶠinterp(ᶜh_tot))     [central; VanLeer corr. explicit]
    ᶠρwₜ = −ᶠinterp(ᶜρ)·cp_d·(ᶠinterp(ᶜθ)·ᶠgradᵥ(Π') + ᶠinterp(ᶜθ')·ᶠgradᵥ(Π_ref))
    ρu_c: no implicit tendency.

Nonzero Jacobian blocks (h_tot, K, and the interpolated coefficients ᶠinterp(ρ),
ᶠinterp(θ) frozen — the analog of the validated :no_∂ᶜp∂ᶜK default; ∂ᶠρwₜ/∂ᶠρw ≡
0 under frozen K). With K frozen, ∂p/∂ρe = R_d/cv_d, ∂p/∂ρ = R_d(−(K+Φ)/cv_d +
T_tri), and ∂Π/∂p = κ Π/p; Π_ref is a fixed reference field. Define
C = ᶠinterp(ρ)·cp_d·ᶠinterp(θ) and A = cp_d(ᶠinterp(θ)ᶠgradᵥ(Π') +
ᶠinterp(θ')ᶠgradᵥ(Π_ref)) (a Covariant3 face field), so ᶠρwₜ = −ᶠinterp(ρ)·A:
    ∂ᶜρₜ/∂ᶠρw  = −ᶜdivᵥ_matrix ⋅ Diag(g³³)
    ∂ᶜρeₜ/∂ᶠρw = −ᶜdivᵥ_matrix ⋅ Diag(ᶠinterp(ᶜh_tot)·g³³)
    ∂ᶠρwₜ/∂ᶜρe = −Diag(C) ⋅ ᶠgradᵥ_matrix ⋅ Diag((κΠ/p)(R_d/cv_d))
    ∂ᶠρwₜ/∂ᶜρ  = −Diag(C) ⋅ ᶠgradᵥ_matrix ⋅ Diag((κΠ/p)R_d(−(K+Φ)/cv_d + T_tri))
                 − Diag(A) ⋅ ᶠinterp_matrix
The last term (∂/∂ρ through the ᶠinterp(ρ) prefactor) is the Exner analog of the
old −Diag(ᶠgradᵥ(Φ))·ᶠinterp gravity term; A ≈ 0 at hydrostatic balance. The
leading ∂/∂ρe block reduces to the old −ᶠgradᵥ(R_d/cv_d) since
C·κΠ/p ≈ ρ R_d T/p = 1 — a consistent refinement of the full-p Jacobian.
=#

import LinearAlgebra: ldiv!
using ClimaCore.MatrixFields
using ClimaCore.MatrixFields: @name

const ᶜdivᵥ_matrix = MatrixFields.operator_matrix(vdivf2c)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)
const ᶠinterp_matrix = MatrixFields.operator_matrix(If)

const ᶜρ_name = @name(Yc.ρ)
const ᶜ𝔼_name = @name(Yc.ρe)

const ᶠ𝕄F_name = @name(ρw)

struct FDDGImplicitEquationJacobian{TJ, RJ}
    ∂Yₜ∂Y::TJ
    ∂R∂Y::RJ
end

function FDDGImplicitEquationJacobian(Y)
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{typeof(CT3(FT(0))')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(C3(FT(0)) * CT3(FT(0))')}
    ∂Yₜ∂Y = MatrixFields.FieldMatrix(
        (ᶜρ_name, ᶠ𝕄F_name) => zeros(BidiagonalRow_ACT3, axes(Y.Yc)),
        (ᶜ𝔼_name, ᶠ𝕄F_name) => zeros(BidiagonalRow_ACT3, axes(Y.Yc)),
        (ᶠ𝕄F_name, ᶜρ_name) => zeros(BidiagonalRow_C3, axes(Y.ρw)),
        (ᶠ𝕄F_name, ᶜ𝔼_name) => zeros(BidiagonalRow_C3, axes(Y.ρw)),
        # kept (at zero) so the arrowhead structure matches the solver
        (ᶠ𝕄F_name, ᶠ𝕄F_name) => zeros(TridiagonalRow_C3xACT3, axes(Y.ρw)),
    )
    I = MatrixFields.identity_field_matrix(Y)
    ∂R∂Y = FT(1) .* ∂Yₜ∂Y .- I
    alg = MatrixFields.BlockArrowheadSolve(ᶜρ_name, ᶜ𝔼_name)
    return FDDGImplicitEquationJacobian(
        ∂Yₜ∂Y,
        FieldMatrixWithSolver(∂R∂Y, Y, alg),
    )
end

Base.similar(j::FDDGImplicitEquationJacobian) =
    FDDGImplicitEquationJacobian(similar(j.∂Yₜ∂Y), similar(j.∂R∂Y))
Base.zero(j::FDDGImplicitEquationJacobian) =
    FDDGImplicitEquationJacobian(zero(j.∂Yₜ∂Y), zero(j.∂R∂Y))

ldiv!(
    δY::Fields.FieldVector,
    j::FDDGImplicitEquationJacobian,
    R::Fields.FieldVector,
) = ldiv!(δY, j.∂R∂Y, R)

function fddg_implicit_equation_jacobian!(
    j::FDDGImplicitEquationJacobian,
    Y,
    p,
    δtγ,
    t,
)
    (; ∂Yₜ∂Y, ∂R∂Y) = j
    ρ = Y.Yc.ρ
    ρe = Y.Yc.ρe

    ∂ᶜρₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜρ_name, ᶠ𝕄F_name]
    ∂ᶜ𝔼ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜ𝔼_name, ᶠ𝕄F_name]
    ∂ᶠ𝕄ₜ∂ᶜρ = ∂Yₜ∂Y[ᶠ𝕄F_name, ᶜρ_name]
    ∂ᶠ𝕄ₜ∂ᶜ𝔼 = ∂Yₜ∂Y[ᶠ𝕄F_name, ᶜ𝔼_name]

    uE = @. (Y.Yc.ρu1 * eE1 + Y.Yc.ρu2 * eE2 + Y.Yc.ρu3 * eE3) / ρ
    uN = @. (Y.Yc.ρu1 * eN1 + Y.Yc.ρu2 * eN2 + Y.Yc.ρu3 * eN3) / ρ
    w_c = @. Ic(Geometry.WVector(Y.ρw)).components.data.:1 / ρ
    K = @. (uE^2 + uN^2 + w_c^2) / 2
    p_thermo = @. pressure_ρe(ρe, K, ᶜΦ, ρ)
    h_tot = @. (ρe + p_thermo) / ρ

    # Exner-perturbation thermodynamics (PGF=exner only; (p/p₀)^κ requires p>0).
    # K frozen ⇒ ∂p/∂ρe = R_d/cv_d, ∂p/∂ρ = R_d(−(K+Φ)/cv_d + T_tri); ∂Π/∂p = κΠ/p.
    if pgf == :exner
        Π = @. (p_thermo / p_0)^κ_gas
        θ = @. p_thermo / (ρ * R_d) / Π
        Πp = @. Π - ᶜΠ_ref
        θp = @. θ - ᶜθ_ref
        dΠ_dρe = @. κ_gas * Π / p_thermo * (R_d / cv_d)
        dΠ_dρ = @. κ_gas * Π / p_thermo * R_d * (-(K + ᶜΦ) / cv_d + T_tri)
        Cpg = @. If(ρ) * cp_d * If(θ)                       # frozen prefactor
        Apg = @. cp_d * (If(θ) * ᶠgradᵥ(Πp) + If(θp) * ᶠgradᵥ(ᶜΠ_ref))  # C3 face
    end

    ᶠgⁱʲ = Fields.local_geometry_field(Y.ρw).gⁱʲ
    g³³(gⁱʲ) = reshape(
        gⁱʲ,
        Geometry.Contravariant3Axis(),
        Geometry.Contravariant3Axis(),
    )

    # ᶜρₜ = −ᶜdivᵥ(ᶠρw)
    @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(g³³(ᶠgⁱʲ))

    # ᶜρeₜ = −ᶜdivᵥ(ᶠρw · ᶠinterp(ᶜh_tot)); h_tot frozen
    @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
        -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(If(h_tot) * g³³(ᶠgⁱʲ))

    if pgf == :exner
        # Exner-perturbation ρw (see header):
        # ᶠρwₜ = −ᶠinterp(ρ)cp_d(ᶠinterp(θ)ᶠgradᵥ(Π') + ᶠinterp(θ')ᶠgradᵥ(Π_ref))
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 =
            -DiagonalMatrixRow(Cpg) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(dΠ_dρe)
        @. ∂ᶠ𝕄ₜ∂ᶜρ =
            -DiagonalMatrixRow(Cpg) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(dΠ_dρ) -
            DiagonalMatrixRow(Apg) * ᶠinterp_matrix()
    elseif is_moist
        #   ∂p/∂ρe = R_m/cv_m = κ_m,
        #   ∂p/∂ρ  = R_m T − κ_m (ρe/ρ)   [since ∂(ρe/ρ)/∂ρ = −(ρe/ρ)/ρ].
        q_tot = @. Y.Yc.ρq_tot / ρ
        R_m = @. R_d * (1 - q_tot) + R_v * q_tot
        cv_m = @. cv_d * (1 - q_tot) + cv_v * q_tot
        κ_m = @. R_m / cv_m
        Tair = (@. moist_p_dyn(ρ, ρe / ρ - K - ᶜΦ, q_tot)).T
        dp_dρ = @. R_m * Tair - κ_m * (ρe / ρ)
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = -(ᶠgradᵥ_matrix()) * DiagonalMatrixRow(κ_m)
        @. ∂ᶠ𝕄ₜ∂ᶜρ =
            -(ᶠgradᵥ_matrix()) * DiagonalMatrixRow(dp_dρ) -
            DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ)) * ᶠinterp_matrix()
    else
        # Conservative full-p ρw: ᶠρwₜ = −ᶠgradᵥ(p) − ᶠinterp(ρ)·ᶠgradᵥ(Φ),
        # ∂p/∂ρe = R_d/cv_d, ∂p/∂ρ = R_d(−(K+Φ)/cv_d + T_tri).
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = -(ᶠgradᵥ_matrix() * R_d / cv_d)
        @. ∂ᶠ𝕄ₜ∂ᶜρ =
            -(ᶠgradᵥ_matrix()) *
            DiagonalMatrixRow(R_d * (-(K + ᶜΦ) / cv_d + T_tri)) -
            DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ)) * ᶠinterp_matrix()
    end

    I = one(∂R∂Y)
    @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
end
