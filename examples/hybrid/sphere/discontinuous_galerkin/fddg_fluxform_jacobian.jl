#=
Column-wise analytic Jacobian of the HEVI implicit tendency for the FLUX-FORM
FDDG sphere driver (state: Yc = (ρ, ρe, ρu1, ρu2, ρu3) at centers, ρw as a
Covariant3 face field). Reuses the operator matrices and thermodynamic
constants defined by sphere_dg_fd_jacobian.jl (included by the model file).

Implicit tendency (vertical acoustic subsystem; everything else explicit):
    ᶜρₜ  = −ᶜdivᵥ(ᶠρw)                       [linear in ρw]
    ᶜρeₜ = −ᶜdivᵥ(ᶠρw · ᶠinterp(ᶜh_tot))     [central; VanLeer corr. explicit]
    ᶠρwₜ = −ᶠgradᵥ(ᶜp) − ᶠinterp(ᶜρ)·ᶠgradᵥ(ᶜΦ)
    ρu_c: no implicit tendency.

Nonzero Jacobian blocks (h_tot and K frozen, the analog of the validated
:no_∂ᶜp∂ᶜK default; ∂ᶠρwₜ/∂ᶠρw ≡ 0 under frozen K since the ρw equation is
otherwise independent of ρw):
    ∂ᶜρₜ/∂ᶠρw  = −ᶜdivᵥ_matrix ⋅ Diag(g³³)
    ∂ᶜρeₜ/∂ᶠρw = −ᶜdivᵥ_matrix ⋅ Diag(ᶠinterp(ᶜh_tot)·g³³)
    ∂ᶠρwₜ/∂ᶜρe = −ᶠgradᵥ_matrix ⋅ (R_d/cv_d)
    ∂ᶠρwₜ/∂ᶜρ  = −ᶠgradᵥ_matrix ⋅ Diag(R_d(−(K+Φ)/cv_d + T_tri))
                 − Diag(ᶠgradᵥ(ᶜΦ)) ⋅ ᶠinterp_matrix
=#

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

    # ᶠρwₜ = −ᶠgradᵥ(ᶜp) − ᶠinterp(ᶜρ)·ᶠgradᵥ(ᶜΦ)
    @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = -(ᶠgradᵥ_matrix() * R_d / cv_d)
    @. ∂ᶠ𝕄ₜ∂ᶜρ =
        -(ᶠgradᵥ_matrix()) *
        DiagonalMatrixRow(R_d * (-(K + ᶜΦ) / cv_d + T_tri)) -
        DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ)) * ᶠinterp_matrix()

    I = one(∂R∂Y)
    @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
end
