#=
Column-wise analytic Jacobian of the HEVI implicit tendency for the DG-FD
sphere model (adapted from examples/hybrid/implicit_equation_jacobian.jl and
the ρe blocks of staggered_nonhydrostatic_model.jl, which cannot be included
here without pulling in the CG model). The implicit part is purely vertical
FD and column-local — the DG horizontal discretization never enters it.

State layout: (Y.Yc.ρ, Y.Yc.ρe, Y.uₕ, Y.w), 𝔼 = ρe, central implicit
vertical energy flux If(ρ)·w·If(h_tot) (ClimaAtmos `Val(:none)` transport;
the Lin–VanLeer upwind correction lives in the explicit tendency and does
not enter the Jacobian). The residual convention is the ClimaTimeSteppers
one (transform = false):

    R(Y) = Yᵖʳᵉᵛ + δtγ·Yₜ(Y) − Y,   ∂R/∂Y = δtγ·∂Yₜ/∂Y − I

with nonzero tendency blocks ∂ᶜρₜ/∂ᶠw, ∂ᶜρeₜ/∂ᶠw, ∂ᶠwₜ/∂ᶜρ, ∂ᶠwₜ/∂ᶜρe and
∂ᶠwₜ/∂ᶠw, solved per column with MatrixFields.BlockArrowheadSolve.

Flags (same semantics as the CG model):
  ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode — :no_∂ᶜp∂ᶜK (default; drops ∂p/∂K so the block is
                  bidiagonal) or :exact (quaddiagonal)
  ∂ᶠ𝕄ₜ∂ᶜρ_mode — :exact (default) or :hydrostatic_balance
=#

import LinearAlgebra: ldiv!
using ClimaCore.MatrixFields
using ClimaCore.MatrixFields: @name

const ᶜinterp_matrix = MatrixFields.operator_matrix(Ic)
const ᶠinterp_matrix = MatrixFields.operator_matrix(If)
const ᶜdivᵥ_matrix = MatrixFields.operator_matrix(vdivf2c)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)

const ᶜρ_name = @name(Yc.ρ)
const ᶜ𝔼_name = @name(Yc.ρe)
const ᶠ𝕄_name = @name(w)

struct DGImplicitEquationJacobian{TJ, RJ, F}
    ∂Yₜ∂Y::TJ # nonzero blocks of the implicit tendency's Jacobian
    ∂R∂Y::RJ # the full implicit residual's Jacobian, and its linear solver
    flags::F
end

function DGImplicitEquationJacobian(
    Y;
    ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK,
    ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact,
)
    ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode in (:no_∂ᶜp∂ᶜK, :exact) ||
        error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :no_∂ᶜp∂ᶜK or :exact")
    ∂ᶠ𝕄ₜ∂ᶜρ_mode in (:exact, :hydrostatic_balance) ||
        error("∂ᶠ𝕄ₜ∂ᶜρ_mode must be :exact or :hydrostatic_balance")

    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{typeof(CT3(FT(0))')}
    QuaddiagonalRow_ACT3 = QuaddiagonalMatrixRow{typeof(CT3(FT(0))')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(C3(FT(0)) * CT3(FT(0))')}
    ∂ᶜ𝔼ₜ∂ᶠ𝕄_Row_ACT3 =
        ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact ? QuaddiagonalRow_ACT3 : BidiagonalRow_ACT3
    ∂Yₜ∂Y = MatrixFields.FieldMatrix(
        (ᶜρ_name, ᶠ𝕄_name) => zeros(BidiagonalRow_ACT3, axes(Y.Yc)),
        (ᶜ𝔼_name, ᶠ𝕄_name) => zeros(∂ᶜ𝔼ₜ∂ᶠ𝕄_Row_ACT3, axes(Y.Yc)),
        (ᶠ𝕄_name, ᶜρ_name) => zeros(BidiagonalRow_C3, axes(Y.w)),
        (ᶠ𝕄_name, ᶜ𝔼_name) => zeros(BidiagonalRow_C3, axes(Y.w)),
        (ᶠ𝕄_name, ᶠ𝕄_name) => zeros(TridiagonalRow_C3xACT3, axes(Y.w)),
    )

    # When ∂Yₜ∂Y is sparse, one(∂Yₜ∂Y) doesn't contain every diagonal block
    # (uₕ has no implicit tendency), so build the identity over all of Y.
    I = MatrixFields.identity_field_matrix(Y)
    ∂R∂Y = FT(1) .* ∂Yₜ∂Y .- I
    alg = MatrixFields.BlockArrowheadSolve(ᶜρ_name, ᶜ𝔼_name)

    return DGImplicitEquationJacobian(
        ∂Yₜ∂Y,
        FieldMatrixWithSolver(∂R∂Y, Y, alg),
        (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode, ∂ᶠ𝕄ₜ∂ᶜρ_mode),
    )
end

Base.similar(j::DGImplicitEquationJacobian) = DGImplicitEquationJacobian(
    similar(j.∂Yₜ∂Y),
    similar(j.∂R∂Y),
    j.flags,
)
Base.zero(j::DGImplicitEquationJacobian) = DGImplicitEquationJacobian(
    zero(j.∂Yₜ∂Y),
    zero(j.∂R∂Y),
    j.flags,
)

# Called by Newton's method from ClimaTimeSteppers: solves ∂R∂Y * δY = R.
ldiv!(
    δY::Fields.FieldVector,
    j::DGImplicitEquationJacobian,
    R::Fields.FieldVector,
) = ldiv!(δY, j.∂R∂Y, R)

# Wfact for ClimaTimeSteppers: assembles ∂R∂Y = δtγ ∂Yₜ∂Y − I at state Y.
function implicit_equation_jacobian!(j::DGImplicitEquationJacobian, Y, p, δtγ, t)
    (; ∂Yₜ∂Y, ∂R∂Y, flags) = j
    ρ = Y.Yc.ρ
    ρe = Y.Yc.ρe
    uₕ = Y.uₕ
    w = Y.w

    ∂ᶜρₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜρ_name, ᶠ𝕄_name]
    ∂ᶜ𝔼ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜ𝔼_name, ᶠ𝕄_name]
    ∂ᶠ𝕄ₜ∂ᶜρ = ∂Yₜ∂Y[ᶠ𝕄_name, ᶜρ_name]
    ∂ᶠ𝕄ₜ∂ᶜ𝔼 = ∂Yₜ∂Y[ᶠ𝕄_name, ᶜ𝔼_name]
    ∂ᶠ𝕄ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶠ𝕄_name, ᶠ𝕄_name]

    uv = @. Geometry.UVVector(uₕ)
    w_c = @. Ic(Geometry.WVector(w))
    K = @. (norm_sqr(uv) + norm_sqr(w_c)) / 2
    p_thermo = @. pressure_ρe(ρe, K, ᶜΦ, ρ)
    h_tot = @. (ρe + p_thermo) / ρ

    ᶠgⁱʲ = Fields.local_geometry_field(w).gⁱʲ
    g³³(gⁱʲ) = reshape(
        gⁱʲ,
        Geometry.Contravariant3Axis(),
        Geometry.Contravariant3Axis(),
    )

    # ᶜK = norm_sqr(uv) / 2 + ACT3(Ic(w)) * Ic(w) / 2
    # ∂(ᶜK)/∂(ᶠw) = ACT3(Ic(w)) * ᶜinterp_matrix()
    ∂ᶜK∂ᶠw = @. DiagonalMatrixRow(adjoint(CT3(Ic(w)))) * ᶜinterp_matrix()

    # ᶜρₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw)
    # ∂(ᶜρₜ)/∂(ᶠw) = -ᶜdivᵥ_matrix() * ᶠinterp(ᶜρ) * ᶠg³³
    @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(If(ρ) * g³³(ᶠgⁱʲ))

    # ᶜρeₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw * ᶠinterp(ᶜh_tot))
    # (ClimaAtmos central form; the VanLeer correction is explicit)
    if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact
        # ∂(ᶜρeₜ)/∂(ᶠw) = -ᶜdivᵥ_matrix() * (
        #     ᶠinterp(ᶜρ) * ᶠinterp(ᶜh_tot) * ᶠg³³ +
        #     CT3(ᶠw) * ᶠinterp(ᶜρ) * ᶠinterp_matrix() *
        #         ∂(ᶜh_tot)/∂(ᶜK) * ∂(ᶜK)/∂(ᶠw)
        # ), with ∂(ᶜh_tot)/∂(ᶜK) = ∂(ᶜp)/∂(ᶜK) / ᶜρ = -R_d / cv_d
        @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
            -(ᶜdivᵥ_matrix()) * (
                DiagonalMatrixRow(If(ρ) * If(h_tot) * g³³(ᶠgⁱʲ)) +
                DiagonalMatrixRow(CT3(w) * If(ρ)) *
                ᶠinterp_matrix() *
                (-R_d / cv_d * ∂ᶜK∂ᶠw)
            )
    else # :no_∂ᶜp∂ᶜK — approximate ∂(ᶜh_tot)/∂(ᶜK) = 0 (ClimaAtmos default)
        @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
            -(ᶜdivᵥ_matrix()) *
            DiagonalMatrixRow(If(ρ) * If(h_tot) * g³³(ᶠgⁱʲ))
    end

    # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
    # ∂(ᶠwₜ)/∂(ᶜρe) = -1 / ᶠinterp(ᶜρ) * ᶠgradᵥ_matrix() * R_d / cv_d
    @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 =
        -DiagonalMatrixRow(1 / If(ρ)) * (ᶠgradᵥ_matrix() * R_d / cv_d)

    if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
        # ∂(ᶠwₜ)/∂(ᶜρ) =
        #     -1 / ᶠinterp(ᶜρ) * ᶠgradᵥ_matrix() *
        #         R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri) +
        #     ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2 * ᶠinterp_matrix()
        @. ∂ᶠ𝕄ₜ∂ᶜρ =
            -DiagonalMatrixRow(1 / If(ρ)) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(R_d * (-(K + ᶜΦ) / cv_d + T_tri)) +
            DiagonalMatrixRow(ᶠgradᵥ(p_thermo) / If(ρ)^2) * ᶠinterp_matrix()
    else # :hydrostatic_balance — assume ᶠgradᵥ(ᶜp)/ᶠinterp(ᶜρ) = -ᶠgradᵥ(ᶜΦ)
        # and neglect ᶜK relative to ᶜΦ
        @. ∂ᶠ𝕄ₜ∂ᶜρ =
            -DiagonalMatrixRow(1 / If(ρ)) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(R_d * (-(ᶜΦ) / cv_d + T_tri)) -
            DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ) / If(ρ)) * ᶠinterp_matrix()
    end

    # ∂(ᶠwₜ)/∂(ᶠw) = -(
    #     1 / ᶠinterp(ᶜρ) * ᶠgradᵥ_matrix() * (-ᶜρ * R_d / cv_d) +
    #     ᶠgradᵥ_matrix()
    # ) * ∂(ᶜK)/∂(ᶠw)
    @. ∂ᶠ𝕄ₜ∂ᶠ𝕄 =
        -(
            DiagonalMatrixRow(1 / If(ρ)) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(-(ρ * R_d / cv_d)) + ᶠgradᵥ_matrix()
        ) * ∂ᶜK∂ᶠw

    I = one(∂R∂Y)
    @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
end
