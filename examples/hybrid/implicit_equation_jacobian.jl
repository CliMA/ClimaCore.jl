import LinearAlgebra: ldiv!
using ClimaCore: Spaces, Fields, Operators
using ClimaCore.Utilities: half
using ClimaCore.MatrixFields
import ClimaCore.MatrixFields: ⋆
using ClimaCore.MatrixFields: @name

"""
    ImplicitEquationJacobian(Y, transform, [flags])

Represents the Jacobian of the implicit residual, `R(Y)`, which is defined as
either `Yᵖʳᵉᵛ + δt * γ * Yₜ - Y` or `(Y - Yᵖʳᵉᵛ) / (δt * γ) - Yₜ`, with the
second version used when `transform` is `true`. In this expression, `δt` is the
timestep, `γ` is a scalar quantity determined by the timestepping scheme, `Y` is
the current approximation of the timestepper's implicit stage value, `Yₜ` is the
value obtained by calling `implicit_tendency!(Yₜ, Y, p, t)`, and `Yᵖʳᵉᵛ` is a
linear combination of the timestepper's previous stage values. The residual's
Jacobian, `∂R/∂Y`, can be expressed in terms of the tendency's Jacobian,
`∂Yₜ/∂Y`, as either `δt * γ * ∂Yₜ/∂Y - I` or `I / (δt * γ) - ∂Yₜ/∂Y`.

The `ImplicitEquationJacobian` allows the `staggered_nonhydrostatic_model.jl` 
file to be compatible with `ClimaTimeSteppers` and `OrdinaryDiffEq`. This file
defines both `implicit_tendency!` and `implicit_equation_jacobian!`, where the
latter sets `∂R/∂Y` based on the values of `Y` and `δt * γ`. An optional set of
flags can also be passed to the `ImplicitEquationJacobian` constructor, which
are then accessible from within `implicit_equation_jacobian!`. For all
timestepping schemes from `ClimaTimeSteppers` and the Rosenbrock schemes from
`OrdinaryDiffEq`, the `transform` flag should be set to `false`. For all other
timestepping schemes from `OrdinaryDiffEq`, it should be set to `true`.

Within both `ClimaTimeSteppers` and `OrdinaryDiffEq`, this data structure is
used to solve the linear equation `∂R/∂Y * δY = R`, which allows Newton's method
to iteratively find the root of `R(Y) = 0`. In `ClimaTimeSteppers`, this is done
by calling `ldiv!(δY, ∂R/∂Y, R)`, where `δY` and `R` are represeted as
`FieldVector`s and `∂R/∂Y` is represented as an `ImplicitEquationJacobian`. When
using Krylov methods from `ClimaTimeSteppers`, `δY` and `R` can also be
represented as other `AbstractVector`s (which are internally converted to
`FieldVector`s). For `OrdinaryDiffEq`, we use the `linsolve!` function instead
of `ldiv!` by passing it to the timestepping scheme's constructor.

TODO: Compatibility with `OrdinaryDiffEq` is out of date and should be updated.
"""
struct ImplicitEquationJacobian{TJ, RJ, F, V}
    ∂Yₜ∂Y::TJ # nonzero blocks of the implicit tendency's Jacobian
    ∂R∂Y::RJ # the full implicit residual's Jacobian, and its linear solver
    transform::Bool # whether this struct is used to compute Wfact or Wfact_t
    flags::F # flags for computing the implicit tendency's Jacobian
    R_field_vector::V # cache that is used to evaluate ldiv! for Krylov methods
    δY_field_vector::V # cache that is used to evaluate ldiv! for Krylov methods
end

function ImplicitEquationJacobian(Y, transform, flags = (;))
    FT = eltype(Y)

    ᶜρ_name = @name(c.ρ)
    ᶜ𝔼_name = if :ρθ in propertynames(Y.c)
        @name(c.ρθ)
    elseif :ρe in propertynames(Y.c)
        @name(c.ρe)
    elseif :ρe_int in propertynames(Y.c)
        @name(c.ρe_int)
    end
    ᶠ𝕄_name = @name(f.w)

    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    QuaddiagonalRow_ACT3 = QuaddiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(C3(FT(0)) * CT3(FT(0))')}
    ∂ᶜ𝔼ₜ∂ᶠ𝕄_Row_ACT3 =
        flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact && :ρe in propertynames(Y.c) ?
        QuaddiagonalRow_ACT3 : BidiagonalRow_ACT3
    ∂Yₜ∂Y = FieldMatrix(
        (ᶜρ_name, ᶠ𝕄_name) => zeros(BidiagonalRow_ACT3, axes(Y.c)),
        (ᶜ𝔼_name, ᶠ𝕄_name) => zeros(∂ᶜ𝔼ₜ∂ᶠ𝕄_Row_ACT3, axes(Y.c)),
        (ᶠ𝕄_name, ᶜρ_name) => zeros(BidiagonalRow_C3, axes(Y.f)),
        (ᶠ𝕄_name, ᶜ𝔼_name) => zeros(BidiagonalRow_C3, axes(Y.f)),
        (ᶠ𝕄_name, ᶠ𝕄_name) => zeros(TridiagonalRow_C3xACT3, axes(Y.f)),
    )

    # When ∂Yₜ∂Y is sparse, one(∂Yₜ∂Y) doesn't contain every diagonal block.
    # To ensure that ∂R∂Y is invertible, we need to call identity_field_matrix.
    I = MatrixFields.identity_field_matrix(Y)

    δtγ = FT(1)
    ∂R∂Y = transform ? I ./ δtγ .- ∂Yₜ∂Y : δtγ .* ∂Yₜ∂Y .- I
    alg = MatrixFields.BlockArrowheadSolve(ᶜρ_name, ᶜ𝔼_name)

    return ImplicitEquationJacobian(
        ∂Yₜ∂Y,
        FieldMatrixWithSolver(∂R∂Y, Y, alg),
        transform,
        flags,
        similar(Y),
        similar(Y),
    )
end

# Required for compatibility with OrdinaryDiffEq.jl
Base.similar(j::ImplicitEquationJacobian) = ImplicitEquationJacobian(
    similar(j.∂Yₜ∂Y),
    similar(j.∂R∂Y),
    j.transform,
    j.flags,
    j.R_field_vector,
    j.δY_field_vector,
)

# Required for compatibility with ClimaTimeSteppers.jl
Base.zero(j::ImplicitEquationJacobian) = ImplicitEquationJacobian(
    zero(j.∂Yₜ∂Y),
    zero(j.∂R∂Y),
    j.transform,
    j.flags,
    j.R_field_vector,
    j.δY_field_vector,
)

# This method for ldiv! is called by Newton's method from ClimaTimeSteppers.jl.
# It solves ∂R∂Y * δY = R for δY, where R is the implicit residual.
ldiv!(
    δY::Fields.FieldVector,
    j::ImplicitEquationJacobian,
    R::Fields.FieldVector,
) = ldiv!(δY, j.∂R∂Y, R)

# This method for ldiv! is called by Krylov.jl from ClimaTimeSteppers.jl.
# See https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/605 for a
# a similar way of handling the AbstractVectors generated by Krylov.jl.
function ldiv!(
    δY::AbstractVector,
    j::ImplicitEquationJacobian,
    R::AbstractVector,
)
    j.R_field_vector .= R
    ldiv!(j.δY_field_vector, j, j.R_field_vector)
    δY .= j.δY_field_vector
end

# This function can be called by Newton's method from OrdinaryDiffEq.jl.
linsolve!(::Type{Val{:init}}, f, u0; kwargs...) = _linsolve!
_linsolve!(δY, j, R, update_matrix = false; kwargs...) = ldiv!(δY, j, R)
