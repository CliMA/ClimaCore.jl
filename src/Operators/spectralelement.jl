import UnrolledUtilities: unrolled_map, unrolled_sum
import ..Utilities.Unrolled: unrolled_map_with_inbounds

"""
    FormType

Supertype of the singleton types [`StrongForm`](@ref) and [`WeakForm`](@ref),
which distinguish the variational form of a spectral element operator.

The strong and weak variants of an operator share the same interior
computation; they differ only in three form-dependent factors:
 - whether the derivative matrix is applied directly or transposed with a sign
   flip (from integration by parts); see `deriv_matrix`,
 - whether the argument is weighted by the quadrature weights `W` or by the
   Jacobian factor; see `materialize_quadrature_weighted` and `materialize_jacobian_weighted`,
 - whether the result is rescaled by `J` or by `WJ`; see
   `lazy_jacobian_unweighted`, and by `W` or not at all; see
   `lazy_quadrature_unweighted`.

Operators with strong/weak variants carry a `FormType` as their type parameter,
e.g. `Divergence{StrongForm}`, with the weak variant available under an alias,
e.g. `WeakDivergence = Divergence{WeakForm}`.
"""
abstract type FormType end

"""
    StrongForm()

The [`FormType`](@ref) of an operator that discretizes a derivative directly at
the quadrature points (e.g. [`Divergence`](@ref), [`Gradient`](@ref),
[`Curl`](@ref)).
"""
struct StrongForm <: FormType end

"""
    WeakForm()

The [`FormType`](@ref) of an operator that discretizes the volume-integral
contribution of the corresponding weak-form expression, obtained after
integration by parts (e.g. [`WeakDivergence`](@ref), [`WeakGradient`](@ref),
[`WeakCurl`](@ref)).
"""
struct WeakForm <: FormType end

"""
    materialize_jacobian_weighted(form, arg)

The Jacobian factor that scales the contravariant components summed by an
operator of the given [`FormType`](@ref): `J` for the strong form, and `WJ`
for the weak form. Used by [`Divergence`](@ref).
"""
@inline materialize_jacobian_weighted(::StrongForm, arg) =
    arg .* Fields.local_geometry_field(arg).J
@inline materialize_jacobian_weighted(::WeakForm, arg) =
    arg .* Fields.local_geometry_field(arg).WJ

"""
    materialize_quadrature_weighted(form, arg)

The argument value `arg`, weighted as required by an operator of the given
[`FormType`](@ref) whose weak variant integrates against test functions: `arg`
itself for the strong form, and `W * arg` (with the quadrature weights
`W = WJ * J⁻¹`) for the weak form. Used by [`Gradient`](@ref) and
[`Curl`](@ref).
"""
@inline materialize_quadrature_weighted(::StrongForm, arg) = Base.materialize(arg)
@inline function materialize_quadrature_weighted(::WeakForm, arg)
    (; WJ, invJ) = Fields.local_geometry_field(arg)
    return Base.broadcasted(*, arg, Base.broadcasted(*, WJ, invJ))
end

"""
    lazy_jacobian_unweighted(form, dest)

The result value `dest`, divided by the `materialize_jacobian_weighted` of the given
[`FormType`](@ref): `dest * J⁻¹` for the strong form (using the precomputed
inverse), and `dest / WJ` for the weak form. Used by [`Divergence`](@ref) and
[`Curl`](@ref).
"""
@inline lazy_jacobian_unweighted(::StrongForm, dest) =
    Base.broadcasted(*, dest, Fields.local_geometry_field(dest).invJ)
@inline lazy_jacobian_unweighted(::WeakForm, dest) =
    Base.broadcasted(/, dest, Fields.local_geometry_field(dest).WJ)

"""
    lazy_quadrature_unweighted(form, dest)

The result value `dest`, divided by the quadrature weights `W = WJ * J⁻¹` if the
given [`FormType`](@ref) requires it: `dest` itself for the strong form, and
`dest / W` for the weak form. Used by [`Gradient`](@ref), whose weak variant
weights its argument by `W` without a Jacobian factor to divide out.

The CPU `apply_operator` methods for [`Gradient`](@ref) inline this rescale
behind an `F === WeakForm` branch instead of calling it, so that the strong form
skips the loop over quadrature points entirely; on GPUs each thread rescales
only its own point, so there is no loop to skip.
"""
@inline lazy_quadrature_unweighted(::StrongForm, dest) = dest
@inline function lazy_quadrature_unweighted(::WeakForm, dest)
    (; WJ, invJ) = Fields.local_geometry_field(dest)
    return Base.broadcasted(/, dest, Base.broadcasted(*, WJ, invJ))
end

"""
    deriv_matrix(form, dest)

Entry of the derivative matrix `D` applied by an operator of the given
[`FormType`](@ref) when accumulating the contribution of quadrature point `i`
to quadrature point `ii`: `D[ii, i]` for the strong form, and `-D[i, ii]` (the
transpose, with the sign flip from integration by parts) for the weak form.
"""
@inline deriv_matrix(::StrongForm, dest) = Quadratures.differentiation_matrix(
    Spaces.undertype(axes(dest)),
    Spaces.quadrature_style(axes(dest)),
)
@inline deriv_matrix(::WeakForm, dest) = -deriv_matrix(StrongForm(), dest)'

"""
    interp_matrix(dest, arg)

Interpolation matrix.
"""
@inline interp_matrix(dest, arg) = Quadratures.interpolation_matrix(
    Spaces.undertype(axes(dest)),
    Spaces.quadrature_style(axes(dest)),
    Spaces.quadrature_style(axes(arg)),
)

"""
    horizontal_dims(arg)

Tuple of the horizontal dimensions covered by a `Field` (either 0, 1, or 2).
"""
function horizontal_dims(arg)
    (; Ni, Nj) = DataLayouts.vijh_params(arg)
    Nq = DataLayouts.ncomponents(arg)
    return Ni == Nj == Nq ? (1, 2) : Ni == Nq ? (1,) : Nj == Nq ? (2,) : ()
end

"""
    SpectralStyle()

Broadcasting requires use of spectral-element operations.
"""
struct SpectralStyle <: Fields.AbstractFieldStyle end

Broadcast.BroadcastStyle(::SpectralStyle, ::Fields.FieldStyle) = SpectralStyle()

"""
    SpectralElementOperator

Operator applied to the quadrature points in each spectral element of a `Field`.
Each subtype must define [`return_eltype`](@ref) and [`apply_operator`](@ref).
"""
abstract type SpectralElementOperator <: AbstractOperator end

const SpectralBroadcasted = Broadcast.Broadcasted{SpectralStyle}
const LazySpectralOperation =
    Broadcast.Broadcasted{SpectralStyle, <:SpectralElementOperator}

Base.eltype(bc::LazySpectralOperation) = return_eltype(bc.f, bc.args...)

function Broadcast.broadcasted(op::SpectralElementOperator, args...)
    args′ = unrolled_map(Broadcast.broadcastable, args)
    style = Broadcast.result_style(SpectralStyle(), Broadcast.combine_styles(args′...))
    return Broadcast.broadcasted(style, op, args′...)
end

function Base.copyto!(dest::Fields.Field, bc::SpectralBroadcasted; kwargs...)
    bc_no_space = strip_space(bc, axes(dest)) # Drop copies of space before sending to GPU.
    DataLayouts.foreach_slab(dest, bc_no_space; kwargs...) do dest_slab, bc_slab_no_space
        bc_slab = unstrip_space(bc_slab_no_space, axes(dest_slab))
        dest_slab .= materialize_spectral_operations(bc_slab)
    end
    call_post_op_callback() && post_op_callback(dest, dest, bc; kwargs...)
    return dest
end

# Use @noinline so allocations by materialize_spectral_operations can be freed
# immediately. If it's inlined, allocations propagate into the calling function.
@inline materialize_spectral_operations(arg) = arg
@inline materialize_spectral_operations(bc::SpectralBroadcasted) =
    Broadcast.broadcasted(bc.f, unrolled_map(materialize_spectral_operations, bc.args)...)
@noinline materialize_spectral_operations(bc::LazySpectralOperation) =
    apply_operator(bc.f, unrolled_map(materialize_spectral_operations, bc.args)...)

# Set dest_slice .+= matrix * arg_slice for each 1D i or j slice of the inputs.
# Threads using the same argument must be synchronized before this is called.
@inline muladd_slab!(dest, matrix, arg, slice_val::Union{Val{:i}, Val{:j}}) =
    DataLayouts.foreach_column(dest; no_index = false) do dest_index, dest_point
        (i, j, _) = Tuple(dest_index)
        (; Ni, Nj) = vijh_params(arg)
        if i <= Ni && j <= Nj
            axis2 = axes(matrix, 2)
            arg_point(i, j) = @inbounds column(arg, i, j, 1)
            @inbounds dest_point[] +=
                slice_val isa Val{:i} ?
                unrolled_sum(i′ -> (@inbounds matrix[i, i′] * arg_point(i′, j)[]), axis2) :
                unrolled_sum(j′ -> (@inbounds matrix[j, j′] * arg_point(i, j′)[]), axis2)
        end
        nothing
    end

# Set dest .= arg' when Ni == Nj, or set dest .= arg[1] if either Ni or Nj is 1.
@inline transpose_slab!(dest, arg) =
    DataLayouts.foreach_column(dest; no_index = false) do dest_index, dest_point
        (i, j, _) = Tuple(dest_index)
        (; Ni, Nj) = vijh_params(arg)
        @inbounds dest_point[] = column(arg, min(j, Ni), min(i, Nj), 1)[]
    end

"""
    div = Divergence()
    div.(u)

Computes the per-element spectral (strong) divergence of a vector field ``u``.

The divergence of a vector field ``u`` is defined as

```math
\\nabla \\cdot u = \\sum_i \\frac{1}{J} \\frac{\\partial (J u^i)}{\\partial \\xi^i}
```

where ``J`` is the Jacobian determinant, ``u^i`` is the ``i``th contravariant
component of ``u``.

This is discretized by

```math
\\sum_i I \\left\\{\\frac{1}{J} \\frac{\\partial (I\\{J u^i\\})}{\\partial \\xi^i} \\right\\}
```

where ``I\\{x\\}`` is the interpolation operator that projects to the
unique polynomial interpolating ``x`` at the quadrature points. In matrix
form, this can be written as

```math
J^{-1} \\sum_i D_i J u^i
```

where ``D_i`` is the derivative matrix along the ``i``th dimension

## References

  - [Taylor2010](@cite), equation 15
"""
struct Divergence{F <: FormType} <: SpectralElementOperator end
Divergence() = Divergence{StrongForm}()

return_space(::Divergence, space) = space
return_eltype(::Divergence, arg) = Geometry.divergence_result_type(eltype(arg))

# Strong form is J⁻¹ ∑ₕ Dₕ J argʰ, weak form is -(WJ)⁻¹ ∑ₕ Dₕᵀ WJ argʰ.
@inline function apply_operator(op::Divergence{F}, arg) where {F}
    dims = horizontal_dims(arg)
    arg′ = materialize_jacobian_weighted(F(), arg)
    lg = Fields.local_geometry_field(arg′)
    arg′¹ = 1 in dims ? Geometry.contravariant1.(arg′, lg) : nothing
    arg′² = 2 in dims ? Geometry.contravariant2.(arg′, lg) : nothing
    DataLayouts.synchronize(DataLayouts.DataScope(arg′))
    dest = similar(arg′, return_eltype(op, arg))
    dest .= zero(eltype(dest))
    matrix = deriv_matrix(F(), dest)
    1 in dims && muladd_slab!(dest, matrix, arg′¹, Val(:i))
    2 in dims && muladd_slab!(dest, matrix, arg′², Val(:j))
    return lazy_jacobian_unweighted(F(), dest)
end

"""
    split_div = SplitDivergence()
    split_div.(ρu, ψ)

Computes the divergence of the product `ρu * ψ` using a **split-form (entropy-stable)** discretization.

This operator is designed for the advection of scalar quantities in conservation laws (e.g.,
thermodynamic variables or tracers). By evaluating the divergence using a specific averaging of the
conservative and advective forms, this formulation cancels aliasing errors that arise from the product
of two spectrally variable fields, thereby inhibiting the growth of quadratic instabilities (such as
cold temperature spikes) without requiring hyperviscosity.

# Arguments

  - `ρu`: The transport vector field, typically the **mass flux**. It must be a vector quantity (e.g., `Geometry.Contravariant12Vector`).
  - `ψ`: The **specific** scalar quantity to be advected (e.g., specific total energy ``e_{tot}`` or specific humidity ``q_{tot}``).

# Mathematical Formulation

## Continuous

The split form of the divergence operator is defined as the arithmetic mean of
the conservative and advective forms:

```math
\\nabla \\cdot (\\rho \\mathbf{u} \\psi)|_\\textrm{split} =
    \\frac{1}{2} \\nabla \\cdot (\\rho \\mathbf{u} \\psi) +
    \\frac{1}{2} \\left(
        \\psi \\nabla \\cdot (\\rho \\mathbf{u}) +
        \\rho \\mathbf{u} \\cdot \\nabla \\psi
    \\right)
```

## Discrete

The discretized split operator is equivalent to using the strong formulation of
the gradient operator and the weak formulation of the divergence operator:

```math
\\textrm{split_div}(\\rho \\mathbf{u}, \\psi) =
    \\frac{1}{2} \\textrm{wdiv}(\\rho \\mathbf{u} \\psi) +
    \\frac{1}{2} \\left(
        \\psi \\textrm{wdiv}(\\rho \\mathbf{u}) +
        \\rho \\mathbf{u} \\cdot \\textrm{grad}(\\psi)
    \\right)
```

Swapping the weak and strong formulations in the last two terms also results in
the same operator. The discrete form of the divergence theorem, which stems from
the generalized summation-by-parts (SBP) property, guarantees that the integral
of the first term vanishes,

```math
\\int_\\Omega \\textrm{wdiv}(\\rho \\mathbf{u} \\psi) dV = 0
```

while the integrals of the other two terms cancel out,

```math
\\int_\\Omega \\psi \\textrm{wdiv}(\\rho \\mathbf{u}) dV =
    -\\int_\\Omega \\rho \\mathbf{u} \\cdot \\textrm{grad}(\\psi) dV
```

So, this discretization ensures that the split operator conserves the integral
of ``\\rho \\mathbf{u} \\psi``.

## Two-Point

A more compact representation of the discretized operator can be obtained with
the symmetric two-point flux, whose values in one dimension are

```math
(F^1)_{ij} =
    \\frac{1}{2} (\\rho_i J_i (u^1)_i + \\rho_j J_j (u^1)_j) (\\psi_i + \\psi_j)
```

With ``D`` denoting the spectral derivative matrix, the split operator in one
dimension can be expressed as

```math
\\textrm{split_div}(\\rho \\mathbf{u}, \\psi)_i =
    \\frac{1}{J_i} \\sum_{j \\neq i} D_{ij} (F^1)_{ij}
```

In two dimensions, ``F^1`` and the analogous quantity ``F^2`` provide a similar
expression for the split divergence, with the one-dimensional operator applied
sequentially along each dimension.

# Properties

 1. **Conservation:** The split operator conserves ``\\rho \\mathbf{u} \\psi``
 2. **Consistency:** If ``\\psi = 1``, the split operator degenerates to the
    weak formulation of ``\\nabla \\cdot \\rho \\mathbf{u}`` (mass continuity)
 3. **Complexity:** The split operator has the same ``O(N^2)`` complexity per
    element as the strong and weak operators, but needs twice as many operations

# References

  - Fisher, T. C., & Carpenter, M. H. (2013). High-order entropy stable finite difference schemes for nonlinear conservation laws: Finite domains. Journal of Computational Physics, 252, 518-557. [https://doi.org/10.1016/j.jcp.2013.06.014](https://doi.org/10.1016/j.jcp.2013.06.014)
  - Gassner, G. J. (2013). A skew-symmetric discontinuous Galerkin spectral element discretization and its relation to SBP-SAT finite difference methods. SIAM Journal on Scientific Computing, 35, A1233-A1253. [https://doi.org/10.1137/120890144](https://doi.org/10.1137/120890144)
"""
struct SplitDivergence <: SpectralElementOperator end

return_space(::SplitDivergence, space, _) = space
return_eltype(::SplitDivergence, arg1, arg2) =
    Geometry.mul_return_type(Geometry.divergence_result_type(eltype(arg1)), eltype(arg2))

# Split form is J⁻¹ ∑ₕ Dₕ Fʰ, where F is @. (arg1 + arg1') * (arg2 + arg2') / 2.
@noinline function apply_operator(op::SplitDivergence, arg1, arg2)
    dims = horizontal_dims(arg1)
    Ju = materialize_jacobian_weighted(StrongForm(), arg1)
    ψ = Base.materialize(arg2)
    DataLayouts.synchronize(DataLayouts.DataScope(Ju, ψ))
    transposed_Ju = similar(Ju)
    transposed_ψ = similar(ψ)
    transpose_slab!(transposed_Ju, Ju)
    transpose_slab!(transposed_ψ, ψ)
    arg′ = (Ju .+ transposed_Ju) .* (ψ .+ transposed_ψ) ./ 2
    lg = Fields.local_geometry_field(arg′)
    arg′¹ = 1 in dims ? Geometry.contravariant1.(arg′, lg) : nothing
    arg′² = 2 in dims ? Geometry.contravariant2.(arg′, lg) : nothing
    DataLayouts.synchronize(DataLayouts.DataScope(arg′))
    dest = similar(arg′, return_eltype(op, arg1, arg2))
    dest .= zero(eltype(dest))
    matrix = deriv_matrix(StrongForm(), dest)
    matrix -= LinearAlgebra.Diagonal(matrix)
    1 in dims && muladd_slab!(dest, matrix, arg′¹, Val(:i))
    2 in dims && muladd_slab!(dest, matrix, arg′², Val(:j))
    return lazy_jacobian_unweighted(StrongForm(), dest)
end

"""
    grad = Gradient()
    grad.(f)

Compute the (strong) gradient of `f` on each element, returning a
`CovariantVector`-field.

The ``i``th covariant component of the gradient is the partial derivative with
respect to the reference element:

```math
(\\nabla f)_i = \\frac{\\partial f}{\\partial \\xi^i}
```

Discretely, this can be written in matrix form as

```math
D_i f
```

where ``D_i`` is the derivative matrix along the ``i``th dimension.

## References

  - [Taylor2010](@cite), equation 16
"""
struct Gradient{F <: FormType} <: SpectralElementOperator end
Gradient() = Gradient{StrongForm}()

return_space(::Gradient, space) = space
return_eltype(::Gradient, arg) =
    Geometry.gradient_result_type(Val(horizontal_dims(arg)), eltype(arg))

# Strong form is ∑ₕ Dₕ eʰ ⊗ f, weak form is -W⁻¹ ∑ₕ Dₕᵀ eʰ ⊗ W f.
@inline function apply_operator(op::Gradient{F}, arg) where {F}
    dims = horizontal_dims(arg)
    arg′ = materialize_quadrature_weighted(F(), arg)
    e¹_arg′ = 1 in dims ? (Geometry.Covariant1Vector(true),) .⊗ arg′ : nothing
    e²_arg′ = 2 in dims ? (Geometry.Covariant2Vector(true),) .⊗ arg′ : nothing
    DataLayouts.synchronize(DataLayouts.DataScope(arg′))
    dest = similar(arg′, return_eltype(op, arg))
    dest .= zero(eltype(dest))
    matrix = deriv_matrix(F(), dest)
    1 in dims && muladd_slab!(dest, matrix, e¹_arg′, Val(:i))
    2 in dims && muladd_slab!(dest, matrix, e²_arg′, Val(:j))
    return lazy_quadrature_unweighted(F(), dest)
end

"""
    curl = Curl()
    curl.(u)

Computes the per-element spectral (strong) curl of a covariant vector field ``u``.

Note: The vector field ``u`` needs to be excliclty converted to a `CovaraintVector`,
as then the `Curl` is independent of the local metric tensor.

The curl of a vector field ``u`` is a vector field with contravariant components

```math
(\\nabla \\times u)^i = \\frac{1}{J} \\sum_{jk} \\epsilon^{ijk} \\frac{\\partial u_k}{\\partial \\xi^j}
```

where ``J`` is the Jacobian determinant, ``u_k`` is the ``k``th covariant
component of ``u``, and ``\\epsilon^{ijk}`` are the [Levi-Civita
symbols](https://en.wikipedia.org/wiki/Levi-Civita_symbol#Three_dimensions_2).
In other words

```math
\\begin{bmatrix}
  (\\nabla \\times u)^1 \\\\
  (\\nabla \\times u)^2 \\\\
  (\\nabla \\times u)^3
\\end{bmatrix}
=
\\frac{1}{J} \\begin{bmatrix}
  \\frac{\\partial u_3}{\\partial \\xi^2} - \\frac{\\partial u_2}{\\partial \\xi^3} \\\\
  \\frac{\\partial u_1}{\\partial \\xi^3} - \\frac{\\partial u_3}{\\partial \\xi^1} \\\\
  \\frac{\\partial u_2}{\\partial \\xi^1} - \\frac{\\partial u_1}{\\partial \\xi^2}
\\end{bmatrix}
```

In matrix form, this becomes

```math
\\epsilon^{ijk} J^{-1} D_j u_k
```

Note that unused dimensions will be dropped: e.g. the 2D curl of a
`Covariant12Vector`-field will return a `Contravariant3Vector`.

## References

  - [Taylor2010](@cite), equation 17
"""
struct Curl{F <: FormType} <: SpectralElementOperator end
Curl() = Curl{StrongForm}()

return_space(::Curl, space) = space
return_eltype(::Curl, arg) =
    Geometry.curl_result_type(Val(horizontal_dims(arg)), eltype(arg))

# Strong form is J⁻¹ ∑ₕₙₘ εʰⁿᵐ Dₕ argₙ eₘ, weak form is -(WJ)⁻¹ ∑ₕₙₘ εʰⁿᵐ Dₕᵀ W argₙ eₘ.
@inline function apply_operator(op::Curl{F}, arg) where {F}
    dims = horizontal_dims(arg)
    arg′ = materialize_quadrature_weighted(F(), arg)
    lg = Fields.local_geometry_field(arg′)
    ε¹ⁿᵐ_arg′ₙ_eₘ =
        1 in dims ?
        Geometry.Contravariant3Vector.(Geometry.covariant2.(arg′, lg)) .-
        Geometry.Contravariant2Vector.(Geometry.covariant3.(arg′, lg)) : nothing
    ε²ⁿᵐ_arg′ₙ_eₘ =
        2 in dims ?
        Geometry.Contravariant1Vector.(Geometry.covariant3.(arg′, lg)) .-
        Geometry.Contravariant3Vector.(Geometry.covariant1.(arg′, lg)) : nothing
    DataLayouts.synchronize(DataLayouts.DataScope(arg′))
    dest = similar(arg′, return_eltype(op, arg))
    dest .= zero(eltype(dest))
    matrix = deriv_matrix(F(), dest)
    1 in dims && muladd_slab!(dest, matrix, ε¹ⁿᵐ_arg′ₙ_eₘ, Val(:i))
    2 in dims && muladd_slab!(dest, matrix, ε²ⁿᵐ_arg′ₙ_eₘ, Val(:j))
    return lazy_jacobian_unweighted(F(), dest)
end

"""
    i = Interpolate(space)
    i.(f)

Interpolates `f` to the `space`. If `space` has equal or higher polynomial
degree as the space of `f`, this is exact, otherwise it will be lossy.

In matrix form, it is the linear operator

```math
I = \\bigotimes_i I_i
```

where ``I_i`` is the barycentric interpolation matrix in the ``i``th dimension.

See also [`Restrict`](@ref).
"""
struct Interpolate{S} <: SpectralElementOperator
    space::S
end

return_space(op::Interpolate, _) = op.space
return_eltype(::Interpolate, arg) = eltype(arg)

@inline function apply_operator(op::Interpolate, arg)
    arg′ = Base.materialize(arg)
    DataLayouts.synchronize(DataLayouts.DataScope(arg′))
    dest = Fields.Field(eltype(arg), slab(op.space, 1, 1))
    dest .= zero(eltype(dest))
    matrix = interp_matrix(dest, arg)
    if horizontal_dims(arg) == (1,)
        muladd_slab!(dest, matrix, arg′, Val(:i))
    elseif horizontal_dims(arg) == (2,)
        muladd_slab!(dest, matrix, arg′, Val(:j))
    elseif horizontal_dims(arg) == (1, 2)
        partial_result =
            DataLayouts.nquadpoints(dest) > DataLayouts.nquadpoints(arg) ?
            similar(dest) : similar(arg)
        muladd_slab!(partial_result, matrix, arg′, Val(:i))
        DataLayouts.synchronize(DataLayouts.DataScope(partial_result))
        muladd_slab!(dest, matrix, partial_result, Val(:j))
    end
    return dest
end

"""
    r = Restrict(space)
    r.(f)

Computes the projection of a field `f` on ``\\mathcal{V}_0`` to a lower degree
polynomial space `space` (``\\mathcal{V}_0^*``). `space` must be on the same
topology as the space of `f`, but have a lower polynomial degree.

It is defined as the field ``\\theta \\in \\mathcal{V}_0^*`` such that for all ``\\phi \\in \\mathcal{V}_0^*``

```math
\\int_\\Omega \\phi \\theta \\,d\\Omega = \\int_\\Omega \\phi f \\,d\\Omega
```

In matrix form, this is

```math
\\phi^\\top W^* J^* \\theta = (I \\phi)^\\top WJ f
```

where ``W^*`` and ``J^*`` are the quadrature weights and Jacobian determinant of
``\\mathcal{V}_0^*``, and ``I`` is the interpolation operator (see [`Interpolate`](@ref))
from ``\\mathcal{V}_0^*`` to ``\\mathcal{V}_0``. This reduces to

```math
\\theta = (W^* J^*)^{-1} I^\\top WJ f
```
"""
struct Restrict{S} <: SpectralElementOperator
    space::S
end

return_space(op::Restrict, _) = op.space
return_eltype(::Restrict, arg) = eltype(arg)

@inline function apply_operator(op::Restrict, arg)
    arg′ = materialize_jacobian_weighted(WeakForm(), arg)
    DataLayouts.synchronize(DataLayouts.DataScope(arg′))
    dest = Fields.Field(eltype(arg), slab(op.space, 1, 1))
    dest .= zero(eltype(dest))
    matrix = interp_matrix(arg, dest)'
    if horizontal_dims(arg) == (1,)
        muladd_slab!(dest, matrix, arg′, Val(:i))
    elseif horizontal_dims(arg) == (2,)
        muladd_slab!(dest, matrix, arg′, Val(:j))
    elseif horizontal_dims(arg) == (1, 2)
        partial_result =
            DataLayouts.nquadpoints(dest) > DataLayouts.nquadpoints(arg) ?
            similar(dest) : similar(arg)
        muladd_slab!(partial_result, matrix, arg′, Val(:i))
        DataLayouts.synchronize(DataLayouts.DataScope(partial_result))
        muladd_slab!(dest, matrix, partial_result, Val(:j))
    end
    return lazy_jacobian_unweighted(WeakForm(), dest)
end

"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element.
"""
function tensor_product! end

function tensor_product!(
    out::Union{
        DataLayouts.VIJHWithF{S, Nv, Ni_out, 1},
        DataLayouts.VIH1{S, Nv, Ni_out},
    },
    indata::DataLayouts.VIJHWithF{S, Nv, Ni_in, 1},
    M::SMatrix{Ni_out, Ni_in},
) where {S, Nv, Ni_out, Ni_in}
    Nh_in = DataLayouts.nelems(indata)
    Nh_out = DataLayouts.nelems(out)
    # TODO: assumes the same number of levels (horizontal only)
    @assert Nh_in == Nh_out
    @inbounds for h in 1:Nh_out, v in 1:Nv
        in_slab = slab(indata, v, h)
        out_slab = slab(out, v, h)
        for i in 1:Ni_out
            r = M[i, 1] * in_slab[1]
            for ii in 2:Ni_in
                r = muladd(M[i, ii], in_slab[ii], r)
            end
            out_slab[i] = r
        end
    end
    return out
end

function tensor_product!(
    out::DataLayouts.VIJHWithF{S, 1, Nij_out, Nij_out},
    indata::DataLayouts.VIJHWithF{S, 1, Nij_in, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}
    Nh = size(indata, 4)
    @assert Nh == size(out, 4)

    # temporary storage
    temp = MArray{Tuple{1, Nij_out, Nij_in, 1}, S, 4, Nij_out * Nij_in}(undef)

    @inbounds for h in 1:Nh
        in_slab = slab(indata, 1, h)
        out_slab = slab(out, 1, h)
        for j in 1:Nij_in, i in 1:Nij_out
            temp[1, i, j, 1] = rmatmul1(M, in_slab, i, j)
        end
        for j in 1:Nij_out, i in 1:Nij_out
            out_slab[1, i, j, 1] = rmatmul2(M, temp, i, j)
        end
    end
    return out
end

function tensor_product!(
    out::DataLayouts.IH1JH2{S, Nij_out, Nij_out},
    indata::DataLayouts.VIJHWithF{S, 1, Nij_in, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}
    Nh = DataLayouts.nelems(indata)
    @assert Nh == DataLayouts.nelems(out)

    # temporary storage
    temp = MArray{Tuple{1, Nij_out, Nij_in, 1}, S, 4, Nij_out * Nij_in}(undef)

    @inbounds for h in 1:Nh
        in_slab = slab(indata, 1, h)
        out_slab = slab(out, 1, h)
        for j in 1:Nij_in, i in 1:Nij_out
            temp[1, i, j, 1] = rmatmul1(M, in_slab, i, j)
        end
        for j in 1:Nij_out, i in 1:Nij_out
            out_slab[i, j] = rmatmul2(M, temp, i, j)
        end
    end
    return out
end

function tensor_product!(
    out_slab::DataLayouts.VIJHWithF{S, 1, Nij_out, Nij_out, 1},
    in_slab::DataLayouts.VIJHWithF{S, 1, Nij_in, Nij_in, 1},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}
    # temporary storage
    temp = MArray{Tuple{1, Nij_out, Nij_in, 1}, S, 4, Nij_out * Nij_in}(undef)
    @inbounds for j in 1:Nij_in, i in 1:Nij_out
        temp[1, i, j, 1] = rmatmul1(M, in_slab, i, j)
    end
    @inbounds for j in 1:Nij_out, i in 1:Nij_out
        out_slab[1, i, j, 1] = rmatmul2(M, temp, i, j)
    end
    return out_slab
end

function tensor_product!(
    inout::DataLayouts.VIJHWithF{S, 1, Nij, Nij},
    M::SMatrix{Nij, Nij},
) where {S, Nij}
    inout_bc = Base.broadcastable(inout)
    tensor_product!(inout_bc, inout_bc, M)
end

"""
    matrix_interpolate(field, quadrature)

Computes the tensor product given a uniform quadrature `out = (M ⊗ M) * in` on each element.
Returns a 2D Matrix for plotting / visualizing 2D Fields.
"""
function matrix_interpolate end

function matrix_interpolate(
    field::Fields.SpectralElementField2D,
    Q_interp::Quadratures.Uniform{Nu},
) where {Nu}
    S = eltype(field)
    space = axes(field)
    topology = Spaces.topology(space)
    quadrature_style = Spaces.quadrature_style(space)
    mesh = topology.mesh
    n1, n2 = size(Meshes.elements(mesh))
    interp_data =
        DataLayouts.IH1JH2{S, Nu, Nu, nothing}(Matrix{S}(undef, (Nu * n1, Nu * n2)))
    M = Quadratures.interpolation_matrix(Float64, Q_interp, quadrature_style)
    tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end

function matrix_interpolate(
    field::Fields.ExtrudedFiniteDifferenceField,
    Q_interp::Union{Quadratures.Uniform{Nu}, Quadratures.ClosedUniform{Nu}},
) where {Nu}
    S = eltype(field)
    space = axes(field)
    quadrature_style = Spaces.quadrature_style(space)
    nl = Spaces.nlevels(space)
    n1 = Topologies.nlocalelems(Spaces.topology(space))
    interp_data =
        DataLayouts.VIH1{S, nl, Nu, nothing}(Matrix{S}(undef, (nl, Nu * n1)))
    M = Quadratures.interpolation_matrix(Float64, Q_interp, quadrature_style)
    tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end

"""
    matrix_interpolate(field, Nu::Integer)

Computes the tensor product given a uniform quadrature degree of Nu on each element.
Returns a 2D Matrix for plotting / visualizing 2D Fields.
"""
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())

"""
    rmatmul1(W, S, i, j)

Recursive matrix product along the 1st dimension of `S`. Equivalent to:

    mapreduce(*, +, W[i,:], S[:,j])
"""
function rmatmul1(W, S, i, j)
    Nq = size(W, 2)
    @inbounds r = W[i, 1] * S[1, 1, j, 1]
    @inbounds for ii in 2:Nq
        r = muladd(W[i, ii], S[1, ii, j, 1], r)
    end
    return r
end

"""
    rmatmul2(W, S, i, j)

Recursive matrix product along the 2nd dimension `S`. Equivalent to:

    mapreduce(*, +, W[j,:], S[i, :])
"""
function rmatmul2(W, S, i, j)
    Nq = size(W, 2)
    @inbounds r = W[j, 1] * S[1, i, 1, 1]
    @inbounds for jj in 2:Nq
        r = muladd(W[j, jj], S[1, i, jj, 1], r)
    end
    return r
end
