abstract type AbstractSpectralStyle <: Fields.AbstractFieldStyle end

struct SpectralStyle <: AbstractSpectralStyle end

struct CompositeSpectralStyle <: AbstractSpectralStyle end

"""
    SpectralElementOperator

Represents an operation that is applied to each element.

Subtypes `Op` of this should define the following:
- [`operator_return_eltype(::Op, ElTypes...)`](@ref)
- [`allocate_work(::Op, args...)`](@ref)
- [`apply_slab(::Op, work, args...)`](@ref)

Additionally, the result type `OpResult <: OperatorSlabResult` of `apply_slab` should define `get_node(::OpResult, i, j)`.
"""
abstract type SpectralElementOperator end

"""
    operator_axes(space)

Return a tuple of the axis indicies a given field operator works over.
"""
function operator_axes end

operator_axes(space::Spaces.AbstractSpace) = ()
operator_axes(space::Spaces.SpectralElementSpace1D) = (1,)
operator_axes(space::Spaces.SpectralElementSpace2D) = (1, 2)
operator_axes(space::Spaces.SpectralElementSpaceSlab1D) = (1,)
operator_axes(space::Spaces.SpectralElementSpaceSlab2D) = (1, 2)
operator_axes(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    operator_axes(space.horizontal_space)

"""
    SpectralBroadcasted{Style}(op, args[,axes[, work]])

This is similar to a `Base.Broadcast.Broadcasted` object, except it contains space for an intermediate `work` storage.

This is returned by `Base.Broadcast.broadcasted(op::SpectralElementOperator)`.
"""
struct SpectralBroadcasted{Style, Op, Args, Axes, InputSpace} <:
       Base.AbstractBroadcasted
    op::Op
    args::Args
    axes::Axes
    input_space::InputSpace
end
SpectralBroadcasted{Style}(
    op::Op,
    args::Args,
    axes::Axes = nothing,
    input_space::InputSpace = nothing,
) where {Style, Op, Args, Axes, InputSpace} =
    SpectralBroadcasted{Style, Op, Args, Axes, InputSpace}(
        op,
        args,
        axes,
        input_space,
    )

input_space(arg) = axes(arg)
input_space(::SpectralElementOperator, space) = space

input_space(sbc::SpectralBroadcasted) =
    isnothing(sbc.input_space) ?
    input_space(sbc.op, tuplemap(axes, sbc.args)...) : sbc.input_space

return_space(::SpectralElementOperator, space) = space

Base.axes(sbc::SpectralBroadcasted) =
    isnothing(sbc.axes) ? return_space(sbc.op, tuplemap(axes, sbc.args)...) :
    sbc.axes

Base.Broadcast.broadcasted(op::SpectralElementOperator, args...) =
    Base.Broadcast.broadcasted(SpectralStyle(), op, args...)

Base.Broadcast.broadcasted(
    ::SpectralStyle,
    op::SpectralElementOperator,
    args...,
) = SpectralBroadcasted{SpectralStyle}(op, args)

Base.eltype(sbc::SpectralBroadcasted) =
    operator_return_eltype(sbc.op, tuplemap(eltype, sbc.args)...)

@inline instantiate_args(args::Tuple) =
    (Base.Broadcast.instantiate(args[1]), instantiate_args(Base.tail(args))...)
@inline instantiate_args(args::Tuple{Any}) =
    (Base.Broadcast.instantiate(args[1]),)
@inline instantiate_args(::Tuple{}) = ()

function Base.Broadcast.instantiate(
    sbc::SpectralBroadcasted{Style},
) where {Style}
    op = sbc.op
    # recursively instantiate the arguments to allocate intermediate work arrays
    args = instantiate_args(sbc.args)
    # axes: same logic as Broadcasted
    if sbc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Base.axes(sbc)
    else
        axes = sbc.axes
        Base.Broadcast.check_broadcast_axes(axes, args...)
    end
    if sbc.input_space isa Nothing
        inspace = input_space(sbc)
    else
        inspace = sbc.input_space
    end
    op = typeof(op)(axes)
    return SpectralBroadcasted{Style}(op, args, axes, inspace)
end

function Base.Broadcast.instantiate(
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractSpectralStyle}
    # recursively instantiate the arguments to allocate intermediate work arrays
    args = instantiate_args(bc.args)
    # axes: same logic as Broadcasted
    if bc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Base.Broadcast.combine_axes(args...)
    else
        axes = bc.axes
        Base.Broadcast.check_broadcast_axes(axes, args...)
    end
    return Base.Broadcast.Broadcasted{Style}(bc.f, args, axes)
end

function Base.similar(sbc::SpectralBroadcasted, ::Type{Eltype}) where {Eltype}
    space = axes(sbc)
    return Field(Eltype, space)
end

function Base.similar(
    bc::Base.Broadcast.Broadcasted{<:AbstractSpectralStyle},
    ::Type{Eltype},
) where {Eltype}
    space = axes(bc)
    return Field(Eltype, space)
end

function Base.copy(sbc::SpectralBroadcasted)
    # figure out return type
    dest = similar(sbc, eltype(sbc))
    # allocate dest
    copyto!(dest, sbc)
end
Base.Broadcast.broadcastable(sbc::SpectralBroadcasted) = sbc

function Base.Broadcast.materialize(sbc::SpectralBroadcasted)
    copy(Base.Broadcast.instantiate(sbc))
end

function Base.copyto!(out::Field, sbc::SpectralBroadcasted)
    Fields.byslab(axes(out)) do slabidx
        copyto_slab!(out, sbc, slabidx)
    end
    return out
end

function Base.Broadcast.materialize!(dest, sbc::SpectralBroadcasted)
    copyto!(dest, Base.Broadcast.instantiate(sbc))
end

function Base.copyto!(
    out::Field,
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractSpectralStyle}
    Fields.byslab(axes(out)) do slabidx
        copyto_slab!(out, bc, slabidx)
    end
    return out
end

"""
    copyto_slab!(out, bc, slabidx)

Copy the slab indexed by `slabidx` from `bc` to `out`.
"""
function copyto_slab!(out, bc, slabidx)
    space = axes(out)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    rbc = resolve_operator(bc, slabidx)
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        set_node!(out, ij, slabidx, get_node(rbc, ij, slabidx))
    end
    return nothing
end


"""
    resolve_operator(bc, slabidx)

Recursively evaluate any operators in `bc` at `slabidx`, replacing any
`SpectralBroadcasted` objects with an `IF`/`IJF` object.
"""
function resolve_operator(bc::SpectralBroadcasted, slabidx)
    args = _resolve_operator_args(slabidx, bc.args...)
    apply_operator(bc.op, bc.axes, slabidx, args...)
end
function resolve_operator(bc::Base.Broadcast.Broadcasted{CompositeSpectralStyle}, slabidx)
    args = _resolve_operator_args(slabidx, bc.args...)
    Base.Broadcast.Broadcasted{CompositeSpectralStyle}(bc.f, args, bc.axes)
end
resolve_operator(x, slabidx) = x

"""
    _resolve_operator(slabidx, args...)

Calls `resolve_operator(arg, slabidx)` for each `arg` in `args`
"""
_resolve_operator_args(slabidx) = ()
@inline _resolve_operator_args(slabidx, arg, xargs...) =
    (resolve_operator(arg, slabidx), _resolve_operator_args(slabidx, xargs...)...)



_get_node(ij, slabidx) = ()
_get_node(ij, slabidx, arg, xargs...) = (get_node(arg, ij, slabidx), _get_node(ij, slabidx, xargs...)...)

Base.@propagate_inbounds function get_node(scalar, ij, slabidx)
    scalar[]
end
Base.@propagate_inbounds function get_node(field::Fields.Field, ij, slabidx)
    #F = eltype(field)
    i,j = Tuple(ij)
    h = slabidx.h
    Fields.field_values(field)[i, j, h]
    #Nq = 5
    #DataLayouts.get_struct(parent(field), F, (i-1) + Nq * (j-1) + Nq*Nq*h, Nq*Nq)
    # column(Fields.field_values(field), i,j,slabidx.h)[]
    # TODO: get rid of slab call
    # getindex(Fields.field_values(slab(field, slabidx)), ij)
end
Base.@propagate_inbounds function get_node(bc::Base.Broadcast.Broadcasted, ij, slabidx)
    bc.f(_get_node(ij, slabidx, bc.args...)...)
end
Base.@propagate_inbounds function get_node(data::Union{DataLayouts.IJF, DataLayouts.IF}, ij, slabidx)
    data[ij]
end
Base.@propagate_inbounds function get_node(data::StaticArrays.SArray, ij, slabidx)
    data[ij]
end

Base.@propagate_inbounds function get_local_geometry(space::Spaces.AbstractSpectralElementSpace, ij, slabidx)
    i,j = Tuple(ij)
    h = slabidx.h
    space.local_geometry[i, j, h]
end


Base.@propagate_inbounds function set_node!(
    field::Fields.Field,
    ij, slabidx, val
)
    i,j = Tuple(ij)
    h = slabidx.h
    Fields.field_values(field)[i, j, h] = val
    # setindex!(Fields.field_values(slab(field, slabidx)), val, ij)
end


function Base.Broadcast.BroadcastStyle(
    ::Type{SB},
) where {SB <: SpectralBroadcasted}
    CompositeSpectralStyle()
end

function Base.Broadcast.BroadcastStyle(
    ::CompositeSpectralStyle,
    ::Fields.AbstractFieldStyle,
)
    CompositeSpectralStyle()
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
struct Divergence{I} <: SpectralElementOperator end
Divergence() = Divergence{()}()
Divergence{()}(space) = Divergence{operator_axes(space)}()

operator_return_eltype(op::Divergence{I}, ::Type{S}) where {I, S} =
    RecursiveApply.rmaptype(Geometry.divergence_result_type, S)

function apply_operator(op::Divergence{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MVector{Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(arg, ij, slabidx)
        Jv¹ =
            local_geometry.J ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant1(v, local_geometry),
                v,
            )
        for ii in 1:Nq
            out[ii] = out[ii] ⊞ (D[ii, i] ⊠ Jv¹)
        end
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[i] = RecursiveApply.rdiv(out[i], local_geometry.J)
    end
    return SArray(out)
end

function apply_operator(op::Divergence{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(arg, ij, slabidx)
        Jv¹ =
            local_geometry.J ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant1(v, local_geometry),
                v,
            )
        for ii in 1:Nq
            out[ii, j] = out[ii, j] ⊞ (D[ii, i] ⊠ Jv¹)
        end
        Jv² =
            local_geometry.J ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant2(v, local_geometry),
                v,
            )
        for jj in 1:Nq
            out[i, jj] = out[i, jj] ⊞ (D[jj, j] ⊠ Jv²)
        end
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[i, j] = RecursiveApply.rdiv(out[i, j], local_geometry.J)
    end
    return SArray(out)
end


"""
    wdiv = WeakDivergence()
    wdiv.(u)

Computes the "weak divergence" of a vector field `u`.

This is defined as the scalar field ``\\theta \\in \\mathcal{V}_0`` such that
for all ``\\phi\\in \\mathcal{V}_0``
```math
\\int_\\Omega \\phi \\theta \\, d \\Omega
=
- \\int_\\Omega (\\nabla \\phi) \\cdot u \\,d \\Omega
```
where ``\\mathcal{V}_0`` is the space of ``u``.

This arises as the contribution of the volume integral after by applying
integration by parts to the weak form expression of the divergence
```math
\\int_\\Omega \\phi (\\nabla \\cdot u) \\, d \\Omega
=
- \\int_\\Omega (\\nabla \\phi) \\cdot u \\,d \\Omega
+ \\oint_{\\partial \\Omega} \\phi (u \\cdot n) \\,d \\sigma
```

It can be written in matrix form as
```math
ϕ^\\top WJ θ = - \\sum_i (D_i ϕ)^\\top WJ u^i
```
which reduces to
```math
θ = -(WJ)^{-1} \\sum_i D_i^\\top WJ u^i
```
where
 - ``J`` is the diagonal Jacobian matrix
 - ``W`` is the diagonal matrix of quadrature weights
 - ``D_i`` is the derivative matrix along the ``i``th dimension
"""
struct WeakDivergence{I} <: SpectralElementOperator end
WeakDivergence() = WeakDivergence{()}()
WeakDivergence{()}(space) = WeakDivergence{operator_axes(space)}()

operator_return_eltype(::WeakDivergence{I}, ::Type{S}) where {I, S} =
    RecursiveApply.rmaptype(Geometry.divergence_result_type, S)

function apply_operator(op::WeakDivergence{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MVector{Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(arg, ij, slabidx)
        WJv¹ =
            local_geometry.WJ ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant1(v, local_geometry),
                v,
            )
        for ii in 1:Nq
            out[ii] = out[ii] ⊞ (D[i, ii] ⊠ WJv¹)
        end
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[i] = RecursiveApply.rdiv(out[i], ⊟(local_geometry.WJ))
    end
    return SArray(out)
end

function apply_operator(op::WeakDivergence{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(arg, ij, slabidx)
        WJv¹ =
            local_geometry.WJ ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant1(v, local_geometry),
                v,
            )
        for ii in 1:Nq
            out[ii, j] = out[ii, j] ⊞ (D[i, ii] ⊠ WJv¹)
        end
        WJv² =
            local_geometry.WJ ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant2(v, local_geometry),
                v,
            )
        for jj in 1:Nq
            out[i, jj] = out[i, jj] ⊞ (D[j, jj] ⊠ WJv²)
        end
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[i, j] = RecursiveApply.rdiv(out[i, j], ⊟(local_geometry.WJ))
    end
    return SArray(out)
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
struct Gradient{I} <: SpectralElementOperator end
Gradient() = Gradient{()}()
Gradient{()}(space) = Gradient{operator_axes(space)}()

operator_return_eltype(::Gradient{I}, ::Type{S}) where {I, S} =
    RecursiveApply.rmaptype(T -> Geometry.gradient_result_type(Val(I), T), S)

function apply_operator(op::Gradient{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MVector{Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        x = get_node(arg, ij, slabidx)
        for ii in 1:Nq
            ∂f∂ξ = Geometry.Covariant1Vector(D[ii, i]) ⊗ x
            out[ii] += ∂f∂ξ
        end
    end
    return SArray(out)
end

function apply_operator(op::Gradient{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        x = get_node(arg, ij, slabidx)
        for ii in 1:Nq
            ∂f∂ξ₁ = Geometry.Covariant12Vector(D[ii, i], zero(eltype(D))) ⊗ x
            out[ii, j] = out[ii, j] ⊞ ∂f∂ξ₁
        end
        for jj in 1:Nq
            ∂f∂ξ₂ = Geometry.Covariant12Vector(zero(eltype(D)), D[jj, j]) ⊗ x
            out[i, jj] = out[i, jj] ⊞ ∂f∂ξ₂
        end
    end
    return SArray(out)
end


"""
    wgrad = WeakGradient()
    wgrad.(f)

Compute the "weak gradient" of `f` on each element.

This is defined as the the vector field ``\\theta \\in \\mathcal{V}_0`` such
that for all ``\\phi \\in \\mathcal{V}_0``
```math
\\int_\\Omega \\phi \\cdot \\theta \\, d \\Omega
=
- \\int_\\Omega (\\nabla \\cdot \\phi) f \\, d\\Omega
```
where ``\\mathcal{V}_0`` is the space of ``f``.

This arises from the contribution of the volume integral after by applying
integration by parts to the weak form expression of the gradient
```math
\\int_\\Omega \\phi \\cdot (\\nabla f) \\, d \\Omega
=
- \\int_\\Omega f (\\nabla \\cdot \\phi) \\, d\\Omega
+ \\oint_{\\partial \\Omega} f (\\phi \\cdot n) \\, d \\sigma
```

In matrix form, this becomes
```math
{\\phi^i}^\\top W J \\theta_i = - ( J^{-1} D_i J \\phi^i )^\\top W J f
```
which reduces to
```math
\\theta_i = -W^{-1} D_i^\\top W f
```
where ``D_i`` is the derivative matrix along the ``i``th dimension.
"""
struct WeakGradient{I} <: SpectralElementOperator end
WeakGradient() = WeakGradient{()}()
WeakGradient{()}(space) = WeakGradient{operator_axes(space)}()

operator_return_eltype(::WeakGradient{I}, ::Type{S}) where {I, S} =
    RecursiveApply.rmaptype(T -> Geometry.gradient_result_type(Val(I), T), S)

function apply_operator(op::WeakGradient{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MVector{Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        W = local_geometry.WJ / local_geometry.J
        Wx = W ⊠ get_node(slab_data, i)
        for ii in 1:Nq
            Dᵀ₁Wf = Geometry.Covariant1Vector(D[i, ii]) ⊗ Wx
            out[ii] = out[ii] ⊟ Dᵀ₁Wf
        end
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        W = local_geometry.WJ / local_geometry.J
        out[i] = RecursiveApply.rdiv(out[i], W)
    end
    return SArray(out)
end

function apply_operator(op::WeakGradient{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        W = local_geometry.WJ / local_geometry.J
        Wx = W ⊠ get_node(arg, ij, slabidx)
        for ii in 1:Nq
            Dᵀ₁Wf = Geometry.Covariant12Vector(D[i, ii], zero(eltype(D))) ⊗ Wx
            out[ii, j] = out[ii, j] ⊟ Dᵀ₁Wf
        end
        for jj in 1:Nq
            Dᵀ₂Wf = Geometry.Covariant12Vector(zero(eltype(D)), D[j, jj]) ⊗ Wx
            out[i, jj] = out[i, jj] ⊟ Dᵀ₂Wf
        end
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        W = local_geometry.WJ / local_geometry.J
        out[i, j] = RecursiveApply.rdiv(out[i, j], W)
    end
    return SArray(out)
end

abstract type CurlSpectralElementOperator <: SpectralElementOperator end

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
struct Curl{I} <: CurlSpectralElementOperator end
Curl() = Curl{()}()
Curl{()}(space) = Curl{operator_axes(space)}()

operator_return_eltype(::Curl{I}, ::Type{S}) where {I, S} =
    RecursiveApply.rmaptype(T -> Geometry.curl_result_type(Val(I), T), S)

function apply_operator(op::Curl{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MVector{Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    if RT <: Geometry.Contravariant2Vector
        @inbounds for i in 1:Nq
            ij = CartesianIndex((i,))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(arg, ij, slabidx)
            v₃ = Geometry.covariant3(
                v,
                local_geometry,
            )
            for ii in 1:Nq
                D₁v₃ = D[ii, i] ⊠ v₃
                out[ii] = out[ii] ⊞ Geometry.Contravariant2Vector(⊟(D₁v₃))
            end
        end
    else
        error("invalid return type: $RT")
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[i] = RecursiveApply.rdiv(out[i], local_geometry.J)
    end
    return SArray(out)
end

function apply_operator(op::Curl{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    # input data is a Covariant12Vector field
    if RT <: Geometry.Contravariant3Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i,j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(arg, ij, slabidx)
            v₁ = Geometry.covariant1(v, local_geometry)
            for jj in 1:Nq
                D₂v₁ = D[jj, j] ⊠ v₁
                out[i, jj] =
                    out[i, jj] ⊞ Geometry.Contravariant3Vector(
                        ⊟(D₂v₁)
                    )
            end
            v₂ = Geometry.covariant2(v, local_geometry)
            for ii in 1:Nq
                D₁v₂ = D[ii, i] ⊠ v₂
                out[ii, j] =
                    out[ii, j] ⊞ Geometry.Contravariant3Vector(
                        D₁v₂
                    )
            end
        end
        # input data is a Covariant3Vector field
    elseif RT <: Geometry.Contravariant12Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i,j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(arg, ij, slabidx)
            v₃ = Geometry.covariant3(v, local_geometry)
            for ii in 1:Nq
                D₁v₃ = D[ii, i] ⊠ v₃
                out[ii, j] =
                    out[ii, j] ⊞
                    Geometry.Contravariant12Vector(zero(D₁v₃), ⊟(D₁v₃))
            end
            for jj in 1:Nq
                D₂v₃ = D[jj, j] ⊠ v₃
                out[i, jj] =
                    out[i, jj] ⊞
                    Geometry.Contravariant12Vector(D₂v₃, zero(D₂v₃))
            end
        end
    else
        error("invalid return type: $RT")
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[i, j] = RecursiveApply.rdiv(out[i, j], local_geometry.J)
    end
    return SArray(out)
end

"""
    wcurl = WeakCurl()
    wcurl.(u)

Computes the "weak curl" on each element of a covariant vector field `u`.

Note: The vector field ``u`` needs to be excliclty converted to a `CovaraintVector`,
as then the `WeakCurl` is independent of the local metric tensor.

This is defined as the vector field ``\\theta \\in \\mathcal{V}_0`` such that
for all ``\\phi \\in \\mathcal{V}_0``
```math
\\int_\\Omega \\phi \\cdot \\theta \\, d \\Omega
=
\\int_\\Omega (\\nabla \\times \\phi) \\cdot u \\,d \\Omega
```
where ``\\mathcal{V}_0`` is the space of ``f``.

This arises from the contribution of the volume integral after by applying
integration by parts to the weak form expression of the curl
```math
\\int_\\Omega \\phi \\cdot (\\nabla \\times u) \\,d\\Omega
=
\\int_\\Omega (\\nabla \\times \\phi) \\cdot u \\,d \\Omega
- \\oint_{\\partial \\Omega} (\\phi \\times u) \\cdot n \\,d\\sigma
```

In matrix form, this becomes
```math
{\\phi_i}^\\top W J \\theta^i = (J^{-1} \\epsilon^{kji} D_j \\phi_i)^\\top W J u_k
```
which, by using the anti-symmetry of the Levi-Civita symbol, reduces to
```math
\\theta^i = - \\epsilon^{ijk} (WJ)^{-1} D_j^\\top W u_k
```
"""
struct WeakCurl{I} <: CurlSpectralElementOperator end
WeakCurl() = WeakCurl{()}()
WeakCurl{()}(space) = WeakCurl{operator_axes(space)}()

operator_return_eltype(::WeakCurl{I}, ::Type{S}) where {I, S} =
    RecursiveApply.rmaptype(T -> Geometry.curl_result_type(Val(I), T), S)

function apply_operator(op::WeakCurl{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MVector{Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    # input data is a Covariant3Vector field
    if RT <: Geometry.Contravariant2Vector
        @inbounds for i in 1:Nq
            ij = CartesianIndex((i,))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(arg, ij, slabidx)
            W = local_geometry.WJ / local_geometry.J
            Wv₃ =
                W ⊠ Geometry.covariant3(
                    v,
                    local_geometry,
                )
            for ii in 1:Nq
                Dᵀ₁Wv₃ = D[i, ii] ⊠ Wv₃
                out[ii] = out[ii] ⊞ Geometry.Contravariant2Vector(⊟(Dᵀ₁Wv₃))
            end
        end
    else
        error("invalid return type: $RT")
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[i] = RecursiveApply.rdiv(out[i], local_geometry.WJ)
    end
    return SArray(out)
end

function apply_operator(op::WeakCurl{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    # input data is a Covariant12Vector field
    if RT <: Geometry.Contravariant3Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i,j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(arg, ij, slabidx)
            W = local_geometry.WJ / local_geometry.J
            Wv₁ =
                W ⊠ Geometry.covariant1(v, local_geometry)
            for jj in 1:Nq
                Dᵀ₂Wv₁ = D[j, jj] ⊠ Wv₁
                out[i, jj] =
                    out[i, jj] ⊞ Geometry.Contravariant3Vector(
                        Dᵀ₂Wv₁,
                    )
            end
            Wv₂ =
                W ⊠ Geometry.covariant2(v, local_geometry)
            for ii in 1:Nq
                Dᵀ₁Wv₂ = D[i, ii] ⊠ Wv₂
                out[ii, j] =
                    out[ii, j] ⊞ Geometry.Contravariant3Vector(
                        ⊟(Dᵀ₁Wv₂),
                    )
            end
        end
        # input data is a Covariant3Vector field
    elseif RT <: Geometry.Contravariant12Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i,j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(arg, ij, slabidx)
            W = local_geometry.WJ / local_geometry.J
            Wv₃ =
                W ⊠ Geometry.covariant3(v, local_geometry)
            for ii in 1:Nq
                Dᵀ₁Wv₃ = D[i, ii] ⊠ Wv₃
                out[ii, j] =
                    out[ii, j] ⊞
                    Geometry.Contravariant12Vector(zero(Dᵀ₁Wv₃), Dᵀ₁Wv₃)
            end
            for jj in 1:Nq
                Dᵀ₂Wv₃ = D[j, jj] ⊠ Wv₃
                out[i, jj] =
                    out[i, jj] ⊞
                    Geometry.Contravariant12Vector(⊟(Dᵀ₂Wv₃), zero(Dᵀ₂Wv₃))
            end
        end
    else
        error("invalid return type: $RT")
    end
    for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i,j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[i, j] = RecursiveApply.rdiv(out[i, j], local_geometry.WJ)
    end
    return SArray(out)
end

# interplation / restriction
abstract type TensorOperator <: SpectralElementOperator end

input_space(op::TensorOperator, inspace) = inspace
return_space(op::TensorOperator, inspace) = op.space
operator_return_eltype(::TensorOperator, ::Type{S}) where {S} = S

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
struct Interpolate{I, S} <: TensorOperator
    space::S
end
Interpolate(space) = Interpolate{operator_axes(space), typeof(space)}(space)

function apply_slab(
    op::Interpolate{(1,)},
    slab_space_out,
    slab_space_in,
    slab_data,
)
    FT = Spaces.undertype(slab_space_out)
    QS_in = Spaces.quadrature_style(slab_space_in)
    QS_out = Spaces.quadrature_style(slab_space_out)
    Nq_in = Quadratures.degrees_of_freedom(QS_in)
    Nq_out = Quadratures.degrees_of_freedom(QS_out)
    Imat = Quadratures.interpolation_matrix(FT, QS_out, QS_in)
    S = eltype(slab_data)
    slab_data_out = MVector{Nq_out, S}(undef)
    @inbounds for i in 1:Nq_out
        # manually inlined rmatmul with slab_getnode
        r = Imat[i, 1] ⊠ get_node(slab_data, 1)
        for ii in 2:Nq_in
            r = RecursiveApply.rmuladd(Imat[i, ii], get_node(slab_data, ii), r)
        end
        slab_data_out[i] = r
    end
    return slab_data_out
end

function apply_slab(
    op::Interpolate{(1, 2)},
    slab_space_out,
    slab_space_in,
    slab_data,
)
    FT = Spaces.undertype(slab_space_out)
    QS_in = Spaces.quadrature_style(slab_space_in)
    QS_out = Spaces.quadrature_style(slab_space_out)
    Nq_in = Quadratures.degrees_of_freedom(QS_in)
    Nq_out = Quadratures.degrees_of_freedom(QS_out)
    Imat = Quadratures.interpolation_matrix(FT, QS_out, QS_in)
    S = eltype(slab_data)
    # temporary storage
    temp = MArray{Tuple{Nq_out, Nq_in}, S, 2, Nq_out * Nq_in}(undef)
    slab_data_out = MArray{Tuple{Nq_out, Nq_out}, S, 2, Nq_out * Nq_out}(undef)
    @inbounds for j in 1:Nq_in, i in 1:Nq_out
        # manually inlined rmatmul1 with slab get_node
        # we do this to remove one allocated intermediate array
        r = Imat[i, 1] ⊠ get_node(slab_data, 1, j)
        for ii in 2:Nq_in
            r = RecursiveApply.rmuladd(
                Imat[i, ii],
                get_node(slab_data, ii, j),
                r,
            )
        end
        temp[i, j] = r
    end
    @inbounds for j in 1:Nq_out, i in 1:Nq_out
        slab_data_out[i, j] = RecursiveApply.rmatmul2(Imat, temp, i, j)
    end
    return SMatrix(slab_data_out)
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
struct Restrict{I, S} <: TensorOperator
    space::S
end
Restrict(space) = Restrict{operator_axes(space), typeof(space)}(space)

function apply_slab(
    op::Restrict{(1,)},
    slab_space_out,
    slab_space_in,
    slab_data,
)
    FT = Spaces.undertype(slab_space_out)
    QS_in = Spaces.quadrature_style(slab_space_in)
    QS_out = Spaces.quadrature_style(slab_space_out)
    Nq_in = Quadratures.degrees_of_freedom(QS_in)
    Nq_out = Quadratures.degrees_of_freedom(QS_out)
    ImatT = Quadratures.interpolation_matrix(FT, QS_in, QS_out)' # transpose
    S = eltype(slab_data)
    slab_data_out = MVector{Nq_out, S}(undef)
    slab_local_geometry_in = Spaces.local_geometry_data(slab_space_in)
    slab_local_geometry_out = Spaces.local_geometry_data(slab_space_out)
    WJ_in = slab_local_geometry_in.WJ
    WJ_out = slab_local_geometry_out.WJ
    @inbounds for i in 1:Nq_out
        # manually inlined rmatmul with slab get_node
        r = ImatT[i, 1] ⊠ (WJ_in[1] ⊠ get_node(slab_data, 1))
        for ii in 2:Nq_in
            WJ_node = WJ_in[ii] ⊠ get_node(slab_data, ii)
            r = RecursiveApply.rmuladd(ImatT[i, ii], WJ_node, r)
        end
        slab_data_out[i] = RecursiveApply.rdiv(r, WJ_out[i])
    end
    return slab_data_out
end

function apply_slab(
    op::Restrict{(1, 2)},
    slab_space_out,
    slab_space_in,
    slab_data,
)
    FT = Spaces.undertype(slab_space_out)
    QS_in = Spaces.quadrature_style(slab_space_in)
    QS_out = Spaces.quadrature_style(slab_space_out)
    Nq_in = Quadratures.degrees_of_freedom(QS_in)
    Nq_out = Quadratures.degrees_of_freedom(QS_out)
    ImatT = Quadratures.interpolation_matrix(FT, QS_in, QS_out)' # transpose
    S = eltype(slab_data)
    # temporary storage
    temp = MArray{Tuple{Nq_out, Nq_in}, S, 2, Nq_out * Nq_in}(undef)
    slab_data_out = MArray{Tuple{Nq_out, Nq_out}, S, 2, Nq_out * Nq_out}(undef)
    slab_local_geometry_in = Spaces.local_geometry_data(slab_space_in)
    slab_local_geometry_out = Spaces.local_geometry_data(slab_space_out)
    WJ_in = slab_local_geometry_in.WJ
    WJ_out = slab_local_geometry_out.WJ
    @inbounds for j in 1:Nq_in, i in 1:Nq_out
        # manually inlined rmatmul1 with slab get_node
        r = ImatT[i, 1] ⊠ (WJ_in[1, j] ⊠ get_node(slab_data, 1, j))
        for ii in 2:Nq_in
            WJ_node = WJ_in[ii, j] ⊠ get_node(slab_data, ii, j)
            r = RecursiveApply.rmuladd(ImatT[i, ii], WJ_node, r)
        end
        temp[i, j] = r
    end
    @inbounds for j in 1:Nq_out, i in 1:Nq_out
        slab_data_out[i, j] = RecursiveApply.rdiv(
            RecursiveApply.rmatmul2(ImatT, temp, i, j),
            WJ_out[i, j],
        )
    end
    return SMatrix(slab_data_out)
end


"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element.
"""
function tensor_product! end

function tensor_product!(
    out::DataLayouts.Data1DX{S, Ni_out},
    in::DataLayouts.Data1DX{S, Ni_in},
    M::SMatrix{Ni_out, Ni_in},
) where {S, Ni_out, Ni_in}
    (_, _, _, Nv_in, Nh_in) = size(in)
    (_, _, _, Nv_out, Nh_out) = size(out)
    # TODO: assumes the same number of levels (horizontal only)
    @assert Nv_in == Nv_out
    @assert Nh_in == Nh_out
    for h in 1:Nh_out, v in 1:Nv_out
        in_slab = slab(in, v, h)
        out_slab = slab(out, v, h)
        @inbounds for i in 1:Ni_out
            r = M[i, 1] ⊠ in_slab[1]
            for ii in 2:Ni_in
                r = RecursiveApply.rmuladd(M[i, ii], in_slab[ii], r)
            end
            out_slab[i] = r
        end
    end
    return out
end

function tensor_product!(
    out::DataLayouts.Data2D{S, Nij_out},
    in::DataLayouts.Data2D{S, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}

    Nh = length(in)
    @assert Nh == length(out)

    # temporary storage
    temp = MArray{Tuple{Nij_out, Nij_in}, S, 2, Nij_out * Nij_in}(undef)

    for h in 1:Nh
        in_slab = slab(in, h)
        out_slab = slab(out, h)
        for j in 1:Nij_in, i in 1:Nij_out
            temp[i, j] = RecursiveApply.rmatmul1(M, in_slab, i, j)
        end
        for j in 1:Nij_out, i in 1:Nij_out
            out_slab[i, j] = RecursiveApply.rmatmul2(M, temp, i, j)
        end
    end
    return out
end

function tensor_product!(
    out_slab::DataLayouts.DataSlab2D{S, Nij_out},
    in_slab::DataLayouts.DataSlab2D{S, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}
    # temporary storage
    temp = MArray{Tuple{Nij_out, Nij_in}, S, 2, Nij_out * Nij_in}(undef)
    for j in 1:Nij_in, i in 1:Nij_out
        temp[i, j] = RecursiveApply.rmatmul1(M, in_slab, i, j)
    end
    for j in 1:Nij_out, i in 1:Nij_out
        out_slab[i, j] = RecursiveApply.rmatmul2(M, temp, i, j)
    end
    return out_slab
end

function tensor_product!(
    inout::Data2D{S, Nij},
    M::SMatrix{Nij, Nij},
) where {S, Nij}
    tensor_product!(inout, inout, M)
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
        DataLayouts.IH1JH2{S, Nu}(Matrix{S}(undef, (Nu * n1, Nu * n2)))
    M = Quadratures.interpolation_matrix(Float64, Q_interp, quadrature_style)
    Operators.tensor_product!(interp_data, Fields.field_values(field), M)
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
    interp_data = DataLayouts.IV1JH2{S, Nu}(Matrix{S}(undef, (nl, Nu * n1)))
    M = Quadratures.interpolation_matrix(Float64, Q_interp, quadrature_style)
    Operators.tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end

"""
    matrix_interpolate(field, Nu::Integer)

Computes the tensor product given a uniform quadrature degree of Nu on each element.
Returns a 2D Matrix for plotting / visualizing 2D Fields.
"""
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())
