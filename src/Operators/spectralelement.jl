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
    isnothing(sbc.input_space) ? input_space(sbc.op, map(axes, sbc.args)...) :
    sbc.input_space

return_space(::SpectralElementOperator, space) = space

Base.axes(sbc::SpectralBroadcasted) =
    isnothing(sbc.axes) ? return_space(sbc.op, map(axes, sbc.args)...) :
    sbc.axes

Base.Broadcast.broadcasted(op::SpectralElementOperator, args...) =
    Base.Broadcast.broadcasted(SpectralStyle(), op, args...)

Base.Broadcast.broadcasted(
    ::SpectralStyle,
    op::SpectralElementOperator,
    args...,
) = SpectralBroadcasted{SpectralStyle}(op, args)

Base.eltype(sbc::SpectralBroadcasted) =
    operator_return_eltype(sbc.op, map(eltype, sbc.args)...)

function Base.Broadcast.instantiate(
    sbc::SpectralBroadcasted{Style},
) where {Style}
    op = sbc.op
    # recursively instantiate the arguments to allocate intermediate work arrays
    args = map(Base.Broadcast.instantiate, sbc.args)
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
    args = map(Base.Broadcast.instantiate, bc.args)
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

function Base.copyto!(field_out::Field, sbc::SpectralBroadcasted)
    data_out = Fields.field_values(field_out)
    space = axes(field_out)
    Nv = Spaces.nlevels(space)
    Nh = Topologies.nlocalelems(Spaces.topology(space))
    for h in 1:Nh, v in 1:Nv
        slab_out = slab(field_out, v, h)
        out_slab_space = slab(axes(sbc), v, h)
        in_slab_space = slab(input_space(sbc), v, h)
        slab_args = map(arg -> _apply_slab(slab(arg, v, h), v, h), sbc.args)
        copy_slab!(
            slab_out,
            apply_slab(sbc.op, out_slab_space, in_slab_space, slab_args...),
        )
    end
    return field_out
end

function Base.Broadcast.materialize!(dest, sbc::SpectralBroadcasted)
    copyto!(dest, Base.Broadcast.instantiate(sbc))
end

function slab(
    sbc::SpectralBroadcasted{Style},
    inds...,
) where {Style <: SpectralStyle}
    _args = map(a -> slab(a, inds...), sbc.args)
    _axes = slab(axes(sbc), inds...)
    SpectralBroadcasted{Style}(sbc.op, _args, _axes, sbc.input_space)
end

function slab(
    bc::Base.Broadcast.Broadcasted{Style},
    inds...,
) where {Style <: AbstractSpectralStyle}
    _args = map(a -> slab(a, inds...), bc.args)
    _axes = slab(axes(bc), inds...)
    Base.Broadcast.Broadcasted{Style}(bc.f, _args, _axes)
end

function Base.copyto!(
    field_out::Field,
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractSpectralStyle}
    data_out = Fields.field_values(field_out)
    space = axes(field_out)
    Nv = Spaces.nlevels(space)
    Nh = Topologies.nlocalelems(Spaces.topology(space))
    for h in 1:Nh, v in 1:Nv
        slab_out = slab(field_out, v, h)
        slab_args = map(arg -> _apply_slab(slab(arg, v, h), v, h), bc.args)
        copy_slab!(
            slab_out,
            Base.Broadcast.Broadcasted{Style}(bc.f, slab_args, axes(slab_out)),
        )
    end
    return field_out
end

function copy_slab!(slab_out::Fields.SlabField1D, res)
    space = axes(slab_out)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)
    @inbounds for i in 1:Nq
        set_node!(slab_out, i, get_node(res, i))
    end
    return slab_out
end

function copy_slab!(slab_out::Fields.SlabField2D, res)
    space = axes(slab_out)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)
    @inbounds for j in 1:Nq, i in 1:Nq
        set_node!(slab_out, i, j, get_node(res, i, j))
    end
    return slab_out
end

# 1D get/set node

# 1D intermediate slab data types
Base.Base.@propagate_inbounds function get_node(v::MVector, i)
    v[i]
end

Base.Base.@propagate_inbounds function get_node(v::SVector, i)
    v[i]
end

Base.Base.@propagate_inbounds function get_node(scalar, i)
    scalar[]
end

Base.@propagate_inbounds function get_node(field::Fields.SlabField1D, i)
    getindex(Fields.field_values(field), i)
end

@inline function get_node(bc::Base.Broadcast.Broadcasted, i)
    args = map(arg -> get_node(arg, i), bc.args)
    bc.f(args...)
end

Base.@propagate_inbounds set_node!(field::Fields.SlabField1D, i, val) =
    setindex!(Fields.field_values(field), val, i)

# 2D get/set node
Base.@propagate_inbounds function get_node(v::MMatrix, i, j)
    v[i, j]
end

Base.@propagate_inbounds function get_node(v::SMatrix, i, j)
    v[i, j]
end

Base.@propagate_inbounds function get_node(scalar, i, j)
    scalar[]
end

Base.@propagate_inbounds function get_node(field::Fields.SlabField2D, i, j)
    getindex(Fields.field_values(field), i, j)
end

@inline function get_node(bc::Base.Broadcast.Broadcasted, i, j)
    args = map(arg -> get_node(arg, i, j), bc.args)
    bc.f(args...)
end

Base.@propagate_inbounds function set_node!(
    field::Fields.SlabField2D,
    i,
    j,
    val,
)
    setindex!(Fields.field_values(field), val, i, j)
end

# Broadcast recursive machinery
_apply_slab(x, inds...) = x

_apply_slab(sbc::SpectralBroadcasted, inds...) = apply_slab(
    sbc.op,
    slab(axes(sbc), inds...),
    slab(input_space(sbc), inds...),
    map(a -> _apply_slab(a, inds...), sbc.args)...,
)

_apply_slab(bc::Base.Broadcast.Broadcasted{CompositeSpectralStyle}, inds...) =
    Base.Broadcast.Broadcasted{CompositeSpectralStyle}(
        bc.f,
        map(a -> _apply_slab(a, inds...), bc.args),
        bc.axes,
    )

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
    Divergence()

Computes the "strong" divergence of a vector field `v`.

We compute the divergence as

    [∂(Jv¹)/∂ξ¹ + ∂(Jv²)/∂ξ²] / J

where `J` is the Jacobian determinant, `vⁱ` is the `i`th contravariant component of `v`.

This is discretized at the quadrature points as

    I{[∂(I{Jv¹})/∂ξ¹ + ∂(I{Jv²})/∂ξ²] / J}

where `I{x}` is the interpolation operator applied to a field `x`.

## References
 - Taylor and Fournier (2010), equation 15
"""
struct Divergence{I} <: SpectralElementOperator end
Divergence() = Divergence{()}()
Divergence{()}(space) = Divergence{operator_axes(space)}()

operator_return_eltype(op::Divergence, S) =
    RecursiveApply.rmaptype(Geometry.divergence_result_type, S)

function apply_slab(op::Divergence{(1,)}, slab_space, _, slab_data)
    slab_local_geometry = Spaces.local_geometry_data(slab_space)
    FT = Spaces.undertype(slab_space)
    QS = Spaces.quadrature_style(slab_space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(slab_data))
    out = StaticArrays.MVector{Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for i in 1:Nq
        local_geometry = slab_local_geometry[i]
        Jv¹ =
            local_geometry.J ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant1(v, local_geometry),
                get_node(slab_data, i),
            )
        for ii in 1:Nq
            out[ii] = out[ii] ⊞ (D[ii, i] ⊠ Jv¹)
        end
    end
    @inbounds for i in 1:Nq
        local_geometry = slab_local_geometry[i]
        out[i] = RecursiveApply.rdiv(out[i], local_geometry.J)
    end
    return SVector(out)
end

function apply_slab(op::Divergence{(1, 2)}, slab_space, _, slab_data)
    slab_local_geometry = Spaces.local_geometry_data(slab_space)
    FT = Spaces.undertype(slab_space)
    QS = Spaces.quadrature_style(slab_space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(slab_data))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for j in 1:Nq, i in 1:Nq
        local_geometry = slab_local_geometry[i, j]
        Jv¹ =
            local_geometry.J ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant1(v, local_geometry),
                get_node(slab_data, i, j),
            )
        for ii in 1:Nq
            out[ii, j] = out[ii, j] ⊞ (D[ii, i] ⊠ Jv¹)
        end
        Jv² =
            local_geometry.J ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant2(v, local_geometry),
                get_node(slab_data, i, j),
            )
        for jj in 1:Nq
            out[i, jj] = out[i, jj] ⊞ (D[jj, j] ⊠ Jv²)
        end
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        local_geometry = slab_local_geometry[i, j]
        out[i, j] = RecursiveApply.rdiv(out[i, j], local_geometry.J)
    end
    return SMatrix(out)
end


"""
    WeakDivergence()

Computes the divergence of a vector field `v` weakly.

This spolves the variational problem of finding `θ` such that

    ⟨ϕ, J θ⟩ = - ⟨∂ϕ/∂ξ¹, J v¹⟩ + ⟨∂ϕ/∂ξ², J v²⟩

for all `ϕ` (which arises by integration by parts).

Discretely it is equivalent to

    - J \\ (D₁' * W * J * v¹ + D₂' * W * J * v²)

where

 - `J` is the diagonal Jacobian matrix
 - `W` is the diagonal matrix of quadrature weights
 - `D₁` and `D₂` are the discrete derivative matrices along the first and second dimensions.
"""
struct WeakDivergence{I} <: SpectralElementOperator end
WeakDivergence() = WeakDivergence{()}()
WeakDivergence{()}(space) = WeakDivergence{operator_axes(space)}()

operator_return_eltype(op::WeakDivergence, S) =
    RecursiveApply.rmaptype(Geometry.divergence_result_type, S)

function apply_slab(op::WeakDivergence{(1, 2)}, slab_space, _, slab_data)
    slab_local_geometry = Spaces.local_geometry_data(slab_space)
    FT = Spaces.undertype(slab_space)
    QS = Spaces.quadrature_style(slab_space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    RT = operator_return_eltype(op, eltype(slab_data))
    # allocate temp output
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for j in 1:Nq, i in 1:Nq
        local_geometry = slab_local_geometry[i, j]
        WJv¹ =
            local_geometry.WJ ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant1(v, local_geometry),
                get_node(slab_data, i, j),
            )
        for ii in 1:Nq
            out[ii, j] = out[ii, j] ⊞ (D[i, ii] ⊠ WJv¹)
        end
        WJv² =
            local_geometry.WJ ⊠ RecursiveApply.rmap(
                v -> Geometry.contravariant2(v, local_geometry),
                get_node(slab_data, i, j),
            )
        for jj in 1:Nq
            out[i, jj] = out[i, jj] ⊞ (D[j, jj] ⊠ WJv²)
        end
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        local_geometry = slab_local_geometry[i, j]
        out[i, j] = RecursiveApply.rdiv(out[i, j], ⊟(local_geometry.WJ))
    end
    return SMatrix(out)
end

"""
    Gradient()

Compute the (strong) gradient on each element via the chain rule:

    ∂f/∂xⁱ = ∂f/∂ξʲ * ∂ξʲ/∂xⁱ
"""
struct Gradient{I} <: SpectralElementOperator end
Gradient() = Gradient{()}()
Gradient{()}(space) = Gradient{operator_axes(space)}()

function operator_return_eltype(::Gradient{I}, ::Type{T}) where {I, T <: Number}
    N = length(I)
    Geometry.AxisVector{T, Geometry.CovariantAxis{I}, SVector{N, T}}
end
function operator_return_eltype(
    ::Gradient{I},
    ::Type{V},
) where {I, V <: Geometry.AxisVector{T, A, SVector{N, T}}} where {T, A, N}
    M = length(I)
    Geometry.Axis2Tensor{
        T,
        Tuple{Geometry.CovariantAxis{I}, A},
        SMatrix{M, N, T, M * N},
    }
end
operator_return_eltype(grad::Gradient, S) where {I} =
    RecursiveApply.rmaptype(T -> operator_return_eltype(grad, T), S)

function apply_slab(op::Gradient{(1,)}, slab_space, _, slab_data)
    FT = Spaces.undertype(slab_space)
    QS = Spaces.quadrature_style(slab_space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(slab_data))
    out = StaticArrays.MVector{Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for i in 1:Nq
        x = get_node(slab_data, i)
        for ii in 1:Nq
            ∂f∂ξ = Geometry.Covariant1Vector(D[ii, i]) ⊗ x
            out[ii] += ∂f∂ξ
        end
    end
    return SVector(out)
end

function apply_slab(op::Gradient{(1, 2)}, slab_space, _, slab_data)
    FT = Spaces.undertype(slab_space)
    QS = Spaces.quadrature_style(slab_space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(slab_data))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for j in 1:Nq, i in 1:Nq
        x = get_node(slab_data, i, j)
        for ii in 1:Nq
            ∂f∂ξ₁ = Geometry.Covariant12Vector(D[ii, i], zero(eltype(D))) ⊗ x
            out[ii, j] = out[ii, j] ⊞ ∂f∂ξ₁
        end
        for jj in 1:Nq
            ∂f∂ξ₂ = Geometry.Covariant12Vector(zero(eltype(D)), D[jj, j]) ⊗ x
            out[i, jj] = out[i, jj] ⊞ ∂f∂ξ₂
        end
    end
    return SMatrix(out)
end




"""
    WeakGradient()

Compute the (strong) gradient on each element via the chain rule:

    ∂f/∂xⁱ = ∂f/∂ξʲ * ∂ξʲ/∂xⁱ
"""
struct WeakGradient{I} <: SpectralElementOperator end
WeakGradient() = WeakGradient{()}()
WeakGradient{()}(space) = WeakGradient{operator_axes(space)}()

operator_return_eltype(op::WeakGradient{(1, 2)}, S) =
    RecursiveApply.rmaptype(T -> Covariant12Vector{T}, S)

function apply_slab(op::WeakGradient{(1,)}, slab_space, _, slab_data)
    slab_local_geometry = Spaces.local_geometry_data(slab_space)
    FT = Spaces.undertype(slab_space)
    QS = Spaces.quadrature_style(slab_space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(slab_data))
    out = StaticArrays.MVector{Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for i in 1:Nq
        local_geometry = slab_local_geometry[i]
        W = local_geometry.WJ / local_geometry.J
        Wx = W ⊠ get_node(slab_data, i)
        for ii in 1:Nq
            Dᵀ₁Wf = Geometry.Covariant1Vector(D[i, ii]) ⊗ Wx
            out[ii] = out[ii] ⊟ Dᵀ₁Wf
        end
    end
    @inbounds for i in 1:Nq
        local_geometry = slab_local_geometry[i]
        W = local_geometry.WJ / local_geometry.J
        out[i] = RecursiveApply.rdiv(out[i], W)
    end
    return SVector(out)
end

function apply_slab(op::WeakGradient{(1, 2)}, slab_space, _, slab_data)
    slab_local_geometry = Spaces.local_geometry_data(slab_space)
    FT = Spaces.undertype(slab_space)
    QS = Spaces.quadrature_style(slab_space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(slab_data))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    @inbounds for j in 1:Nq, i in 1:Nq
        local_geometry = slab_local_geometry[i, j]
        W = local_geometry.WJ / local_geometry.J
        Wx = W ⊠ get_node(slab_data, i, j)
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
        local_geometry = slab_local_geometry[i, j]
        W = local_geometry.WJ / local_geometry.J
        out[i, j] = RecursiveApply.rdiv(out[i, j], W)
    end
    return SMatrix(out)
end

abstract type CurlSpectralElementOperator <: SpectralElementOperator end

"""
    Curl()

Compute the (strong) curl on each element
"""
struct Curl{I} <: CurlSpectralElementOperator end
Curl() = Curl{()}()
Curl{()}(space) = Curl{operator_axes(space)}()

operator_return_eltype(::Curl{(1, 2)}, S) =
    RecursiveApply.rmaptype(T -> Geometry.curl_result_type(T), S)

function apply_slab(op::Curl{(1, 2)}, slab_space, _, slab_data)
    slab_local_geometry = Spaces.local_geometry_data(slab_space)
    FT = Spaces.undertype(slab_space)
    QS = Spaces.quadrature_style(slab_space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(slab_data))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    if RT <: Geometry.Contravariant3Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            v₁ = Geometry.covariant1(
                get_node(slab_data, i, j),
                slab_local_geometry[i, j],
            )
            for jj in 1:Nq
                D₂v₁ = D[jj, j] ⊠ v₁
                out[i, jj] =
                    out[i, jj] ⊞ Geometry.Contravariant3Vector(
                        ⊟(D₂v₁),
                        slab_local_geometry[i, jj],
                    )
            end
            v₂ = Geometry.covariant2(
                get_node(slab_data, i, j),
                slab_local_geometry[i, j],
            )
            for ii in 1:Nq
                D₁v₂ = D[ii, i] ⊠ v₂
                out[ii, j] =
                    out[ii, j] ⊞ Geometry.Contravariant3Vector(
                        D₁v₂,
                        slab_local_geometry[ii, j],
                    )
            end
        end
    elseif RT <: Geometry.Contravariant12Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            v₃ = Geometry.covariant3(
                get_node(slab_data, i, j),
                slab_local_geometry[i, j],
            )
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
        local_geometry = slab_local_geometry[i, j]
        out[i, j] = RecursiveApply.rdiv(out[i, j], local_geometry.J)
    end
    return SMatrix(out)
end


"""
    WeakCurl()

Compute the weak curl on each element
"""
struct WeakCurl{I} <: CurlSpectralElementOperator end
WeakCurl() = WeakCurl{()}()
WeakCurl{()}(space) = WeakCurl{operator_axes(space)}()

operator_return_eltype(::WeakCurl{(1, 2)}, S) =
    RecursiveApply.rmaptype(T -> Geometry.curl_result_type(T), S)

function apply_slab(op::WeakCurl{(1, 2)}, slab_space, _, slab_data)
    slab_local_geometry = Spaces.local_geometry_data(slab_space)
    FT = Spaces.undertype(slab_space)
    QS = Spaces.quadrature_style(slab_space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(slab_data))
    out = StaticArrays.MMatrix{Nq, Nq, RT}(undef)
    DataLayouts._mzero!(out, FT)
    if RT <: Geometry.Contravariant3Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            local_geometry = slab_local_geometry[i, j]
            W = local_geometry.WJ / local_geometry.J
            Wv₁ =
                W ⊠ Geometry.covariant1(
                    get_node(slab_data, i, j),
                    slab_local_geometry[i, j],
                )
            for jj in 1:Nq
                Dᵀ₂Wv₁ = D[j, jj] ⊠ Wv₁
                out[i, jj] =
                    out[i, jj] ⊞ Geometry.Contravariant3Vector(
                        Dᵀ₂Wv₁,
                        slab_local_geometry[i, jj],
                    )
            end
            Wv₂ =
                W ⊠ Geometry.covariant2(
                    get_node(slab_data, i, j),
                    slab_local_geometry[i, j],
                )
            for ii in 1:Nq
                Dᵀ₁Wv₂ = D[i, ii] ⊠ Wv₂
                out[ii, j] =
                    out[ii, j] ⊞ Geometry.Contravariant3Vector(
                        ⊟(Dᵀ₁Wv₂),
                        slab_local_geometry[ii, j],
                    )
            end
        end
    elseif RT <: Geometry.Contravariant12Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            local_geometry = slab_local_geometry[i, j]
            W = local_geometry.WJ / local_geometry.J
            Wv₃ =
                W ⊠ Geometry.covariant3(
                    get_node(slab_data, i, j),
                    slab_local_geometry[i, j],
                )
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
        local_geometry = slab_local_geometry[i, j]
        out[i, j] = RecursiveApply.rdiv(out[i, j], local_geometry.WJ)
    end
    return SMatrix(out)
end

# interplation / restriction
abstract type TensorOperator <: SpectralElementOperator end

input_space(op::TensorOperator, inspace) = inspace
return_space(op::TensorOperator, inspace) = op.space
operator_return_eltype(op::TensorOperator, S) = S

"""
    Interpolate(space)

Computes the projection of a field to a higher degree polynomial space. `space`
must be on the same element mesh as the field, but have equal or higher
polynomial degree.

    ⟨ϕ, J θ⟩ = ⟨ϕ, J σ⟩

where `ϕ` and `θ` are on the higher degree space.

Discretely it is equivalent to

    I σ

where `I` is the interpolation matrix.
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
    Restrict(space)

Computes the projection of a field to a lower degree polynomial space. `space`
must be on the same element mesh as the field, but have lower polynomial degree.

    ⟨ϕ, J θ⟩ = ⟨ϕ, J σ⟩

where `ϕ` and `θ` are on the lower degree space.

Discretely it is equivalent to

    (JWr) \\  I' (JW) σ

where `I` is the interpolation matrix, and `JWr` is the Jacobian multiplied by
quadrature weights on the lower-degree space.
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
function tensor_product!(
    out::Data2D{S, Nij_out},
    in::Data2D{S, Nij_in},
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

"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element.
"""
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
    slab_gradient!(∇data, data, space)

Compute the gradient on each element via the chain rule:

    ∂f/∂xⁱ = ∂f/∂ξʲ * ∂ξʲ/∂xⁱ
"""
function slab_gradient!(∇data, data, space)
    # all derivatives calculated in the reference local geometry FT precision
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    # for each element in the element stack
    Nh = length(data)
    for h in 1:Nh
        ∇data_slab = slab(∇data, h)
        # TODO: can we just call materialize(slab(data,h))
        data_slab = slab(data, h)
        local_geometry_slab = slab(space.local_geometry, h)

        for i in 1:Nq, j in 1:Nq
            # TODO: materialize data_slab once
            # on GPU this would be done to shared memory, then synchronize threads
            local_geometry = local_geometry_slab[i, j]

            # compute covariant derivatives
            ∂f∂ξ₁ = RecursiveApply.rmatmul1(D, data_slab, i, j)
            ∂f∂ξ₂ = RecursiveApply.rmatmul2(D, data_slab, i, j)
            ∂f∂ξ = RecursiveApply.rmap(Covariant12Vector, ∂f∂ξ₁, ∂f∂ξ₂)

            # convert to desired basis
            ∇data_slab[i, j] = RecursiveApply.rmap(
                x -> Cartesian12Vector(x, local_geometry),
                ∂f∂ξ,
            )
        end
    end
    return ∇data
end


"""
    slab_divergence!(divflux, flux, space)

Compute the divergence of `flux`, storing the result in `divflux`.

Given a vector field `v`, we compute the divergence as

    [∂(Jv¹)/∂ξ¹ + ∂(Jv²)/∂ξ²] / J

where `J` is the Jacobian determinant, `vⁱ` is the `i`th contravariant component of `v`.

This is discretized at the quadrature points as

    I{[∂(I{Jv¹})/∂ξ¹ + ∂(I{Jv²})/∂ξ²] / J}

where `I{x}` is the interpolation operator applied to a field `x`.

## References
 - Taylor and Fournier (2010), equation 15
"""
function slab_divergence!(divflux, flux, space)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        divflux_slab = slab(divflux, h)
        flux_slab = slab(flux, h)
        local_geometry_slab = slab(space.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        Jv¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        Jv² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute flux in contravariant coordinates (v¹,v²)
            # alternatively we could do this conversion _after_ taking the derivatives
            # may have an effect on the accuracy
            # materialize if lazy
            F = flux_slab[i, j]  # materialize the flux
            Jv¹[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.J *
                    Geometry.contravariant1(x, local_geometry),
                F,
            )
            Jv²[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.J *
                    Geometry.contravariant2(x, local_geometry),
                F,
            )
        end
        # GPU synchronize
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            ∂₁v₂ = RecursiveOperators.rmatmul1(D, Jv¹, i, j) # ∂(Jv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
            # compute spectral deriv along second dimesion
            ∂₂v₁ = RecursiveOperators.rmatmul2(D, Jv², i, j)
            divflux_slab[i, j] =
                Contravariant3Vector(inv(local_geometry.J) ⊠ (∂₁v₂ ⊟ ∂₂v₁))
        end
    end
    return divflux
end

function slab_strong_curl!(curlvec, vec, mesh)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Meshes.undertype(mesh)
    D = Quadratures.differentiation_matrix(FT, mesh.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(mesh.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        curlvec = slab(curlvec, h)
        vec_slab = slab(vec, h)
        local_geometry_slab = slab(mesh.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        v₁ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        v₂ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # we convert the vec to covariant coordinates (v₁, v₂),
            # and return an orthogonal vector in contravariant coordinates v³
            v₁[i, j] = RecursiveOperators.rmap(
                x -> Geometry.covariant1(x, local_geometry),
                vec_slab,
            )
            v₂[i, j] = RecursiveOperators.rmap(
                x -> Geometry.covariant2(x, local_geometry),
                vec_slab,
            )
        end
        # GPU synchronize
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            ∂₁v₂ = RecursiveOperators.rmatmul1(D, v₂, i, j)
            # compute spectral deriv along second dimension
            ∂₂v₁ = RecursiveOperators.rmatmul2(D, v₁, i, j)
            curlvec_slab[i, j] =
                Contravariant3Vector(inv(local_geometry.J) ⊠ (∂₁v₂ ⊟ ∂₂v₁))
        end
    end
    return curlvec
end

function slab_weak_curl!(curlvec, vec, mesh)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Meshes.undertype(mesh)
    D = Quadratures.differentiation_matrix(FT, mesh.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(mesh.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        curlvec = slab(curlvec, h)
        vec_slab = slab(vec, h)
        local_geometry_slab = slab(mesh.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        v₁ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        v₂ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # we convert the vec to covariant coordinates (v₁, v₂),
            # and return an orthogonal vector in contravariant coordinates v³
            v₁[i, j] = RecursiveOperators.rmap(
                x -> Geometry.covariant1(x, local_geometry),
                vec_slab,
            )
            v₂[i, j] = RecursiveOperators.rmap(
                x -> Geometry.covariant2(x, local_geometry),
                vec_slab,
            )
        end
        # GPU synchronize
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            ∂₁v₂ = RecursiveOperators.rmatmul1(D, v₂, i, j)
            # compute spectral deriv along second dimension
            ∂₂v₁ = RecursiveOperators.rmatmul2(D, v₁, i, j)
            curlvec_slab[i, j] =
                Contravariant3Vector(inv(local_geometry.J) ⊠ (∂₁v₂ ⊟ ∂₂v₁))
        end
    end
    return curlvec
end


"""
    slab_weak_divergence!(divflux, flux, space)

Compute the right-hand side of the divergence of `flux` weakly, storing the result in `divflux`.

This computes the right-hand side of the variational problem of finding θ such that

    - ⟨ϕ, J θ⟩ = ⟨∂ϕ/∂ξ¹, J u¹⟩ + ⟨∂ϕ/∂ξ², J u²⟩

for all `ϕ` (which arises by integration by parts).

Discretely it is equivalent to

    (D₁' * W * J * u¹ + D₂' * W * J * u²)

where
 - `J` is the diagonal Jacobian matrix
 - `W` is the diagonal matrix of quadrature weights
 - `D₁` and `D₂` are the discrete derivative matrices along the first and second dimensions.
"""
function slab_weak_divergence!(divflux, flux, space)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        divflux_slab = slab(divflux, h)
        flux_slab = slab(flux, h)
        local_geometry_slab = slab(space.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        WJv¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        WJv² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute flux in contravariant coordinates (v¹,v²)
            # alternatively we could do this conversion _after_ taking the derivatives
            # may have an effect on the accuracy
            # materialize if lazy
            F = flux_slab[i, j]
            WJv¹[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.WJ *
                    Geometry.contravariant1(x, local_geometry),
                F,
            )
            WJv²[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.WJ *
                    Geometry.contravariant2(x, local_geometry),
                F,
            )
        end
        # GPU synchronize

        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            Dᵀ₁WJv¹ = RecursiveApply.rmatmul1(D', WJv¹, i, j) # D'WJv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
            # compute spectral deriv along second dimension
            Dᵀ₂WJv² = RecursiveApply.rmatmul2(D', WJv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
            divflux_slab[i, j] = Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv²
        end
    end
    return divflux
end

function slab_weak_divergence!(
    divflux_slab::DataLayouts.DataSlab2D,
    flux_slab::DataLayouts.DataSlab2D,
    space_slab::Spaces.SpectralElementSpaceSlab,
)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Spaces.undertype(space_slab)
    D = Quadratures.differentiation_matrix(FT, space_slab.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space_slab.quadrature_style)

    # for each element in the element stack
    local_geometry_slab = space_slab.local_geometry

    ST = eltype(divflux_slab)
    # Shared on GPU
    WJv¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
    WJv² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
    for i in 1:Nq, j in 1:Nq
        local_geometry = local_geometry_slab[i, j]
        # compute flux in contravariant coordinates (v¹,v²)
        # alternatively we could do this conversion _after_ taking the derivatives
        # may have an effect on the accuracy
        # materialize if lazy
        F = flux_slab[i, j]
        WJv¹[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.WJ * Geometry.contravariant1(x, local_geometry),
            F,
        )
        WJv²[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.WJ * Geometry.contravariant2(x, local_geometry),
            F,
        )
    end
    # GPU synchronize

    for i in 1:Nq, j in 1:Nq
        # compute spectral deriv along first dimension
        Dᵀ₁WJv¹ = RecursiveApply.rmatmul1(D', WJv¹, i, j) # D'WJv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
        # compute spectral deriv along second dimension
        Dᵀ₂WJv² = RecursiveApply.rmatmul2(D', WJv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
        divflux_slab[i, j] = Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv²
    end
    return divflux_slab
end

function slab_gradient!(∇field::Field, field::Field)
    @assert axes(∇field) === axes(field)
    Operators.slab_gradient!(
        Fields.field_values(∇field),
        Fields.field_values(field),
        axes(field),
    )
    return ∇field
end
function slab_divergence!(divflux::Field, flux::Field)
    @assert axes(divflux) === axes(flux)
    Operators.slab_divergence!(
        Fields.field_values(divflux),
        Fields.field_values(flux),
        axes(flux),
    )
    return divflux
end
function slab_weak_divergence!(divflux::Field, flux::Field)
    @assert axes(divflux) === axes(flux)
    Operators.slab_weak_divergence!(
        Fields.field_values(divflux),
        Fields.field_values(flux),
        axes(flux),
    )
    return divflux
end

function slab_gradient(field::Field)
    S = eltype(field)
    ∇S = RecursiveApply.rmaptype(T -> Cartesian12Vector{T}, S)
    Operators.slab_gradient!(similar(field, ∇S), field)
end

function slab_divergence(field::Field)
    S = eltype(field)
    divS = RecursiveApply.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_divergence!(similar(field, divS), field)
end
function slab_weak_divergence(field::Field)
    S = eltype(field)
    divS = RecursiveApply.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_weak_divergence!(similar(field, divS), field)
end

function interpolate(space_to::AbstractSpace, field_from::Field)
    field_to = similar(field_from, (space_to,), eltype(field_from))
    interpolate!(field_to, field_from)
end
function interpolate!(field_to::Field, field_from::Field)
    space_to = axes(field_to)
    space_from = axes(field_from)
    # @assert space_from.topology == space_to.topology

    M = Quadratures.interpolation_matrix(
        Float64,
        space_to.quadrature_style,
        space_from.quadrature_style,
    )
    Operators.tensor_product!(
        Fields.field_values(field_to),
        Fields.field_values(field_from),
        M,
    )
    return field_to
end

function restrict!(field_to::Field, field_from::Field)
    space_to = axes(field_to)
    space_from = axes(field_from)
    # @assert space_from.topology == space_to.topology

    M = Quadratures.interpolation_matrix(
        Float64,
        space_from.quadrature_style,
        space_to.quadrature_style,
    )
    Operators.tensor_product!(
        Fields.field_values(field_to),
        Fields.field_values(field_from),
        M',
    )
    return field_to
end

function matrix_interpolate(
    field::Field,
    Q_interp::Quadratures.Uniform{Nu},
) where {Nu}
    S = eltype(field)
    space = axes(field)
    mesh = space.topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2

    interp_data =
        DataLayouts.IH1JH2{S, Nu}(Matrix{S}(undef, (Nu * n1, Nu * n2)))

    M = Quadratures.interpolation_matrix(
        Float64,
        Q_interp,
        space.quadrature_style,
    )
    Operators.tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())
