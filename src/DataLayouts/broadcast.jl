"""
    DataStyle(D)

`BroadcastStyle` for a [`DataLayout`](@ref) of type `D`, which stores the
[`layout_type`](@ref) and its corresponding value of `ndims` as type parameters.
"""
struct DataStyle{N, D <: DataLayout{<:Any, N}} <: Broadcast.AbstractArrayStyle{N} end
DataStyle(::Type{D}) where {D} = DataStyle{ndims(D), layout_type(D)}()

Base.ndims(::DataStyle{N}) where {N} = N

Broadcast.BroadcastStyle(::Type{D}) where {D <: DataLayout} = DataStyle(D)

# For styles with equal typenames but different dimensionalities, Base's
# fallback for AbstractArrayStyle calls typeof(style)(Val(N)). DataStyle needs a
# layout type D in addition to the dimensionality, so it bypasses the fallback.
Broadcast.BroadcastStyle(style1::DataStyle, style2::DataStyle) =
    style1 == style2 || iszero(ndims(style2)) ? style1 :
    iszero(ndims(style1)) ? style2 : Broadcast.Unknown()

# Pass scalar values in Tuples of length 1 or in 0-dimensional AbstractArrays.
# Add DefaultArrayStyle{0} and DataStyle{0} methods to avoid ambiguities.
Broadcast.BroadcastStyle(style::DataStyle, ::Broadcast.Style{Tuple}) = style
Broadcast.BroadcastStyle(style::DataStyle, ::Broadcast.AbstractArrayStyle{0}) = style
Broadcast.BroadcastStyle(style::DataStyle, ::Broadcast.DefaultArrayStyle{0}) = style
Broadcast.BroadcastStyle(style::DataStyle, ::DataStyle{0}) = style

# Enable automatic nested broadcasting over supported types of iterators.
@inline Broadcast.broadcastable(data::DataLayout) =
    reinterpret(add_auto_broadcasters(eltype(data)), data)
@inline Broadcast.broadcasted(style::DataStyle, f::F, args...) where {F} =
    auto_broadcasted(style, f, args)

"""
    LazyDataLayout{D}

A [`DataStyle`](@ref) broadcast expression whose [`layout_type`](@ref) is `D`.
"""
const LazyDataLayout{D} = Broadcast.Broadcasted{<:DataStyle{<:Any, D}}

# Optimize axes(::LazyDataLayout) with statically inferrable axes when possible.
@inline Broadcast._axes(bc::LazyDataLayout, ::Nothing) =
    has_inferred_size(bc) ? unrolled_map(Base.OneTo, inferred_size(bc)) :
    unrolled_reduce(stable_combine_axes, unrolled_map(axes, bc.args))

# Instead of Base's combine_axes, whose DimensionMismatch error cannot compile
# on GPUs, use a version that always selects the first non-singleton dimension.
@inline stable_combine_axes(axes1::Tuple, axes2::Tuple) =
    isempty(axes2) ? axes1 :
    isempty(axes1) ? axes2 :
    (
        isone(length(first(axes2))) ? first(axes1) : first(axes2),
        stable_combine_axes(Base.tail(axes1), Base.tail(axes2))...,
    )

# Make ndims support nested broadcasts whose axes have not been instantiated.
@inline Base.ndims(::LazyDataLayout{D}) where {D} = ndims(D)

# Allow eltype to return non-concrete types, like an empty Union{}.
@inline Base.eltype(bc::LazyDataLayout) = unsafe_eltype(bc)

# Remove all AutoBroadcaster wrappers when allocating a new DataLayout.
@inline Base.similar(bc::LazyDataLayout) =
    similar(bc, drop_auto_broadcasters(safe_eltype(bc)))
@inline Base.similar(bc::LazyDataLayout, ::Type{T}) where {T} = similar(
    layout_type(bc){T, shape_params(bc)..., typeof(DataScope(bc)), parent_type(bc)},
    size(bc),
)

# Define a MultiBroadcastFusion type, FusedMultiBroadcast, and a corresponding
# @fused macro, as outlined in https://github.com/CliMA/MultiBroadcastFusion.jl.
@make_type FusedMultiBroadcast
@make_fused fused_direct FusedMultiBroadcast fused_direct

# Adapt does not descend into Base.Pair, so Adapt.@adapt_structure would leave
# each pair's destination and broadcast unconverted (e.g. as CuArrays instead
# of CuDeviceArrays in kernel arguments).
Adapt.adapt_structure(to, fmb::FusedMultiBroadcast) = FusedMultiBroadcast(
    unrolled_map(fmb.pairs) do pair
        Pair(Adapt.adapt(to, pair.first), Adapt.adapt(to, pair.second))
    end,
)

const MaybeLazyDataLayout = Union{DataLayout, LazyDataLayout}
const MaybeFusedDataLayoutBroadcast = Union{LazyDataLayout, FusedMultiBroadcast}

"""
    layout_args(bc)

Extracts every [`DataLayout`](@ref) and [`LazyDataLayout`](@ref) from the
arguments of a broadcast expression.
"""
@inline layout_args(bc::LazyDataLayout) =
    unrolled_filter(Base.Fix2(isa, MaybeLazyDataLayout), bc.args)
@inline layout_args(bc::FusedMultiBroadcast) =
    unrolled_filter(Base.Fix2(isa, MaybeLazyDataLayout), unrolled_flatten(bc.pairs))

@inline DataScope(bc::MaybeFusedDataLayoutBroadcast) = DataScope(layout_args(bc)...)

@inline layout_type(::LazyDataLayout{D}) where {D} = D

# Only specify the parent array element type, instead of a concrete array type.
@inline parent_type(bc::LazyDataLayout) =
    AbstractArray{promote_type(unrolled_map(eltype ∘ parent_type, layout_args(bc))...)}

# Allow any combination of f_dim values, taking a maximum to resolve conflicts.
@inline function f_dim(bc::LazyDataLayout)
    f_dims = unrolled_filter(!isnothing, unrolled_map(f_dim, layout_args(bc)))
    return isempty(f_dims) ? nothing : max(f_dims...)
end

# Extrude singleton axes like Broadcast.combine_axes when combining vijh_params.
@inline vijh_params(bc::LazyDataLayout) =
    unrolled_reduce(unrolled_map(vijh_params, layout_args(bc))) do params1, params2
        unrolled_map(params1, params2) do N1, N2
            isnothing(N1) || isnothing(N2) ? nothing :
            N1 == N2 || isone(N2) ? N1 :
            isone(N1) ? N2 : Broadcast.throwdm((Base.OneTo(N1),), (Base.OneTo(N2),))
        end
    end

# Compute layout-specific shape_params from the generic vijh_params and f_dim.
@inline shape_params(::LazyDataLayout{DataF}) = (;)
@inline shape_params(bc::LazyDataLayout{VIJHWithF}) =
    (; vijh_params(bc)..., F = f_dim(bc))
@inline shape_params(bc::LazyDataLayout{VIH1}) =
    (; vijh_params(bc).Nv, vijh_params(bc).Ni, vijh_params(bc).Nh)
@inline shape_params(bc::LazyDataLayout{IH1JH2}) =
    (; vijh_params(bc).Ni, vijh_params(bc).Nj, vijh_params(bc).Nh)

@inline inferred_size(bc::LazyDataLayout) =
    inferred_size(layout_type(bc){<:Any, shape_params(bc)...})

@inline function nelems(bc::LazyDataLayout)
    (; Nv, Ni, Nj, Nh) = vijh_params(bc)
    return isnothing(Nh) ? length(bc) ÷ (Nv * Ni * Nj) : Nh
end

# Forward size queries and primitives to the first layout in a fused broadcast.
const DATA_LAYOUT_PRIMITIVES =
    (:layout_type, :parent_type, :f_dim, :shape_params, :inferred_size, :nelems)
for f in (:ndims, :length, :size, :axes, DATA_LAYOUT_PRIMITIVES...)
    f_with_module_prefix = f in DATA_LAYOUT_PRIMITIVES ? f : :(Base.$f)
    @eval @inline $f_with_module_prefix(bc::FusedMultiBroadcast) =
        unrolled_allequal($f, layout_args(bc)) ? $f(first(layout_args(bc))) :
        throw(DimensionMismatch($("$f is inconsistent among fused broadcasts")))
end

"""
    modify_args(f, bc, f_args...)

Replaces each of the [`layout_args`](@ref) in a broadcast expression with
`f(layout_arg, f_args...)`.
"""
@propagate_inbounds function modify_args(f::F, bc::LazyDataLayout, f_args...) where {F}
    modified_args = unrolled_map_with_inbounds(bc.args) do arg
        Base.@_propagate_inbounds_meta
        arg isa MaybeLazyDataLayout ? f(arg, f_args...) : arg
    end
    return Broadcast.Broadcasted(bc.style, bc.f, modified_args, bc.axes)
end
@propagate_inbounds function modify_args(f::F, bc::FusedMultiBroadcast, f_args...) where {F}
    modified_pairs = unrolled_map_with_inbounds(bc.pairs) do (dest, bc)
        Base.@_propagate_inbounds_meta
        Pair(f(dest, f_args...), bc isa MaybeLazyDataLayout ? f(bc, f_args...) : bc)
    end
    return FusedMultiBroadcast(modified_pairs)
end

@inline reassign(bc::MaybeFusedDataLayoutBroadcast, scope) =
    modify_args(reassign, bc, scope)
@propagate_inbounds level_view(bc::MaybeFusedDataLayoutBroadcast, v) =
    modify_args(level, bc, v)
@propagate_inbounds slab_view(bc::MaybeFusedDataLayoutBroadcast, v, h) =
    modify_args(slab, bc, v, h)
@propagate_inbounds column_view(bc::MaybeFusedDataLayoutBroadcast, i, j, h) =
    modify_args(column, bc, i, j, h)
