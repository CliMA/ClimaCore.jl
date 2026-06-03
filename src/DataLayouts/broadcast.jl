"""
    DataStyle(D)

`BroadcastStyle` for a [`DataLayout`](@ref) of type `D`, which stores the
[`layout_type`](@ref) and its corresponding value of `ndims` as type parameters.
"""
struct DataStyle{N, D <: DataLayout{<:Any, N}} <: Broadcast.AbstractArrayStyle{N} end
DataStyle(::Type{D}) where {D} = DataStyle{ndims(D), layout_type(D)}()

Broadcast.BroadcastStyle(::Type{D}) where {D <: DataLayout} = DataStyle(D)

# Pass scalar values by wrapping them in 0-dimensional AbstractArrays or Tuples.
# Add a DefaultArrayStyle{0} method to avoid ambiguity with a built-in method.
Broadcast.BroadcastStyle(style::DataStyle, ::Broadcast.DefaultArrayStyle{0}) = style
Broadcast.BroadcastStyle(style::DataStyle, ::Broadcast.AbstractArrayStyle{0}) = style
Broadcast.BroadcastStyle(style::DataStyle, ::Broadcast.Style{Tuple}) = style

# Enable automatic nested broadcasting over supported types of iterators.
@inline Broadcast.broadcastable(data::DataLayout) =
    reinterpret(add_auto_broadcasters(eltype(data)), data)
@inline Broadcast.broadcasted(style::DataStyle, f::F, args...) where {F} =
    auto_broadcasted(style, f, args)

# Allow getindex(::LazyDataLayout, _) to avoid Cartesian indices when possible.
@inline Broadcast.newindex(::DataLayout, index::Integer) = index

"""
    LazyDataLayout{D}

A [`DataStyle`](@ref) broadcast expression whose [`layout_type`](@ref) is `D`.
"""
const LazyDataLayout{D} = Broadcast.Broadcasted{<:DataStyle{<:Any, D}}

# Optimize axes(::LazyDataLayout) to use statically inferrable size information.
@inline Broadcast._axes(bc::LazyDataLayout, ::Nothing) =
    has_inferred_size(bc) ? unrolled_map(Base.OneTo, inferred_size(bc)) :
    Broadcast.combine_axes(bc.args...)

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
Adapt.@adapt_structure FusedMultiBroadcast

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
            ismissing(N1) || ismissing(N2) ? missing :
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
    return ismissing(Nh) ? length(bc) ÷ (Nv * Ni * Nj) : Nh
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

Modifies a broadcast expression by replacing each of its [`layout_args`](@ref)
with `f(layout_arg, f_args...)`, optionally passing additional `f_args` to `f`.
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
    modify_args(arg -> (Base.@_propagate_inbounds_meta; reassign(arg, scope)), bc)
@propagate_inbounds level_view(bc::MaybeFusedDataLayoutBroadcast, v) =
    modify_args(arg -> (Base.@_propagate_inbounds_meta; level(arg, v)), bc)
@propagate_inbounds slab_view(bc::MaybeFusedDataLayoutBroadcast, v, h) =
    modify_args(arg -> (Base.@_propagate_inbounds_meta; slab(arg, v, h)), bc)
@propagate_inbounds column_view(bc::MaybeFusedDataLayoutBroadcast, i, j, h) =
    modify_args(arg -> (Base.@_propagate_inbounds_meta; column(arg, i, j, h)), bc)

# Use Broadcast.newindex to match the behavior of getindex for LazyDataLayouts.
@propagate_inbounds Base.view(bc::MaybeFusedDataLayoutBroadcast, index) =
    modify_args(bc) do arg
        Base.@_propagate_inbounds_meta
        view(arg, Broadcast.newindex(arg, index))
    end
