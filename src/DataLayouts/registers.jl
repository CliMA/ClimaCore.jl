"""
    RegisterArray{Sz, F, Stride}(array)

Array that presents the full size `Sz` of a [`DataScope`](@ref)'s data while
only storing the entries that belong to the thread reading it, in a
`StaticArrays.MArray` or `SArray` that a compiler can promote to registers (or in
a view of some of its entries, for a field view of the data). `F`
is the position of the field dimension (see [`f_dim`](@ref)), `Stride` is the
number of threads in the scope it was allocated for (also the stride between
consecutive points of the reading thread), or 1 for a single point that every thread
stores (see [`register_similar`](@ref)), and `register_array_params`
derives the remaining size parameters from `Sz` and `F`.

Every thread is assigned the strided subset `rank:Stride:Np` of the scope's
points (see [`subscope_indices`](@ref)), so the stored array has the
compile-time size `cld(Np, Stride) * Nf`. When a thread holds a single point
(the usual case; see [`slice_subscope`](@ref)), a component's index does not
depend on the point, so every component can live in a register.

Reading or writing a point that belongs to a *different* thread silently
accesses the reading thread's own point at the same subset position, so a
`RegisterArray` may only hold data that no other thread reads; values that
cross threads must be published through [`scoped_static_array`](@ref) buffers.
Unlike shared memory globals, identified by byte size alone (see the equal
sizes invariant in `ext/cuda/scopes.jl`), register arrays are distinct values,
which keeps two live operator destinations from aliasing.
"""
struct RegisterArray{
    T, N, Sz, F, Stride, A <: StaticArrays.StaticArray{<:Any, T, 1},
} <: AbstractArray{T, N}
    array::A
end

@inline function register_array_params(array_size, ::Val{F}) where {F}
    Nf = isnothing(F) ? 1 : array_size[F]
    SB = isnothing(F) ? prod(array_size) : prod(array_size[1:(F - 1)])
    return (; Nf, SB, Np = prod(array_size) ÷ Nf)
end

@inline RegisterArray{Sz, F, Stride}(
    array::A,
) where {Sz, F, Stride, A <: StaticArrays.StaticArray{<:Any, <:Any, 1}} =
    RegisterArray{eltype(A), length(Sz), Sz, F, Stride, A}(array)

@inline Base.size(::RegisterArray{<:Any, <:Any, Sz}) where {Sz} = Sz
@inline Base.IndexStyle(::Type{<:RegisterArray}) = IndexLinear()

# A RegisterArray presents exactly the data of the layout it was allocated for,
# so rebuilding a layout around one keeps the layout's own scope.
Base.@propagate_inbounds rebuild(
    data, array::RegisterArray, ::Type{T}; params...,
) where {T} = layout_constructor(data, T; params...)(array)

# A field's components are a contiguous block of the storage (see register_index), so its
# view is a RegisterArray over that block (writable when the storage is an MArray), which
# keeps the layout's scope; a SubArray would hide that the data is in registers.
@inline function struct_field_view(
    array::RegisterArray{<:Any, <:Any, Sz, F, Stride}, ::Type{T}, ::Val{i}, ::Val{F},
) where {Sz, F, Stride, T, i}
    check_basetype(eltype(array), fieldtype(T, i))
    isnothing(F) && return array
    Nf = num_basetypes(eltype(array), fieldtype(T, i))
    first_f = Int(stable_fieldoffset(T, Val(i))) ÷ sizeof(eltype(array))
    Nl = cld(register_array_params(Sz, Val(F)).Np, Stride)
    storage =
        view(array.array, StaticArrays.SUnitRange(first_f * Nl + 1, (first_f + Nf) * Nl))
    return RegisterArray{Base.setindex(Sz, Nf, F), F, Stride}(storage)
end

# Map a linear index into the full array onto this thread's own storage:
# replace its point index p by (p - 1) ÷ Stride + 1, the point's position in
# this thread's strided subset; every divisor is a compile-time constant.
# The arithmetic is in 32 bits when the array fits, as 64-bit divisions are costly on GPUs.
@inline function register_index(
    ::RegisterArray{<:Any, <:Any, Sz, F, Stride},
    index::Int,
) where {Sz, F, Stride}
    (; Nf, SB, Np) = register_array_params(Sz, Val(F))
    I = prod(Sz) <= typemax(UInt32) ? UInt32 : Int
    (rest, before) = divrem((index - 1) % I, I(SB))
    (after, f) = divrem(rest, I(Nf))
    # With at most one point per thread, skip a division that may spill the array.
    position = cld(Np, Stride) == 1 ? zero(I) : (after * I(SB) + before) ÷ I(Stride)
    return Int(f * I(cld(Np, Stride)) + position) + 1
end

@propagate_inbounds Base.getindex(array::RegisterArray, index::Int) =
    array.array[register_index(array, index)]
@propagate_inbounds Base.setindex!(array::RegisterArray, value, index::Int) =
    setindex!(array.array, value, register_index(array, index))

@inline has_cartesian_components(::RegisterArray) = true

# Cartesian access; two leading indices avoid overlap with the linear methods.
# Taking the component from the F axis keeps register indices constant (no spilling).
@propagate_inbounds Base.getindex(
    array::RegisterArray, index1::Int, index2::Int, indices::Int...,
) = array.array[register_index(array, (index1, index2, indices...))]
@propagate_inbounds Base.setindex!(
    array::RegisterArray, value, index1::Int, index2::Int, indices::Int...,
) = setindex!(array.array, value, register_index(array, (index1, index2, indices...)))
# Splat CartesianIndex entries (of get_struct and set_struct!), which Base would linearize.
@propagate_inbounds Base.getindex(
    array::RegisterArray{<:Any, N}, index::CartesianIndex{N},
) where {N} = array[Tuple(index)...]
@propagate_inbounds Base.setindex!(
    array::RegisterArray{<:Any, N}, value, index::CartesianIndex{N},
) where {N} = setindex!(array, value, Tuple(index)...)
@inline function register_index(
    ::RegisterArray{<:Any, N, Sz, F, Stride}, index::NTuple{N, Int},
) where {N, Sz, F, Stride}
    (; Np) = register_array_params(Sz, Val(F))
    (point, f) = point_and_component(Sz, Val(F), index)
    I = prod(Sz) <= typemax(UInt32) ? UInt32 : Int
    position = cld(Np, Stride) == 1 ? 0 : Int((point - 1) % I ÷ I(Stride))
    return (f - 1) * cld(Np, Stride) + position + 1
end
@inline point_and_component(Sz, ::Val{F}, index) where {F} = (
    linear_index(drop_f_dim(Sz, Val(F)), drop_f_dim(index, Val(F))),
    isnothing(F) ? 1 : index[F],
)

# Single-point view of a RegisterArray (also used by device extensions) that reads through
# the Cartesian methods above: a fast SubArray would divide out components at run time, and
# Cartesian SubArrays give the same kernels but compile more slowly on CPUs (Julia 1.10
# keeps their bounds checks, dead under @inbounds, in the IR).
struct RegisterPointView{T, Nf, N, F, A <: AbstractArray{T, N}} <: AbstractVector{T}
    parent::A
    index::NTuple{N, Int} # the point's index, with the F axis (if any) at 1
end
Base.parent(view::RegisterPointView) = view.parent
Base.size(::RegisterPointView{<:Any, Nf}) where {Nf} = (Nf,)
Base.IndexStyle(::Type{<:RegisterPointView}) = IndexLinear()
@inline component_index(view::RegisterPointView{<:Any, <:Any, <:Any, F}, i) where {F} =
    isnothing(F) ? view.index : Base.setindex(view.index, i, F)
Base.@propagate_inbounds function Base.getindex(view::RegisterPointView, i::Int)
    @boundscheck checkbounds(view, i)
    return @inbounds parent(view)[component_index(view, i)...]
end
Base.@propagate_inbounds function Base.setindex!(view::RegisterPointView, value, i::Int)
    @boundscheck checkbounds(view, i)
    @inbounds parent(view)[component_index(view, i)...] = value
    return view
end
@inline RegisterPointView(
    array::AbstractArray{T, N}, ::Val{Nf}, index, ::Val{F},
) where {T, N, Nf, F} =
    RegisterPointView{T, Nf, N, F, typeof(array)}(array, add_f_dim(Tuple(index), 1, Val(F)))
@inline unchecked_view_struct(a::RegisterArray, Nf::Val, i::CartesianIndex, F::Val) =
    RegisterPointView(a, Nf, i, F)

# Freeze a mutable register array into an immutable one, so that constant_field
# can guarantee stack (rather than heap) storage without full inlining.
@inline StaticArrays.SArray(
    array::RegisterArray{T, N, Sz, F, Stride},
) where {T, N, Sz, F, Stride} =
    RegisterArray{Sz, F, Stride}(StaticArrays.SArray(array.array))
# Point views (like boundary levels of columns) are frozen with their parents.
@inline function StaticArrays.SArray(
    view::RegisterPointView{T, Nf, N, F},
) where {T, Nf, N, F}
    array = StaticArrays.SArray(parent(view))
    return RegisterPointView{T, Nf, N, F, typeof(array)}(array, view.index)
end
Base.@propagate_inbounds rebuild(
    data, view::RegisterPointView, ::Type{T}; params...,
) where {T} = layout_constructor(data, T; params...)(view)

"""
    stored_in_scope_registers(data)

Whether `data` is a [`DataLayout`](@ref) in the registers that [`register_similar`](@ref)
allocates for its [`DataScope`](@ref), with each thread holding its own points' values.
Every thread also holds a copy of a single point, but only the copy of the thread that
owns the point (see `owns_point`) is certain to be written.
"""
@inline stored_in_scope_registers(data) =
    data isa DataLayout && stored_in_registers(data) &&
    unrolled_in(register_stride(parent(data)), (1, static_num_threads(DataScope(data))))
@inline register_stride(
    ::RegisterArray{<:Any, <:Any, <:Any, <:Any, Stride},
) where {Stride} = Stride

# Whether any value in a DataLayout or LazyDataLayout expression is backed by a
# RegisterArray, and therefore cannot be read by any thread other than the one
# that wrote it.
@inline stored_in_registers(data::DataLayout) = parent_type(data) <: RegisterArray
@inline stored_in_registers(bc::LazyDataLayout) =
    unrolled_any(stored_in_registers, layout_args(bc))

"""
    register_similar(data, T)
    register_similar(bc, T)

Allocate a [`DataLayout`](@ref) like `Base.similar(data, T)`, but backed by a
[`RegisterArray`](@ref) that only stores each thread's own points; the result must
not be read by any other thread (see [`RegisterArray`](@ref)). A single point (like a
value derived from a level of a column) is stored by every thread instead, with a stride
of 1, and every thread's copy is written, either by every thread or from the copy of its
owner (see `share_single_point!`), so that every thread can read it. This falls back to
[`buffer_similar`](@ref) when registers cannot be used (a non-inferrable size, a
non-constant thread count, or a single-thread scope, which is already an `MArray`);
lazy layouts are first converted into the layout type that [`buffer_similar`](@ref)
would allocate.
"""
@inline register_similar(bc::LazyDataLayout, ::Type{T}) where {T} =
    has_inferred_size(bc) ?
    register_similar(
        layout_type(bc){T, shape_params(bc)..., typeof(DataScope(bc)), parent_type(bc)},
        T,
    ) : buffer_similar(bc, T)

@inline function register_similar(data, ::Type{T}) where {T}
    B = checked_valid_basetype(eltype(parent_type(data)), T)
    has_scope_registers(data) || return similar_layout(data, T)
    Stride = static_num_threads(DataScope(data))
    Nf = num_basetypes(B, T)
    array_size = add_f_dim(inferred_size(data), Nf, Val(f_dim(data)))
    return register_similar(data, T, B, array_size, Val(f_dim(data)), Val(Stride))
end

@inline function has_scope_registers(data)
    Stride = static_num_threads(DataScope(data))
    return !isnothing(Stride) && Stride != 1 && has_inferred_size(data)
end

@inline function register_similar(
    data, ::Type{T}, ::Type{B}, array_size, ::Val{F}, ::Val{Stride},
) where {T, B, F, Stride}
    (; Nf, Np) = register_array_params(array_size, Val(F))
    storage = StaticArrays.MArray{Tuple{cld(Np, Stride) * Nf}, B}(undef)
    # The register array has the layout's canonical size and a valid base type.
    array = RegisterArray{array_size, F, isone(Np) ? 1 : Stride}(storage)
    return unchecked_rebuild(data, array, T)
end
