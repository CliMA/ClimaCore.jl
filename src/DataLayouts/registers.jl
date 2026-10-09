"""
    RegisterArray{Sz, F, Stride}(array)

Array that presents the full size `Sz` of a [`DataScope`](@ref)'s data while
only storing the entries that belong to the thread reading it, in a
`StaticArrays.MArray` or `SArray` that a compiler can promote to registers. `F`
is the position of the field dimension (see [`f_dim`](@ref)), `Stride` is the
number of threads in the scope it was allocated for (also the stride between
consecutive points of the reading thread), and `register_array_params`
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
    data,
    array::RegisterArray,
    ::Type{T};
    params...,
) where {T} = layout_constructor(data, T; params...)(array)

# Map a linear index into the full array onto this thread's own storage:
# replace its point index p by (p - 1) ÷ Stride + 1, the point's position in
# this thread's strided subset; every divisor is a compile-time constant. The
# arithmetic is done in 32 bits when the array fits in them (which it always
# does for a register array): on GPUs, a 64-bit division by a constant that is
# not a power of two is lowered to several 64-bit multiply-high instructions,
# which cost several times more than the 32-bit ones, at every access.
@inline function register_index(
    ::RegisterArray{<:Any, <:Any, Sz, F, Stride},
    index::Int,
) where {Sz, F, Stride}
    (; Nf, SB, Np) = register_array_params(Sz, Val(F))
    I = prod(Sz) <= typemax(UInt32) ? UInt32 : Int
    (rest, before) = divrem((index - 1) % I, I(SB))
    (after, f) = divrem(rest, I(Nf))
    # When no thread holds more than one point, every point is at position 1
    # of its thread's subset, without dividing the point index (whose bound the
    # compiler does not always know, which would keep the index from being a
    # constant and the array from being promoted to registers).
    position = cld(Np, Stride) == 1 ? zero(I) : (after * I(SB) + before) ÷ I(Stride)
    return Int(f * I(cld(Np, Stride)) + position) + 1
end

@propagate_inbounds Base.getindex(array::RegisterArray, index::Int) =
    array.array[register_index(array, index)]
@propagate_inbounds Base.setindex!(array::RegisterArray, value, index::Int) =
    setindex!(array.array, value, register_index(array, index))

# The points of a layout are read and written through the Cartesian methods
# below (see has_cartesian_components), since a component divided out of a
# linear index is only a constant when the compiler can bound the index.
@inline has_cartesian_components(::RegisterArray) = true

# Cartesian access; two leading indices avoid overlap with the linear methods.
# The component is taken from the index along the F axis rather than divided
# out of a linear index, so that a read of a point's entries (whose component
# indices are compile-time constants; see struct_storage.jl) has a compile-time
# register index whether or not the compiler can bound the point's index. A
# run-time index keeps the array out of registers (it is spilled to memory).
@propagate_inbounds Base.getindex(
    array::RegisterArray, index1::Int, index2::Int, indices::Int...,
) = array.array[register_index(array, (index1, index2, indices...))]
@propagate_inbounds Base.setindex!(
    array::RegisterArray, value, index1::Int, index2::Int, indices::Int...,
) = setindex!(array.array, value, register_index(array, (index1, index2, indices...)))
# A CartesianIndex is splatted into the methods above (get_struct and set_struct!
# index entries with one), since Base converts it into a linear index for an
# IndexLinear array.
@propagate_inbounds Base.getindex(
    array::RegisterArray{<:Any, N},
    index::CartesianIndex{N},
) where {N} = array[Tuple(index)...]
@propagate_inbounds Base.setindex!(
    array::RegisterArray{<:Any, N},
    value,
    index::CartesianIndex{N},
) where {N} = setindex!(array, value, Tuple(index)...)
@inline function register_index(
    ::RegisterArray{<:Any, N, Sz, F, Stride},
    index::NTuple{N, Int},
) where {N, Sz, F, Stride}
    (; Np) = register_array_params(Sz, Val(F))
    f = isnothing(F) ? 1 : index[F]
    I = prod(Sz) <= typemax(UInt32) ? UInt32 : Int
    position =
        cld(Np, Stride) == 1 ? 0 :
        Int(
            (linear_index(drop_f_dim(Sz, Val(F)), drop_f_dim(index, Val(F))) - 1) % I ÷
            I(Stride),
        )
    return (f - 1) * cld(Np, Stride) + position + 1
end

# Single-point view of a RegisterArray: the Nf entries of the point along the F
# axis, read and written through the Cartesian methods above, whose component
# index is a compile-time constant at every read of a point's entries (see
# struct_storage.jl). The generic view (a SubArray) linearizes the index when
# it is built and reads the entries through the linear methods, which divide
# the component out of a run-time index that the compiler cannot bound, and a
# run-time index keeps the array out of registers. Device extensions also use
# this view for their read-only counterparts of a RegisterArray.
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
@inline function view_struct(
    array::RegisterArray{B, N},
    ::Type{T},
    index::CartesianIndex,
    ::Val{F},
) where {B, N, T, F}
    Nf = num_basetypes(B, T)
    @boundscheck checkbounds(array, struct_indices(array, Val(Nf), index, Val(F))...)
    return RegisterPointView{B, Nf, N, F, typeof(array)}(
        array,
        add_f_dim(Tuple(index), 1, Val(F)),
    )
end

# Freeze a mutable register array into an immutable one, so that constant_field
# can guarantee stack (rather than heap) storage without full inlining.
@inline StaticArrays.SArray(
    array::RegisterArray{T, N, Sz, F, Stride},
) where {T, N, Sz, F, Stride} =
    RegisterArray{Sz, F, Stride}(StaticArrays.SArray(array.array))

"""
    stored_in_scope_registers(data)

Whether `data` is a [`DataLayout`](@ref) whose values are in the registers that
[`register_similar`](@ref) allocates for its [`DataScope`](@ref), so that each
thread of the scope holds the values of its own points.
"""
@inline stored_in_scope_registers(data) =
    data isa DataLayout && stored_in_registers(data) &&
    register_stride(parent(data)) == static_num_threads(DataScope(data))
@inline register_stride(
    ::RegisterArray{<:Any, <:Any, <:Any, <:Any, Stride},
) where {Stride} = Stride

"""
    requires_lockstep(data)

Whether reading a value of `data` requires every thread in its
[`DataScope`](@ref) to read the same value at the same time, as when the value
is in the registers of another thread on a device that can read them. Such
values can be read in the rounds of [`update_points!`](@ref), but not by reads
that branch on the point being evaluated. This is determined by the type of the
array that stores `data`, and it is `false` unless a device extension makes it
`true` for that type.
"""
@inline requires_lockstep(data::DataLayout) = requires_lockstep(parent_type(data))
@inline requires_lockstep(::Type) = false

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
not be read by any other thread (see [`RegisterArray`](@ref)). This falls back to
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

# Whether register_similar can keep values with the shape and scope of data in
# registers, which needs a compile-time thread count other than one (a single
# thread's buffers are already private to it) and an inferrable size.
@inline function has_scope_registers(data)
    Stride = static_num_threads(DataScope(data))
    return !isnothing(Stride) && Stride != 1 && has_inferred_size(data)
end

@inline function register_similar(
    data, ::Type{T}, ::Type{B}, array_size, ::Val{F}, ::Val{Stride},
) where {T, B, F, Stride}
    (; Nf, Np) = register_array_params(array_size, Val(F))
    storage = StaticArrays.MArray{Tuple{cld(Np, Stride) * Nf}, B}(undef)
    # The register array has the layout's canonical size by construction, so
    # the parent check in the constructor is skipped.
    return @inbounds rebuild(data, RegisterArray{array_size, F, Stride}(storage), T)
end
