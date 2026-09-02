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
@inline rebuild(data, array::RegisterArray, ::Type{T}; params...) where {T} =
    layout_constructor(data, T; params...)(array)

# Map a linear index into the full array onto this thread's own storage:
# replace its point index p by (p - 1) ÷ Stride + 1, the point's position in
# this thread's strided subset; every divisor is a compile-time constant.
@inline function register_index(
    ::RegisterArray{<:Any, <:Any, Sz, F, Stride},
    index::Int,
) where {Sz, F, Stride}
    (; Nf, SB, Np) = register_array_params(Sz, Val(F))
    (rest, before) = divrem(index - 1, SB)
    (after, f) = divrem(rest, Nf)
    point = after * SB + before
    return f * cld(Np, Stride) + point ÷ Stride + 1
end

@propagate_inbounds Base.getindex(array::RegisterArray, index::Int) =
    array.array[register_index(array, index)]
@propagate_inbounds Base.setindex!(array::RegisterArray, value, index::Int) =
    setindex!(array.array, value, register_index(array, index))

# Cartesian access; two leading indices avoid overlap with the linear methods.
@propagate_inbounds Base.getindex(
    array::RegisterArray, index1::Int, index2::Int, indices::Int...,
) = array[linear_index(size(array), (index1, index2, indices...))]
@propagate_inbounds Base.setindex!(
    array::RegisterArray, value, index1::Int, index2::Int, indices::Int...,
) = setindex!(array, value, linear_index(size(array), (index1, index2, indices...)))

# Freeze a mutable register array into an immutable one, so that constant_field
# can guarantee stack (rather than heap) storage without full inlining.
@inline StaticArrays.SArray(
    array::RegisterArray{T, N, Sz, F, Stride},
) where {T, N, Sz, F, Stride} =
    RegisterArray{Sz, F, Stride}(StaticArrays.SArray(array.array))

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
    Stride = static_num_threads(DataScope(data))
    (isnothing(Stride) || Stride == 1 || !has_inferred_size(data)) &&
        return similar_layout(data, T)
    Nf = num_basetypes(B, T)
    array_size = add_f_dim(inferred_size(data), Nf, Val(f_dim(data)))
    return register_similar(data, T, B, array_size, Val(f_dim(data)), Val(Stride))
end

@inline function register_similar(
    data, ::Type{T}, ::Type{B}, array_size, ::Val{F}, ::Val{Stride},
) where {T, B, F, Stride}
    (; Nf, Np) = register_array_params(array_size, Val(F))
    storage = StaticArrays.MArray{Tuple{cld(Np, Stride) * Nf}, B}(undef)
    return rebuild(data, RegisterArray{array_size, F, Stride}(storage), T)
end
