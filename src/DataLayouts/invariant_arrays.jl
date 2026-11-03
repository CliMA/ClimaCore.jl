"""
    InvariantArray(array)

A dense `array` read with loads that carry `!invariant.load` metadata. LLVM merges
invariant loads of the same element and moves them across stores and barriers, so code
that reads an element several times (e.g., once for every operator in a loop over slices)
only loads it once. Since its loads can be moved before writes, an `InvariantArray` cannot
be modified: `setindex!` throws an `ArgumentError`. `similar` and `copy` return arrays of
the wrapped type, so copies of fields stored in `InvariantArray`s (e.g.,
`copy(Fields.local_geometry_field(space))`) and broadcasts over them can be modified, while
deep copies keep the wrapper. Copies, comparisons, reductions, and broadcasts of an
`InvariantArray` or a view of one use the methods of the wrapped array (e.g., the `CuArray`
methods for GPU memory on the host).

Grids store their local geometry in `InvariantArray`s (see [`invariant_data`](@ref)),
which `Adapt` converts to `InvariantArray`s of device arrays in GPU kernels.
"""
struct InvariantArray{T, N, A <: DenseArray{T, N}} <: DenseArray{T, N}
    array::A
end

Base.size(array::InvariantArray) = size(array.array)
Base.IndexStyle(::Type{<:InvariantArray}) = IndexLinear()
Base.pointer(array::InvariantArray, i::Integer = 1) = pointer(array.array, i)
Base.elsize(::Type{<:InvariantArray{T, N, A}}) where {T, N, A} = Base.elsize(A)
Base.unsafe_convert(::Type{Ptr{T}}, array::InvariantArray{T}) where {T} =
    Base.unsafe_convert(Ptr{T}, array.array)
Base.similar(array::InvariantArray, ::Type{T}, dims::Dims) where {T} =
    similar(array.array, T, dims)

Base.@propagate_inbounds function Base.getindex(array::InvariantArray, i::Integer)
    @boundscheck checkbounds(array, i)
    return invariant_getindex(array.array, pointer(array.array), i % Int)
end

# CPU and device memory is read with invariant loads. Other memory, e.g., a GPU array
# read on the host (where its pointer is a CuPtr), is read by the wrapped array.
@inline invariant_getindex(array, ptr::Union{Ptr, Core.LLVMPtr}, i) =
    GC.@preserve array invariant_load(llvm_pointer(ptr), i - 1)
@inline invariant_getindex(array, _, i) = @inbounds array[i]

Base.setindex!(::InvariantArray, _, _...) =
    throw(ArgumentError("Cannot modify InvariantArray"))

Adapt.adapt_structure(to, array::InvariantArray) =
    InvariantArray(Adapt.adapt(to, array.array))

# An adaptor that wraps every dense array that is not an InvariantArray in an InvariantArray
struct WrapInvariantArrays end
Adapt.adapt_storage(::WrapInvariantArrays, array::DenseArray) = InvariantArray(array)

Adapt.adapt_structure(::WrapInvariantArrays, array::InvariantArray) = array

# An adaptor that replaces every InvariantArray (e.g., the parent of a view) with the
# array it wraps
struct UnwrapInvariantArrays end
Adapt.adapt_structure(::UnwrapInvariantArrays, array::InvariantArray) = array.array

unwrap_invariant(array) = Adapt.adapt(UnwrapInvariantArrays(), array)

# Views of InvariantArrays with the index types of DataLayouts views (whole dimensions,
# ranges, integers), so that the methods below do not overlap with those for views with
# other index types (e.g., the `BlockSlice`s of BlockArrays).
const InvariantView{T, N} = SubArray{
    T,
    N,
    <:InvariantArray,
    <:Tuple{
        Vararg{Union{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}, Base.OneTo{Int}, Int}},
    },
}

# InvariantArrays and the views, reshapes, and reinterprets of them that grids and fields
# create. Invariant loads only help loops and kernels, which read an element repeatedly, so
# copies, comparisons, reductions, and broadcasts use the methods of the wrapped arrays:
# memory copies for CPU arrays, and, for GPU memory on the host, the methods of the GPU
# array type, which launch kernels where the host methods would index elements one at a
# time.
const AnyInvariantArray{T, N} = Union{
    InvariantArray{T, N},
    InvariantView{T, N},
    Base.ReshapedArray{T, N, <:Union{InvariantArray, InvariantView}},
    Base.ReinterpretArray{T, N, <:Any, <:Union{InvariantArray, InvariantView}},
}
Base.copy(array::AnyInvariantArray) = copy(unwrap_invariant(array))
Base.Array(array::AnyInvariantArray) = Array(unwrap_invariant(array))
Base.collect(array::AnyInvariantArray) = collect(unwrap_invariant(array))
Base.copyto!(dest::StridedArray, src::AnyInvariantArray) =
    copyto!(dest, unwrap_invariant(src))
Base.mapreduce(f, op, array::AnyInvariantArray; kwargs...) =
    mapreduce(f, op, unwrap_invariant(array); kwargs...)

# Dense arrays (including InvariantArrays) and the views, reshapes, and reinterprets of them,
# which are compared with InvariantArrays through the methods of the wrapped arrays
const AnyDenseArray{T, N} = Union{
    DenseArray{T, N},
    SubArray{T, N, <:DenseArray},
    Base.ReshapedArray{T, N, <:Union{DenseArray, SubArray{<:Any, <:Any, <:DenseArray}}},
    Base.ReinterpretArray{
        T,
        N,
        <:Any,
        <:Union{DenseArray, SubArray{<:Any, <:Any, <:DenseArray}},
    },
}

Base.:(==)(array1::AnyInvariantArray, array2::AnyInvariantArray) =
    unwrap_invariant(array1) == unwrap_invariant(array2)
Base.:(==)(array1::AnyInvariantArray, array2::AnyDenseArray) =
    unwrap_invariant(array1) == array2
Base.:(==)(array1::AnyDenseArray, array2::AnyInvariantArray) =
    array1 == unwrap_invariant(array2)

Base.isequal(array1::AnyInvariantArray, array2::AnyInvariantArray) =
    isequal(unwrap_invariant(array1), unwrap_invariant(array2))
Base.isequal(array1::AnyInvariantArray, array2::AnyDenseArray) =
    isequal(unwrap_invariant(array1), array2)
Base.isequal(array1::AnyDenseArray, array2::AnyInvariantArray) =
    isequal(array1, unwrap_invariant(array2))

# Reductions that Base computes by iterating over elements (norm is also used by isapprox)
Base.any(f::Function, array::AnyInvariantArray; kwargs...) =
    any(f, unwrap_invariant(array); kwargs...)
Base.all(f::Function, array::AnyInvariantArray; kwargs...) =
    all(f, unwrap_invariant(array); kwargs...)
Base.count(f, array::AnyInvariantArray; kwargs...) =
    count(f, unwrap_invariant(array); kwargs...)
LinearAlgebra.norm(array::AnyInvariantArray, p::Real = 2) =
    LinearAlgebra.norm(unwrap_invariant(array), p)
Base.Broadcast.BroadcastStyle(W::Type{<:AnyInvariantArray}) =
    Base.Broadcast.BroadcastStyle(Base.promote_op(unwrap_invariant, W))

# MPI reads arrays through pointer conversions that it only defines for Arrays and views.
ClimaComms.gather(ctx::ClimaComms.MPICommsContext, array::AnyInvariantArray) =
    ClimaComms.gather(ctx, unwrap_invariant(array))

llvm_pointer(ptr::Ptr{T}) where {T} = reinterpret(Core.LLVMPtr{T, 0}, ptr)
llvm_pointer(ptr::Core.LLVMPtr) = ptr

# The load of LLVM.Interop.pointerref at a 0-based offset, with !invariant.load metadata.
# Unlike CUDA.Const arrays, which are read with the ldg intrinsic before LLVM 20, these
# loads are merged across stores and barriers. LLVM.jl 10 generates the IR with
# @llvmgenerated, which replaces the create_function and call_function of earlier versions.
@static if isdefined(LLVM.Interop, Symbol("@llvmgenerated"))
    LLVM.Interop.@llvmgenerated builder function invariant_load(
        ptr::Core.LLVMPtr{T, A},
        offset::Int,
    )::T where {T, A}
        eltyp = convert(LLVM.LLVMType, T)
        if LLVM.supports_typed_pointers(LLVM.context())
            ptr = LLVM.bitcast!(builder, ptr, LLVM.PointerType(eltyp, A))
        end
        load = LLVM.load!(
            builder,
            eltyp,
            LLVM.inbounds_gep!(builder, eltyp, ptr, [offset]),
        )
        LLVM.metadata(load)[LLVM.MD_invariant_load] = LLVM.MDNode(LLVM.Metadata[])
        LLVM.alignment!(load, Base.datatype_alignment(T))
        load
    end
else
    @generated function invariant_load(ptr::Core.LLVMPtr{T, A}, offset::Int) where {T, A}
        LLVM.@dispose ctx = LLVM.Context() builder = LLVM.IRBuilder() begin
            eltyp = convert(LLVM.LLVMType, T)
            f, _ = LLVM.Interop.create_function(
                eltyp,
                [convert(LLVM.LLVMType, ptr), convert(LLVM.LLVMType, Int)],
            )
            LLVM.position!(builder, LLVM.BasicBlock(f, "entry"))
            base, index = LLVM.parameters(f)
            if LLVM.supports_typed_pointers(ctx)
                base = LLVM.bitcast!(builder, base, LLVM.PointerType(eltyp, A))
            end
            load = LLVM.load!(
                builder,
                eltyp,
                LLVM.inbounds_gep!(builder, eltyp, base, [index]),
            )
            LLVM.metadata(load)[LLVM.MD_invariant_load] = LLVM.MDNode(LLVM.Metadata[])
            LLVM.alignment!(load, Base.datatype_alignment(T))
            LLVM.ret!(builder, load)
            LLVM.Interop.call_function(f, T, Tuple{ptr, Int}, :ptr, :offset)
        end
    end
end

"""
    invariant_data(data)

Wrap the parent arrays of `data` in [`InvariantArray`](@ref)s, whose loads are marked
invariant and which cannot be modified. `data` can be a [`DataLayout`](@ref), a `Field` or
`FieldVector` (whose spaces are unchanged), a `FieldNameDict` (e.g., a `FieldMatrix`), or
any other object that `Adapt` can convert, whose arrays are wrapped. Arrays that are already
`InvariantArray`s are not wrapped again.
"""
invariant_data(data) = Adapt.adapt(WrapInvariantArrays(), data)
# invariant_data goes through Adapt, so that a call with a layout whose type is not inferred
# (e.g., in grid construction, where layout sizes are not inferred) is one dynamic dispatch.
