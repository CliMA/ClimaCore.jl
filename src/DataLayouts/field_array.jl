
abstract type AbstractDataLayoutSingleton end
for DL in (
    :IJKFVH,
    :IJFH,
    :IFH,
    :DataF,
    :IJF,
    :IF,
    :VF,
    :VIJFH,
    :VIFH,
    :IH1JH2,
    :IV1JH2,
)
    @eval struct $(Symbol(DL, :Singleton)) <: AbstractDataLayoutSingleton end
end

@inline field_dim(::IJKFVHSingleton) = 4
@inline field_dim(::IJFHSingleton) = 3
@inline field_dim(::IFHSingleton) = 2
@inline field_dim(::DataFSingleton) = 1
@inline field_dim(::IJFSingleton) = 3
@inline field_dim(::IFSingleton) = 2
@inline field_dim(::VFSingleton) = 2
@inline field_dim(::VIJFHSingleton) = 4
@inline field_dim(::VIFHSingleton) = 3

"""
    EmptyArray{T, N} <: AbstractArray{T, N}

A special type for supporting empty fields (Nf = 0).
"""
struct EmptyArray{FT, N, A <: AbstractArray{FT, N}} <: AbstractArray{FT, N} end
EmptyArray(::Type{A}) where {A <: AbstractArray} =
    EmptyArray{eltype(A), ndims(A), A}()
EmptyArray(A::AbstractArray) = EmptyArray(typeof(A))
Base.eltype(A::EmptyArray{FT, N}) where {FT, N} = FT
Base.eltype(::Type{EmptyArray{FT, N, A}}) where {FT, N, A} = FT
Base.ndims(::Type{EmptyArray{FT, N, A}}) where {FT, N, A} = N
Base.ndims(::EmptyArray{FT, N}) where {FT, N} = N
Base.length(::EmptyArray) = 0 # needed for printing
Base.size(ea::EmptyArray) = () # needed for printing, we don't keep the size, not sure what to do.
parent_array_type(::Type{EmptyArray{FT, N, A}}) where {FT, N, A} = A

"""
    FieldArray{FD, TUP <: Union{NTuple{Nf,AbstractArray}, EmptyArray}}

An `Array` that splits the field dimension, `FD`
into tuples of arrays across all other dimensions.
"""
struct FieldArray{FD, TUP <: Union{NTuple, EmptyArray}}
    arrays::TUP
    FieldArray{FD}(arrays::NT) where {FD, N, NT <: NTuple{N, <:AbstractArray}} =
        new{FD, typeof(arrays)}(arrays)
    FieldArray{FD}(ef::EmptyArray) where {FD} = new{FD, typeof(ef)}(ef)
    FieldArray{FD, TUP}(
        fa::TUP,
    ) where {FD, N, TUP <: NTuple{N, <:AbstractArray}} = new{FD, TUP}(fa)
    FieldArray{FD, EA}(fa::EA) where {FD, EA <: EmptyArray} =
        FieldArray{FD, EA}(fa)
end

field_array_type(::Type{T}) where {T <: FieldArray} = T
arrays_type(::Type{T}) where {FD, TUP, T <: FieldArray{FD, TUP}} = TUP

const EmptyFieldArray{FD, FT, N, A} = FieldArray{FD, EmptyArray{FT, N, A}}
const NonEmptyFieldArray{FD, N, T} = FieldArray{FD, NTuple{N, T}}

field_dim(::FieldArray{FD}) where {FD} = FD
field_dim(::Type{FieldArray{FD, NT}}) where {FD, NT <: NTuple} = FD
function Adapt.adapt_structure(
    to,
    fa::FieldArray{FD, NT},
) where {FD, NT <: NTuple}
    arrays = ntuple(i -> Adapt.adapt(to, fa.arrays[i]), tuple_length(NT))
    FieldArray{FD}(arrays)
end
function rebuild(
    fa::FieldArray{FD, NT},
    ::Type{DA},
) where {FD, NT <: NTuple, DA}
    arrays = ntuple(i -> DA(fa.arrays[i]), Val(tuple_length(NT)))
    FieldArray{FD}(arrays)
end

Base.length(fa::FieldArray{FD, <:EmptyArray}) where {FD} = 0
Base.length(fa::FieldArray{FD, NT}) where {FD, NT <: NTuple} = tuple_length(NT)
full_length(fa::FieldArray) = length(fa) * prod(elsize(fa))
Base.copy(fa::FieldArray{FD, NT}) where {FD, NT <: NTuple} =
    FieldArray{FD}(ntuple(i -> copy(fa.arrays[i]), tuple_length(NT)))

import LinearAlgebra

function LinearAlgebra.mul!(Y::FA, A, B::FA) where {FA <: FieldArray}
    @inbounds for i in 1:ncomponents(Y)
        LinearAlgebra.mul!(vec(Y.arrays[i]), A, vec(B.arrays[i]))
    end
    Y
end

tuple_length(::Type{NT}) where {N, NT <: NTuple{N}} = N
float_type(::Type{FieldArray{FD, NT}}) where {FD, NT <: NTuple} = eltype(NT)
parent_array_type(::Type{FieldArray{FD, NT}}) where {FD, NT <: NTuple} =
    parent_array_type(eltype(NT))
parent_array_type(::FieldArray{FD, NT}) where {FD, NT <: NTuple} =
    parent_array_type(eltype(NT))
parent_array_type(::Type{EmptyFieldArray{FD, FT, N, A}}) where {FD, FT, N, A} =
    parent_array_type(A)

rebuild_type(
    ::Type{FieldArray{FD, NT}},
    as::ArraySize{FD, Nf},
) where {FD, NT <: NTuple, Nf} =
    FieldArray{FD, NTuple{Nf, parent_array_type(eltype(NT), as)}}

function rebuild_field_array_type end

rebuild_field_array_type(
    ::Type{FieldArray{FD, NT}},
    as::ArraySize{FD, Nf},
) where {FD, NT <: NTuple, Nf} =
    Nf == 0 ?
    EmptyFieldArray{
        FD,
        eltype(eltype(NT)),
        ndims(as),
        parent_array_type(eltype(NT), as),
    } : FieldArray{FD, NTuple{Nf, parent_array_type(eltype(NT), as)}}
rebuild_field_array_type(
    ::Type{T},
    as::ArraySize{FD, Nf, S},
) where {FD, T <: AbstractArray, Nf, S} =
    Nf == 0 ?
    EmptyFieldArray{FD, eltype(T), ndims(as), parent_array_type(T, as)} :
    FieldArray{FD, NTuple{Nf, parent_array_type(T, as)}}

# MArrays
rebuild_field_array_type(
    ::Type{FieldArray{FD, NT}},
    as::ArraySize{FD, Nf},
    ::Type{MAT},
) where {FD, NT <: NTuple, Nf, MAT <: MArray} =
    Nf == 0 ? EmptyFieldArray{FD, eltype(MAT), ndims(MAT), MAT} :
    FieldArray{FD, NTuple{Nf, MAT}}
rebuild_field_array_type(
    ::Type{T},
    as::ArraySize{FD, Nf},
    ::Type{MAT},
) where {T <: AbstractArray, FD, Nf, MAT <: MArray} =
    Nf == 0 ? EmptyFieldArray{FD, FT, N, MAT} : FieldArray{FD, NTuple{Nf, MAT}}

# Empty left-hand side case:
rebuild_field_array_type(
    ::Type{EmptyFieldArray{FD, FT, N, A}},
    as::ArraySize{FD, Nf},
) where {FD, FT, N, A, Nf} =
    Nf == 0 ? EmptyFieldArray{FD, FT, N, parent_array_type(A, as)} :
    FieldArray{FD, NTuple{Nf, parent_array_type(A, as)}}
# Empty left-hand side case with MArray:
rebuild_field_array_type(
    ::Type{EmptyFieldArray{FD, FT, N, A}},
    as::ArraySize{FD, Nf},
    ::Type{MAT},
) where {FD, FT, N, A, Nf, MAT <: MArray} =
    Nf == 0 ? EmptyFieldArray{FD, FT, ndims(MAT), MAT} :
    FieldArray{FD, NTuple{Nf, MAT}}

Base.ndims(::Type{FieldArray{FD, NT}}) where {FD, NT <: NTuple} =
    Base.ndims(eltype(NT)) + 1
Base.eltype(::Type{FieldArray{FD, NT}}) where {FD, NT <: NTuple} =
    eltype(eltype(NT))
array_type(::Type{FieldArray{FD, NT}}) where {FD, NT <: NTuple} = eltype(NT)
ncomponents(::Type{FieldArray{FD, NT}}) where {FD, NT <: NTuple} =
    tuple_length(NT)
array_type(fa::FieldArray) = array_type(typeof(fa))
ncomponents(fa::FieldArray) = ncomponents(typeof(fa))

ncomponents(::Type{EmptyFieldArray{FD, FT, N, A}}) where {FD, FT, N, A} = 0
ncomponents(::EmptyFieldArray{FD, FT, N, A}) where {FD, FT, N, A} = 0

# Base.size(fa::FieldArray{N,T}) where {N,T} = size(fa.arrays[1])

promote_parent_array_type(
    ::Type{FieldArray{FDA, NTA}},
    ::Type{FieldArray{FDB, NTB}},
) where {FDA, FDB, NTA <: NTuple, NTB <: NTuple} =
# FieldArray{N, promote_parent_array_type(A, B)} where {N}
    promote_parent_array_type(eltype(NTA), eltype(NTB))

elsize(fa::FieldArray) = size(fa.arrays[1])

@inline function Base.copyto!(x::FA, y::FA) where {FA <: FieldArray}
    @inbounds for i in 1:ncomponents(x)
        Base.copyto!(x.arrays[i], y.arrays[i])
    end
end

@inline function Base.copy!(x::FA, y::FA) where {FA <: FieldArray}
    @inbounds for i in 1:ncomponents(x)
        Base.copy!(x.arrays[i], y.arrays[i])
    end
    x
end

@inline function Base.:(+)(x::FA, y::FA) where {FA <: FieldArray}
    return @inbounds ntuple(Val(ncomponents(x))) do i
        Base.:(+)(x.arrays[i], y.arrays[i])
    end
end
@inline function Base.:(-)(x::FA, y::FA) where {FA <: FieldArray}
    return @inbounds ntuple(Val(ncomponents(x))) do i
        Base.:(-)(x.arrays[i], y.arrays[i])
    end
end

function Base.fill!(fa::FieldArray, val)
    @inbounds for i in 1:ncomponents(fa)
        fill!(fa.arrays[i], val)
    end
    return fa
end

@inline function Base.copyto!(
    x::FieldArray{FD, NT},
    y::AbstractArray,
) where {FD, NT <: NTuple}
    if ndims(eltype(NT)) == ndims(y)
        @inbounds for i in 1:tuple_length(NT)
            Base.copyto!(x.arrays[i], y)
        end
    elseif ndims(eltype(NT)) + 1 == ndims(y)
        @inbounds for I in CartesianIndices(y)
            x[I] = y[I]
        end
    end
    x
end
import StaticArrays
function SArray(fa::FieldArray)
    tup = ntuple(ncomponents(fa)) do f
        SArray(fa.arrays[f])
    end
    return FieldArray{field_dim(fa)}(tup)
end
Base.Array(fa::FieldArray) = Array(collect(fa))

Base.similar(
    fa::FieldArray{FD, NT},
    ::Type{ElType},
) where {FD, NT <: NTuple, ElType} =
    FieldArray{FD}(ntuple(_ -> similar(eltype(NT), ElType), tuple_length(NT)))

Base.similar(
    fa::EmptyFieldArray{FD, FT, N, A},
    ::Type{ElType},
) where {FD, FT, N, A, ElType} = FieldArray{FD}(EmptyArray(similar(A, ElType)))

Base.similar(
    fa::FieldArray{FD, NT},
    a::AbstractArray,
) where {FD, NT <: NTuple} =
    FieldArray{FD}(ntuple(_ -> similar(a), tuple_length(NT)))
Base.similar(fa::EmptyFieldArray{FD}, a::AbstractArray) where {FD} =
    FieldArray{FD}(EmptyArray(a))

Base.similar(
    ::Type{FieldArray{FD, NT}},
    a::AbstractArray,
) where {FD, NT <: NTuple} =
    FieldArray{FD}(ntuple(_ -> similar(a), tuple_length(NT)))
Base.similar(
    ::Type{EmptyFieldArray{FD, FT, N, A}},
    a::AbstractArray,
) where {FD, FT, N, A} = FieldArray{FD}(EmptyArray(a))


Base.similar(
    ::Type{FA},
    a::AbstractArray,
    ::Val{Nf},
) where {FD, Nf, FA <: FieldArray{FD}} =
    FieldArray{FD}(ntuple(i -> similar(a), Val(Nf)))
Base.similar(
    ::Type{FA},
    a::AbstractArray,
    ::Val{0},
) where {FD, FA <: FieldArray{FD}} = FieldArray{FD}(EmptyArray(a))

Base.similar(::Type{FieldArray{FD, NT}}, dims::Dims) where {FD, NT <: NTuple} =
    FieldArray{FD}(ntuple(i -> similar(eltype(NT), dims), tuple_length(NT)))
Base.similar(
    ::Type{EmptyFieldArray{FD, FT, N, A}},
    dims::Dims,
) where {FD, FT, N, A} = FieldArray{FD}(EmptyArray(similar(A, dims)))

function Base.similar(::Type{FieldArray{FD, NT}}) where {FD, NT <: NTuple}
    T = eltype(NT)
    N = tuple_length(NT)
    isconcretetype(T) || error(
        "Array type $T is not concrete, pass `dims` to similar or use concrete array type.",
    )
    FieldArray{FD, N, T}(ntuple(i -> similar(T), N))
end
function Base.similar(
    ::Type{EmptyFieldArray{FD, FT, N, A}},
) where {FD, FT, N, A}
    isconcretetype(T) || error(
        "Array type $T is not concrete, pass `dims` to similar or use concrete array type.",
    )
    FieldArray{FD}(EmptyArray(similar(T)))
end

@inline insertafter(t::Tuple, i::Int, j::Int) =
    0 <= i <= length(t) ? _insertafter(t, i, j) : throw(BoundsError(t, i))
@inline _insertafter(t::Tuple, i::Int, j::Int) =
    i == 0 ? (j, t...) : (t[1], _insertafter(Base.tail(t), i - 1, j)...)

function original_dims(fa::FieldArray{FD, NT}) where {FD, NT <: NTuple}
    insertafter(elsize(fa), FD - 1, tuple_length(NT))
end

function Base.collect(fa::EmptyFieldArray{FD, FT, N, A}) where {FD, FT, N, A}
    similar(parent_array_type(A))
end

function Base.collect(fa::FieldArray{FD, NT}) where {FD, NT <: NTuple}
    odims = original_dims(fa)
    ND = ndims(eltype(NT)) + 1
    a = similar(fa.arrays[1], eltype(eltype(NT)), odims)
    @inbounds for i in 1:tuple_length(NT)
        Ipre = ntuple(i -> Colon(), Val(FD - 1))
        Ipost = ntuple(i -> Colon(), Val(ND - FD))
        av = view(a, Ipre..., i, Ipost...)
        Base.copyto!(av, fa.arrays[i])
    end
    return a
end
import LinearAlgebra
LinearAlgebra.adjoint(fa::FieldArray{FD}) where {FD} = FieldArray{FD}(
    ntuple(i -> LinearAlgebra.adjoint(fa.arrays[i]), Val(ncomponents(fa))),
)

Base.reinterpret(::Type{T}, fa::FieldArray{FD}) where {T, FD} = FieldArray{FD}(
    ntuple(i -> reinterpret(T, fa.arrays[i]), Val(ncomponents(fa))),
)

# TODO: remove need to wrap in ArrayType
similar_rand(fa::FieldArray) = rand(eltype(fa), elsize(fa))
field_arrays(fa::FieldArray) = getfield(fa, :arrays)
field_arrays(data::AbstractData) = field_arrays(parent(data))



field_array(array::AbstractArray, s::AbstractDataLayoutSingleton) =
    field_array(array, field_dim(s))

# TODO: is this even needed? (this is likely not a good way to construct)
field_array(array::AbstractArray, ::Type{T}) where {T <: AbstractData} =
    field_array(array, singleton(T))

function field_array(array::AbstractArray, fdim::Integer)
    as = ArraySize{fdim, size(array, fdim), size(array)}()
    field_array(array, as)
end

function field_array(
    array::AbstractArray,
    as::ArraySize{FD, Nf, S},
) where {FD, Nf, S}
    fdim = FD
    s = S
    _Nf::Int = Nf::Int
    snew = dropat(s, Val(FD))
    scalar_array = similar(array, eltype(array), snew)
    arrays = ntuple(Val(_Nf)) do i
        similar(scalar_array)
    end
    ND = ndims(array) # TODO: confirm
    Ipre = ntuple(i -> Colon(), Val(fdim - 1))
    Ipost = ntuple(i -> Colon(), Val(ND - fdim))

    for i in 1:_Nf
        arrays[i] .= array[Ipre..., i, Ipost...]
    end
    return FieldArray{fdim}(arrays)
end

function field_array(
    array::AbstractArray,
    as::ArraySize{FD, 0, S},
) where {FD, S}
    snew = dropat(S, Val(FD))
    scalar_array = similar(array, eltype(array), snew)
    FieldArray{FD}(EmptyArray(scalar_array))
end

# Warning: this method is type-unstable.
function Base.view(fa::FieldArray{FD}, inds...) where {FD}
    AI = dropat(inds, Val(FD))
    if all(x -> x isa Colon, AI)
        FDI = inds[FD]
        tup = if FDI isa AbstractRange
            Tuple(FDI)
        else
            @show FDI
            error("Uncaught case")
        end
        arrays = ntuple(fj -> fa.arrays[tup[fj]], length(tup))
    else
        arrays = ntuple(ncomponents(fa)) do fj
            view(fa.arrays[fj], AI...)
        end
    end
    return FieldArray{FD}(arrays)
end
Base.iterate(fa::FieldArray, state = 1) = Base.iterate(collect(fa), state)

function Base.:(==)(fa::NonEmptyFieldArray, array::AbstractArray)
    return all(collect(fa) .== array)
end
function Base.:(==)(a::NonEmptyFieldArray, b::NonEmptyFieldArray)
    return all(collect(a) .== collect(b))
end


# These are not strictly true, but empty fields have no data,
# so the only thing that this equality misses is that the
# original array size (which is not preserved when making
# an EmptyFieldArray) is equal.
Base.:(==)(fa::EmptyFieldArray, array::AbstractArray) =
    ndims(fa) == ndims(array) && eltype(fa) == eltype(array)
Base.:(==)(a::FA, b::FA) where {FA <: EmptyFieldArray} = true


Base.getindex(fa::FieldArray, I::Integer...) = getindex(fa, CartesianIndex(I))

Base.getindex(fa::FieldArray, I...) = collect(fa)[I...]

Base.reshape(fa::FieldArray, inds...) = Base.reshape(collect(fa), inds...)
Base.vec(fa::FieldArray) = Base.vec(collect(fa))
Base.isapprox(x::FieldArray, y::FieldArray; kwargs...) =
    Base.isapprox(collect(x), collect(y); kwargs...)
Base.isapprox(x::FieldArray, y::AbstractArray; kwargs...) =
    Base.isapprox(collect(x), y; kwargs...)

# drop element `i` in tuple `t`
@inline function dropat(t::NT, ::Val{i}) where {NT <: NTuple, i}
    if 1 <= i <= length(t)
        _dropat(t, Val(i))::NTuple{tuple_length(NT) - 1, Int}
    else
        throw(BoundsError(t, i))
    end
end
@inline function dropat(t::NT, ::Val{i}) where {N, NT <: NTuple{N, Int}, i}
    if 1 <= i <= length(t)
        _dropat(t, Val(i))::NTuple{tuple_length(NT) - 1, Int}
    else
        throw(BoundsError(t, i))
    end
end
@inline function dropat(t::Tuple, ::Val{i}) where {i}
    if 1 <= i <= length(t)
        _dropat(t, Val(i))
    else
        throw(BoundsError(t, i))
    end
end
@inline _dropat(t::Tuple, ::Val{i}) where {i} =
    i == 1 ? (Base.tail(t)...,) : (t[1], _dropat(Base.tail(t), Val(i - 1))...)

function Base.getindex(fa::FieldArray{FD}, I::CartesianIndex) where {FD}
    FDI = I.I[FD]
    ND = length(I.I)
    IA = CartesianIndex(dropat(I.I, Val(FD)))
    return fa.arrays[FDI][IA]
end

function Base.setindex!(fa::FieldArray{FD}, val, I::CartesianIndex) where {FD}
    FDI = I.I[FD]
    ND = length(I.I)
    IA = CartesianIndex(dropat(I.I, Val(FD)))
    fa.arrays[FDI][IA] = val
end
