
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

struct FieldArray{FD, N, T <: AbstractArray}
    arrays::NTuple{N, T}
    FieldArray{FD}(arrays::NTuple{N, T}) where {FD, N, T} = new{FD, N, T}(arrays)
    FieldArray{FD}(::Tuple{}) where {FD} = error("FieldArray does not support empty fields.")
    FieldArray{FD, N}(
        fa::FA,
    ) where {FD, N, T <: AbstractArray, FA <: NTuple{N, T}} =
        FieldArray{FD}(fa)
    FieldArray{FD, N, T}(
        fa::FA,
    ) where {FD, N, T <: AbstractArray, FA <: NTuple{N, T}} =
        FieldArray{FD}(fa)
end

# FieldArray{FD}(fa::FA) where {FD, N, T <: AbstractArray, FA <: NTuple{N, T}} =
#     FieldArray{FD, N, T}(fa)


field_dim(::FieldArray{FD, N, T}) where {FD, N, T} = FD
field_dim(::Type{FieldArray{FD, N, T}}) where {FD, N, T} = FD
function Adapt.adapt_structure(to, fa::FieldArray{FD, N, T}) where {FD, N, T}
    arrays = ntuple(i -> Adapt.adapt(to, fa.arrays[i]), N)
    FieldArray{FD}(arrays)
end
function rebuild(fa::FieldArray{FD, N, T}, ::Type{DA}) where {FD, N, T, DA}
    arrays = ntuple(i -> DA(fa.arrays[i]), Val(N))
    FieldArray{FD}(arrays)
end

Base.length(fa::FieldArray{FD, N, T}) where {FD, N, T} = N
Base.copy(fa::FieldArray{FD, N, T}) where {FD, N, T} =
    FieldArray{FD}(ntuple(i -> copy(fa.arrays[i]), N))

float_type(::Type{FieldArray{FD, N, T}}) where {FD, N, T} = eltype(T)
parent_array_type(::Type{FieldArray{FD, N, T}}) where {FD, N, T} =
    parent_array_type(T)
# field_array_type(::Type{FieldArray{N,T}}, ::Val{Nf}) where {N,T, Nf} = FieldArray{Nf, parent_array_type(T, Val(ndims(T)))}
# field_array_type(
#     ::Type{FieldArray{FD, N, T}},
#     ::Val{Nf},
#     ::Val{ND},
# ) where {FD, N, T, Nf, ND} = FieldArray{FD, Nf, parent_array_type(T, Val(ND))}

rebuild_type(
    ::Type{FieldArray{FD, N, T}},
    as::ArraySize{FD,Nf},
) where {FD, N, T, Nf} = FieldArray{FD, Nf, parent_array_type(T, as)}

rebuild_field_array_type(
    ::Type{FieldArray{FD, N, T}},
    as::ArraySize{FD,Nf},
) where {FD, N, T, Nf} = FieldArray{FD, Nf, parent_array_type(T, as)}

rebuild_field_array_type(
    ::Type{T},
    as::ArraySize{FD,Nf,S},
) where {FD, T<:AbstractArray, Nf, S} = FieldArray{FD, Nf, parent_array_type(T, as)}

# field_array_type(
#     ::Type{T},
#     ::Val{FD},
#     ::Val{Nf},
#     ::Val{ND},
# ) where {T <: AbstractArray, FD, Nf, ND} =
#     FieldArray{FD, Nf, parent_array_type(T, Val(ND))}
Base.ndims(::Type{FieldArray{FD, N, T}}) where {FD, N, T} = Base.ndims(T) + 1
Base.eltype(::Type{FieldArray{FD, N, T}}) where {FD, N, T} = eltype(T)
array_type(::Type{FieldArray{FD, N, T}}) where {FD, N, T} = T
ncomponents(::Type{FieldArray{FD, N, T}}) where {FD, N, T} = N
ncomponents(::FieldArray{FD, N, T}) where {FD, N, T} = N
# Base.size(fa::FieldArray{N,T}) where {N,T} = size(fa.arrays[1])
to_parent_array_type(::Type{FA}) where {FD, N, T, FA <: FieldArray{FD, N, T}} =
    FieldArray{FD, N, parent_array_type(FA)}

promote_parent_array_type(
    ::Type{FieldArray{FDA, NA, A}},
    ::Type{FieldArray{FDB, NB, B}},
) where {FDA, FDB, NA, NB, A, B} =
# FieldArray{N, promote_parent_array_type(A, B)} where {N}
    promote_parent_array_type(A, B)

elsize(fa::FieldArray{FD, N, T}) where {FD, N, T} = size(fa.arrays[1])

@inline function Base.copyto!(
    x::FA,
    y::FA,
) where {FD, N, T, FA <: FieldArray{FD, N, T}}
    @inbounds for i in 1:N
        Base.copyto!(x.arrays[i], y.arrays[i])
    end
end

function Base.fill!(fa::FieldArray, val)
    @inbounds for i in 1:ncomponents(fa)
        fill!(fa.arrays[i], val)
    end
    return fa
end

@inline function Base.copyto!(
    x::FieldArray{FD, N, T},
    y::AbstractArray{TA, NA},
) where {FD, N, T, TA, NA}
    if ndims(T) == NA
        @inbounds for i in 1:N
            Base.copyto!(x.arrays[i], y)
        end
    elseif ndims(T) + 1 == NA
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
# function field_array_size(fa::FieldArray)
#     @show elsize(fa)
#     @show field_dim(fa)
#     @show ncomponents(fa)
#     s = insertafter(elsize(fa), field_dim(fa)-1, ncomponents(fa))
#     @show s
#     @show original_dims(fa)
#     return s
# end
Base.Array(fa::FieldArray) = collect(fa)
Base.similar(fa::FieldArray{FD, N, T}, ::Type{ElType}) where {FD, N, T, ElType} =
    FieldArray{FD, N, T}(ntuple(i -> similar(T, ElType), N))
# Base.similar(fa::FieldArray{FD, N, T}, et=eltype(fa), dims=field_array_size(fa)) where {FD, N, T} =
#     FieldArray{FD, N, T}(ntuple(i -> similar(T, et, dims), N))
function Base.similar(::Type{FieldArray{FD, N, T}}, dims) where {FD, N, T}
    FieldArray{FD, N, T}(ntuple(i -> similar(T, dims), N))
end

function Base.similar(::Type{FieldArray{FD, N, T}}) where {FD, N, T}
    isconcretetype(T) || error("Array type $T is not concrete, pass `dims` to similar or use concrete array type.")
    FieldArray{FD, N, T}(ntuple(i -> similar(T), N))
end

@inline insertafter(t::Tuple, i::Int, j::Int) =
    0 <= i <= length(t) ? _insertafter(t, i, j) : throw(BoundsError(t, i))
@inline _insertafter(t::Tuple, i::Int, j::Int) =
    i == 0 ? (j, t...) : (t[1], _insertafter(Base.tail(t), i - 1, j)...)

original_dims(fa::FieldArray{FD, N, T}) where {FD, N, T} =
    insertafter(elsize(fa), FD - 1, N)
function Base.collect(fa::FieldArray{FD, N, T}) where {FD, N, T}
    odims = original_dims(fa)
    ND = ndims(T) + 1
    a = similar(fa.arrays[1], eltype(T), odims)
    @inbounds for i in 1:N
        Ipre = ntuple(i -> Colon(), Val(FD - 1))
        Ipost = ntuple(i -> Colon(), Val(ND - FD))
        av = view(a, Ipre..., i, Ipost...)
        Base.copyto!(av, fa.arrays[i])
    end
    return a
end

field_array(array::AbstractArray, s::AbstractDataLayoutSingleton) =
    field_array(array, field_dim(s))

field_array(array::AbstractArray, ::Type{T}) where {T<:AbstractData} =
    field_array(array, singleton(T))

field_arrays(fa::FieldArray) = getfield(fa, :arrays)
field_arrays(data::AbstractData) = field_arrays(parent(data))

function field_array(array::AbstractArray, fdim::Integer)
    as = ArraySize{fdim, size(array, fdim), size(array)}()
    field_array(array, as)
end

function field_array(array::AbstractArray, as::ArraySize{FD,Nf,S}) where {FD, Nf, S}
    # Nf = size(array, fdim)
    fdim = FD
    # if Nf == 1
    #     # if array isa SArray
    #     #     arrays = (dropdims(array; dims = fdim),)
    #     # else
    #     @show fdim
    #     arrays = (dropdims(array; dims = fdim),)
    #     # end
    # else
        s = S
        # snew = Tuple(map(j -> s[j], filter(i -> i ≠ fdim, 1:length(s))))
        snew = dropat(s, FD)
        farray = similar(array, eltype(array), snew)
        arrays = ntuple(Val(Nf)) do i
            similar(farray)
        end
        # Ipre = ntuple(i -> Colon(), Val(length(size(array)[1:(fdim - 1)])))
        # Ipost = ntuple(i -> Colon(), Val(length(size(array)[(fdim + 1):end])))
        ND = ndims(array) # TODO: confirm
        Ipre = ntuple(i -> Colon(), Val(fdim - 1))
        Ipost = ntuple(i -> Colon(), Val(ND - fdim))

        for i in 1:Nf
            arrays[i] .= array[Ipre..., i, Ipost...]
        end
    # end
    return FieldArray{fdim}(arrays)
end

# Warning: this method is type-unstable.
function Base.view(fa::FieldArray{FD}, inds...) where {FD}
    AI = dropat(inds, FD)
    if all(x->x isa Colon, AI)
        FDI = inds[FD]
        tup = if FDI isa AbstractRange
            Tuple(FDI)
        else
            @show FDI
            error("Uncaught case")
        end
        arrays = ntuple(fj->fa.arrays[tup[fj]], length(tup))
    else
        arrays = ntuple(ncomponents(fa)) do fj
            view(fa.arrays[fj], AI...)
        end
    end
    return FieldArray{FD}(arrays)
end
Base.iterate(fa::FieldArray, state = 1) = Base.iterate(collect(fa), state)

function Base.:(==)(fa::FieldArray, array::AbstractArray)
    return collect(fa) == array
end

Base.getindex(fa::FieldArray, I::Integer...) = getindex(fa, CartesianIndex(I))

Base.getindex(fa::FieldArray, I...) = collect(fa)[I...]

Base.reshape(fa::FieldArray, inds...) = Base.reshape(collect(fa), inds...)
Base.vec(fa::FieldArray) = Base.vec(collect(fa))
Base.isapprox(x::FieldArray, y::FieldArray; kwargs...) =
    Base.isapprox(collect(x), collect(y); kwargs...)
Base.isapprox(x::FieldArray, y::AbstractArray; kwargs...) =
    Base.isapprox(collect(x), y; kwargs...)

# drop element `i` in tuple `t`
@inline dropat(t::Tuple, i::Int) =
    1 <= i <= length(t) ? _dropat(t, i) : throw(BoundsError(t, i))
@inline _dropat(t::Tuple, i::Int) =
    i == 1 ? (Base.tail(t)...,) : (t[1], _dropat(Base.tail(t), i - 1)...)

function Base.getindex(fa::FieldArray{FD}, I::CartesianIndex) where {FD}
    FDI = I.I[FD]
    ND = length(I.I)
    IA = CartesianIndex(dropat(I.I, FD))
    # Ipre = ntuple(i -> I.I[i], Val(FD - 1))
    # Ipost = ntuple(i -> I.I[i], Val(ND - FD))
    # IA = CartesianIndex((Ipre..., Ipost...))
    return fa.arrays[FDI][IA]
end

function Base.setindex!(fa::FieldArray{FD}, val, I::CartesianIndex) where {FD}
    FDI = I.I[FD]
    ND = length(I.I)
    IA = CartesianIndex(dropat(I.I, FD))
    # Ipre = ntuple(i -> I.I[i], Val(FD - 1))
    # Ipost = ntuple(i -> I.I[i], Val(ND - FD))
    # IA = CartesianIndex((Ipre..., Ipost...))
    fa.arrays[FDI][IA] = val
end
