
abstract type AbstractDataLayoutSingleton end
for DL in (:IJKFVH,:IJFH,:IFH,:DataF,:IJF,:IF,:VF,:VIJFH,:VIFH,:IH1JH2,:IV1JH2)
    @eval struct $(Symbol(DL, :Singleton)) <: AbstractDataLayoutSingleton end
end

# struct IJKFVHSingleton <: AbstractDataLayoutSingleton end
# struct IJFHSingleton <: AbstractDataLayoutSingleton end
# struct IFHSingleton <: AbstractDataLayoutSingleton end
# struct DataFSingleton <: AbstractDataLayoutSingleton end
# struct IJFSingleton <: AbstractDataLayoutSingleton end
# struct IFSingleton <: AbstractDataLayoutSingleton end
# struct VFSingleton <: AbstractDataLayoutSingleton end
# struct VIJFHSingleton <: AbstractDataLayoutSingleton end
# struct VIFHSingleton <: AbstractDataLayoutSingleton end
# struct IH1JH2Singleton <: AbstractDataLayoutSingleton end
# struct IV1JH2Singleton <: AbstractDataLayoutSingleton end

@inline field_dim(::IJKFVHSingleton) = 4
@inline field_dim(::IJFHSingleton) = 3
@inline field_dim(::IFHSingleton) = 2
@inline field_dim(::DataFSingleton) = 1
@inline field_dim(::IJFSingleton) = 3
@inline field_dim(::IFSingleton) = 2
@inline field_dim(::VFSingleton) = 2
@inline field_dim(::VIJFHSingleton) = 4
@inline field_dim(::VIFHSingleton) = 3

struct TupleOfArrays{N,T}
    arrays::NTuple{N,T}
end
Base.ndims(::Type{TupleOfArrays{N,T}}) where {N,T} = Base.ndims(T)+1
Base.eltype(::Type{TupleOfArrays{N,T}}) where {N,T} = eltype(T)
# Base.size(toa::TupleOfArrays{N,T}) where {N,T} = size(toa.arrays[1])

pre_post_colons(f, dim) = (;
    Ipre = ntuple(i->Colon(), Val(length(size(f)[1:dim-1]))),
    Ipost = ntuple(i->Colon(), Val(length(size(f)[dim+1:end])))
)

tuple_of_arrays(array::AbstractArray, s::AbstractDataLayoutSingleton) =
    tuple_of_arrays(array, field_dim(s))

function tuple_of_arrays(array::AbstractArray, fdim::Integer)
    Nf = size(array, fdim)
    if Nf == 1
        arrays = (dropdims(array; dims=fdim),)
    else
        s = size(array)
        snew = Tuple(map(j->s[j], filter(i -> i≠fdim, 1:length(s))))
        farray = similar(array, eltype(array), snew)
        arrays = ntuple(Nf) do i
            similar(farray)
        end
        (;Ipre, Ipost) = pre_post_colons(array, fdim)
        S = map(x->size(x), arrays)
        for i in 1:Nf
            arrays[i] .= array[Ipre..., i, Ipost...]
        end
    end
    return TupleOfArrays{Nf,eltype(arrays)}(arrays)
end
