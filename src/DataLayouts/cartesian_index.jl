abstract type AbstractDataSpecificCartesianIndex{N} <:
              Base.AbstractCartesianIndex{N} end

struct DataSpecificCartesianIndex{N} <: AbstractDataSpecificCartesianIndex{N}
    I::CartesianIndex{N}
end

# Generic fallback
@propagate_inbounds Base.getindex(x, I::DataSpecificCartesianIndex) =
    Base.getindex(x, I.I)

@propagate_inbounds Base.setindex!(x, val, I::DataSpecificCartesianIndex) =
    Base.setindex!(x, val, I.I)

# Datalayouts
@propagate_inbounds function Base.getindex(
    data::AbstractData{S},
    I::DataSpecificCartesianIndex,
) where {S}
    @inbounds get_struct(parent(data), S, Val(field_dim(data)), I.I)
end
@propagate_inbounds function Base.setindex!(
    data::AbstractData{S},
    val,
    I::DataSpecificCartesianIndex,
) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
        I.I,
    )
end

@inline array_size(::IJKFVH{S, Nij, Nk, Nv, Nh}) where {S, Nij, Nk, Nv, Nh} =
    (Nij, Nij, Nk, 1, Nv, Nh)
@inline array_size(::IJFH{S, Nij, Nh}) where {S, Nij, Nh} = (Nij, Nij, 1, Nh)
@inline array_size(::IFH{S, Ni, Nh}) where {S, Ni, Nh} = (Ni, 1, Nh)
@inline array_size(::DataF{S}) where {S} = (1,)
@inline array_size(::IJF{S, Nij}) where {S, Nij} = (Nij, Nij, 1)
@inline array_size(::IF{S, Ni}) where {S, Ni} = (Ni, 1)
@inline array_size(::VF{S, Nv}) where {S, Nv} = (Nv, 1)
@inline array_size(::VIJFH{S, Nv, Nij, Nh}) where {S, Nv, Nij, Nh} =
    (Nv, Nij, Nij, 1, Nh)
@inline array_size(::VIFH{S, Nv, Ni, Nh}) where {S, Nv, Ni, Nh} =
    (Nv, Ni, 1, Nh)
