#=
This file is for providing a list of arguments
to call different functions with.
=#

#! format: off

#####
##### Helpers
#####
Base.rand(::Type{T}) where {FT, T <: XZPoint{FT}} = T(rand(FT),rand(FT))
Base.rand(::Type{T}) where {FT, T <: XYZPoint{FT}} = T(rand(FT),rand(FT),rand(FT))
Base.rand(::Type{T}) where {FT, T <: LatLongZPoint{FT}} = T(rand(FT),rand(FT),rand(FT))
Base.rand(::Type{T}) where {FT, T <: XYPoint{FT}} = T(rand(FT),rand(FT))
Base.rand(::Type{T}) where {FT, T <: ZPoint{FT}} = T(rand(FT))
Base.rand(::Type{T}) where {FT, T <: LatLongPoint{FT}} = T(rand(FT),rand(FT))
Base.rand(::Type{T}) where {FT, T <: XPoint{FT}} = T(rand(FT))

get_∂x∂ξ(::Type{FT}, I, ::Type{S}) where {FT, S} = rand(Axis2Tensor{FT, Tuple{LocalAxis{I}, CovariantAxis{I}}, S})

get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{2, 2, FT, 4}, C <: XZPoint{FT}       , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(rand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{3, 3, FT, 9}, C <: XYZPoint{FT}      , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(rand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{3, 3, FT, 9}, C <: LatLongZPoint{FT} , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(rand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{2, 2, FT, 4}, C <: XYPoint{FT}       , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(rand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{1, 1, FT, 1}, C <: ZPoint{FT}        , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(rand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{2, 2, FT, 4}, C <: LatLongPoint{FT}  , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(rand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{1, 1, FT, 1}, C <: XPoint{FT}        , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(rand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))

#####
##### func args
#####

function func_args(FT, f::typeof(Geometry.project))
    map(func_arg_types(FT, f)) do at
        if length(at) == 3
            (at[1](), rand(at[2]), get_lg_instance(at[3])) # 3-argument method
        else
            (at[1](), rand(at[2])) # 2-argument method
        end
    end
end

function func_args(FT, f::typeof(Geometry.transform))
    map(func_arg_types(FT, f)) do at
        # TODO: don't use zeros, since this invalidates the correctness tests.
        if length(at) == 3
            (at[1](), zeros(at[2]), get_lg_instance(at[3])) # 3-argument method
        else
            (at[1](), zeros(at[2])) # 2-argument method
        end
    end
end

function func_args(FT, f::Union{
        typeof(Geometry.contravariant1),
        typeof(Geometry.contravariant2),
        typeof(Geometry.contravariant3),
    })
    map(func_arg_types(FT, f)) do at
        (rand(at[1]), get_lg_instance(at[2]))
    end
end

#! format: on
