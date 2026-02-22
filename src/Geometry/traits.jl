"""
    GeometryRequirement

Trait types to describe the geometry data required by an operation.
"""
abstract type GeometryRequirement end

struct NeedsMinimal <: GeometryRequirement end
struct NeedsMetric <: GeometryRequirement end
struct NeedsFull <: GeometryRequirement end

geometry_requirement(::Any) = NeedsMinimal()

geometry_requirement(::typeof(_norm)) = NeedsFull()
geometry_requirement(::typeof(_norm_sqr)) = NeedsFull()
geometry_requirement(::typeof(_cross)) = NeedsFull()
geometry_requirement(::typeof(transform)) = NeedsFull()
geometry_requirement(::typeof(project)) = NeedsFull()
geometry_requirement(::Type{<:AxisVector}) = NeedsFull()
geometry_requirement(::Type{<:CartesianVector}) = NeedsFull()

max_requirement(::NeedsFull, ::GeometryRequirement) = NeedsFull()
max_requirement(::GeometryRequirement, ::NeedsFull) = NeedsFull()
max_requirement(::NeedsMetric, ::NeedsMinimal) = NeedsMetric()
max_requirement(::NeedsMinimal, ::NeedsMetric) = NeedsMetric()
max_requirement(a::T, ::T) where {T <: GeometryRequirement} = a

function geometry_requirement(bc::Base.Broadcast.Broadcasted)
    req = geometry_requirement(bc.f)
    for arg in bc.args
        req = max_requirement(req, geometry_requirement(arg))
    end
    return req
end
