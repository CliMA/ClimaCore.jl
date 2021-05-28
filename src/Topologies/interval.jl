"""
    IntervalTopology([domain::IntervalDomain, ]faces::AbstractVector)

A 1D topology of elements, with element boundaries at `faces`.
"""
struct IntervalTopology{D <: IntervalDomain, V} <: AbstractTopology
    domain::D
    faces::V
end


IntervalTopology(faces::AbstractVector) =
    IntervalTopology(IntervalDomain(faces[1], faces[end]), faces)
