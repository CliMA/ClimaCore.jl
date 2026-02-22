"""
    GeometryRequirement

Trait types to describe the geometry data required by an operation.

By default, operations require full geometry (conservative). Operations that only
need coordinates, J, WJ, and invJ can be explicitly marked as NeedsMinimal().
"""
abstract type GeometryRequirement end

struct NeedsMinimal <: GeometryRequirement end
struct NeedsMetric <: GeometryRequirement end
struct NeedsFull <: GeometryRequirement end

# Default to full geometry (conservative)
geometry_requirement(::Any) = NeedsFull()

# Max requirement hierarchy: Full > Metric > Minimal
# Same types return themselves
max_requirement(::NeedsFull, ::NeedsFull) = NeedsFull()
max_requirement(::NeedsMetric, ::NeedsMetric) = NeedsMetric()
max_requirement(::NeedsMinimal, ::NeedsMinimal) = NeedsMinimal()

# Full dominates everything
max_requirement(::NeedsFull, ::NeedsMetric) = NeedsFull()
max_requirement(::NeedsFull, ::NeedsMinimal) = NeedsFull()
max_requirement(::NeedsMetric, ::NeedsFull) = NeedsFull()
max_requirement(::NeedsMinimal, ::NeedsFull) = NeedsFull()

# Metric dominates Minimal
max_requirement(::NeedsMetric, ::NeedsMinimal) = NeedsMetric()
max_requirement(::NeedsMinimal, ::NeedsMetric) = NeedsMetric()

function geometry_requirement(bc::Base.Broadcast.Broadcasted)
    req = geometry_requirement(bc.f)
    for arg in bc.args
        req = max_requirement(req, geometry_requirement(arg))
    end
    return req
end
