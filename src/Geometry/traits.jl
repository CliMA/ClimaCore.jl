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

# Argument-level requirement: non-broadcasted values don't add geometry needs.
geometry_requirement_arg(::Any) = NeedsMinimal()
geometry_requirement_arg(arg::Base.AbstractBroadcasted) = geometry_requirement(arg)

# Basic arithmetic operations only need minimal geometry
# These operations don't involve coordinate transformations
geometry_requirement(::typeof(+)) = NeedsMinimal()
geometry_requirement(::typeof(-)) = NeedsMinimal()
geometry_requirement(::typeof(*)) = NeedsMinimal()
geometry_requirement(::typeof(/)) = NeedsMinimal()
geometry_requirement(::typeof(^)) = NeedsMinimal()

# Common math functions
import Base: abs, sqrt, exp, log, sin, cos, max, min
geometry_requirement(::typeof(abs)) = NeedsMinimal()
geometry_requirement(::typeof(sqrt)) = NeedsMinimal()
geometry_requirement(::typeof(exp)) = NeedsMinimal()
geometry_requirement(::typeof(log)) = NeedsMinimal()
geometry_requirement(::typeof(sin)) = NeedsMinimal()
geometry_requirement(::typeof(cos)) = NeedsMinimal()
geometry_requirement(::typeof(max)) = NeedsMinimal()
geometry_requirement(::typeof(min)) = NeedsMinimal()

# Comparison operators  
geometry_requirement(::typeof(<)) = NeedsMinimal()
geometry_requirement(::typeof(>)) = NeedsMinimal()
geometry_requirement(::typeof(<=)) = NeedsMinimal()
geometry_requirement(::typeof(>=)) = NeedsMinimal()
geometry_requirement(::typeof(==)) = NeedsMinimal()

# RecursiveApply operations (used extensively in broadcasts)
import ..RecursiveApply: rmul, radd, rsub, rdiv
geometry_requirement(::typeof(RecursiveApply.rmul)) = NeedsMinimal()
geometry_requirement(::typeof(RecursiveApply.radd)) = NeedsMinimal()
geometry_requirement(::typeof(RecursiveApply.rsub)) = NeedsMinimal()
geometry_requirement(::typeof(RecursiveApply.rdiv)) = NeedsMinimal()
geometry_requirement(::typeof(RecursiveApply.rzero)) = NeedsMinimal()
geometry_requirement(::typeof(RecursiveApply.rmap)) = NeedsMinimal()

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
        req = max_requirement(req, geometry_requirement_arg(arg))
    end
    return req
end
