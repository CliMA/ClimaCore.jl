# Field iterator

import StaticArrays

"""
    property_chains

An array of "property chains", used to recursively
`getproperty` until a single scalar field is reached.

A property chain may be, for example
`(:model, :submodel, :temperature)`
where
`model.submodel.temperature` is a scalar field.
"""
function property_chains(f::Union{Field, FieldVector})
    prop_chains = []
    flattened_property_chains!(prop_chains, f)
    return prop_chains
end

function flattened_property_chains!(prop_chains, f::FieldVector, pc = ())
    for pn in propertynames(f)
        p = getproperty(f, pn)
        flattened_property_chains!(prop_chains, p, (pc..., pn))
    end
end

function flattened_property_chains!(prop_chains, f::Vector, pc = ())
    push!(prop_chains, pc) # Perhaps fieldvector contains a Vector
end
function flattened_property_chains!(prop_chains, f::Real, pc = ())
    push!(prop_chains, pc) # Perhaps fieldvector contains a Real
end
function flattened_property_chains!(prop_chains, f::Field, pc = ())
    if isempty(propertynames(f)) # single scalar field
        push!(prop_chains, pc)
    else
        for pn in propertynames(f)
            p = getproperty(f, pn)
            flattened_property_chains!(prop_chains, p, (pc..., pn))
        end
    end
end

function isa_12_covariant_field(
    f::Type{CF},
) where {
    FT,
    T,
    CF <: Geometry.AxisTensor{
        FT,
        1,
        Tuple{Geometry.CovariantAxis{(1, 2)}},
        StaticArrays.SVector{2, FT},
    },
}
    return true
end
isa_12_covariant_field(f) = false
function isa_3_covariant_field(
    f::Type{CF},
) where {
    FT,
    T,
    CF <: Geometry.AxisTensor{
        FT,
        1,
        Tuple{Geometry.CovariantAxis{(3,)}},
        StaticArrays.SVector{1, FT},
    },
}
    return true
end
isa_3_covariant_field(f) = false

transform_field(x::FieldVector) = x
function transform_field(x)
    if isa_12_covariant_field(eltype(x))
        return Geometry.UVVector.(x)
    elseif isa_3_covariant_field(eltype(x))
        return Geometry.WVector.(x)
    else
        return x
    end
end

function single_field(
    f::Union{Field, FieldVector},
    prop_chain,
    transform = transform_field,
)
    var = f
    for pn in prop_chain
        var = getproperty(var, pn)
        var = transform(var)
    end
    return var
end

struct FieldIterator{N, F, PCS}
    f::F
    prop_chains::PCS
end

"""
    field_iterator(::Union{Field, FieldVector})

Returns an iterable field, that recursively calls
`getproperty` for all of the `propertynames`, and
returns a `Tuple` of
 - the individual scalar field and
 - the tuple chain used to reach the scalar field
"""
function field_iterator(f::Union{Field, FieldVector})
    prop_chains = property_chains(f)
    N = length(prop_chains)
    F = typeof(f)
    PCS = typeof(prop_chains)
    FieldIterator{N, F, PCS}(f, prop_chains)
end

Base.length(::FieldIterator{N}) where {N} = N

function Base.iterate(iter::FieldIterator, state = 1)
    state > length(iter) && return nothing
    f = iter.f
    pc = iter.prop_chains[state]
    return ((single_field(f, pc), pc), state + 1)
end
