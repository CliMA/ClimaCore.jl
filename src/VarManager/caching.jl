"""
    VarCachingStrategy

Abstract type for caching strategies that determine which computed variables
should be materialized to global memory vs kept as lazy broadcasts.
"""
abstract type VarCachingStrategy end

"""
    add_to_global_cache(strategy::VarCachingStrategy, result) -> Bool

Determines whether `result` (the output of `compute_var` or `compute_tendency`)
should be materialized and stored in the global cache.

Returns `true` if the result should be cached, `false` otherwise.
"""
function add_to_global_cache end

"""
    EagerGlobalCaching <: VarCachingStrategy

A caching strategy that materializes all lazy broadcast expressions to global
memory. This mimics the current behavior of ClimaAtmos where all precomputed
quantities are cached.

Fields that are already materialized (not lazy broadcasts) are not re-cached
since they already exist in global memory.
"""
struct EagerGlobalCaching <: VarCachingStrategy end

add_to_global_cache(::EagerGlobalCaching, ::Base.AbstractBroadcasted) = true
add_to_global_cache(::EagerGlobalCaching, ::LazyBroadcast.LazyBroadcasted) = true
add_to_global_cache(::EagerGlobalCaching, ::Field) = true
add_to_global_cache(::EagerGlobalCaching, _) = false

"""
    NoCaching <: VarCachingStrategy

A caching strategy that never caches computed results. All lazy broadcasts
remain lazy until they are needed. This is useful for debugging or when
memory is constrained.

Note: This strategy may result in redundant computations if the same
variable is used multiple times.
"""
struct NoCaching <: VarCachingStrategy end

add_to_global_cache(::NoCaching, _) = false

#####
##### VarCache
#####

"""
    VarCache

Stores computed variables and tendencies during graph evaluation.

# Fields
- `vars::Dict{FieldName, Union{Field, Base.AbstractBroadcasted, LazyBroadcast.LazyBroadcasted}}`: Maps variable names to their computed values
- `cache_fields::Dict{FieldName, Field}`: Pre-allocated cache fields for materialization
"""
struct VarCache
    vars::Dict{FieldName, Union{Field, Base.AbstractBroadcasted, LazyBroadcast.LazyBroadcasted}}
    cache_fields::Dict{FieldName, Field}
end

VarCache() = VarCache(Dict(), Dict())

Base.getindex(cache::VarCache, name::FieldName) = cache.vars[name]
Base.setindex!(cache::VarCache, value, name::FieldName) = cache.vars[name] = value
Base.haskey(cache::VarCache, name::FieldName) = haskey(cache.vars, name)

#####
##### VarCacheView - property-style access (vars.c.ρ instead of vars[@name(c.ρ)])
#####

struct VarCacheView
    cache::VarCache
    s1::Symbol
end

@inline function Base.getproperty(cache::VarCache, sym::Symbol)
    if sym === :vars || sym === :cache_fields
        return getfield(cache, sym)
    end
    return VarCacheView(cache, sym)
end

# vars.f.u₃ → single Dict lookup, no recursion, no tuple allocation
@inline function Base.getproperty(v::VarCacheView, sym::Symbol)
    name = FieldName(getfield(v, :s1), sym)
    return getfield(getfield(v, :cache), :vars)[name]
end

#####
##### Cache allocation and materialization
#####

"""
    allocate_cache_field(result, space)

Allocates a Field to store a cached result. The Field is allocated based on
the element type of the result and the provided space.
"""
function allocate_cache_field(result::Base.AbstractBroadcasted, space)
    # Get the element type from the broadcasted expression
    T = Base.Broadcast.combine_eltypes(result.f, result.args)
    return Field(T, space)
end

allocate_cache_field(result::LazyBroadcast.LazyBroadcasted, space) =
    allocate_cache_field(result.value, space)

allocate_cache_field(result::Field, _) = similar(result)

"""
    materialize_to_cache!(cache_field, result)

Materializes a lazy broadcast result into a pre-allocated cache field.
"""
function materialize_to_cache!(cache_field::Field, result::Base.AbstractBroadcasted)
    Base.Broadcast.materialize!(cache_field, result)
    return cache_field
end

function materialize_to_cache!(cache_field::Field, result::LazyBroadcast.LazyBroadcasted)
    materialize_to_cache!(cache_field, result.value)
    return cache_field
end

function materialize_to_cache!(cache_field::Field, result::Field)
    cache_field .= result
    return cache_field
end

#####
##### Store result helpers
#####

"""
    _extract_space(bc::Base.AbstractBroadcasted)

Extracts the output space from a broadcast. Uses axes(bc) when available (correct
for space-changing ops like ᶠinterp). For nested broadcasts, uses axes(arg) to get
the output space of each arg, not the first Field (which would be an input).
"""
function _extract_space(bc::Base.AbstractBroadcasted)
    # 1. Try the broadcast's own output axes
    space = _axes_to_space(bc)
    !isnothing(space) && return space
    # 2. For each arg: prefer axes(arg) for broadcasts (output), else Field axes
    for arg in bc.args
        if arg isa Field
            return axes(arg)
        elseif arg isa Base.AbstractBroadcasted
            space = _axes_to_space(arg)
            if !isnothing(space)
                return space
            end
            space = _extract_space(arg)
            if !isnothing(space)
                return space
            end
        elseif arg isa LazyBroadcast.LazyBroadcasted
            space = _extract_space(arg)
            if !isnothing(space)
                return space
            end
        end
    end
    return nothing
end

function _axes_to_space(bc)
    axs = axes(bc)
    # ClimaCore Fields and OperatorBroadcasted return space directly; Base uses Tuple
    axs isa Spaces.AbstractSpace && return axs
    axs isa Tuple && !isempty(axs) && first(axs) isa Spaces.AbstractSpace && return first(axs)

    return nothing
end

_extract_space(field::Field) = axes(field)
_extract_space(x::LazyBroadcast.LazyBroadcasted) = _extract_space(x.value)
_extract_space(_) = nothing

"""
    _store_result!(cache, name, result, caching_strategy)

Stores a computed result in the cache, potentially materializing it
based on the caching strategy.
"""
function _store_result!(cache::VarCache, name::FieldName, result, strategy::VarCachingStrategy)
    if add_to_global_cache(strategy, result)
        # Need to materialize the lazy broadcast
        if !haskey(cache.cache_fields, name)
            # Allocate a new cache field
            # We need to get the space from one of the fields in the broadcast
            space = _extract_space(result)
            if isnothing(space)
                error("""
                    Cannot cache result for $name: unable to determine space.
                    The broadcast expression must contain at least one Field to infer the space.
                    Result type: $(typeof(result))
                    """)
            end
            cache.cache_fields[name] = allocate_cache_field(result, space)
        end
        cache[name] = materialize_to_cache!(cache.cache_fields[name], result)
    else
        # Store as-is (either already a Field or keep lazy)
        cache[name] = result
    end
end
