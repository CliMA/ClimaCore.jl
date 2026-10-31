"""
    Utilities.Cache

Module that maintains ClimaCore's internal cache of topology and grid objects.
When a constructor is invoked again with the same arguments (e.g., when reading
from a file), the cached object is returned (memoization). This has two
advantages:

 1. Topology and metric information is reused, reducing memory usage.
 2. Two fields can be checked to live on the same grid by comparing the
    underlying grid objects with `===`, rather than comparing all of their
    fields with `==`.

Objects in the cache are never garbage collected, so the module provides
[`clean_cache!`](@ref) to remove them.
"""
module Cache


# Key of an object in OBJECT_CACHE: a tag, the object that the cached object is built from
# (compared with isequal), and optionally a type and an integer. The hash is computed from the
# concrete types of the fields when the key is constructed, so that a lookup never allocates,
# as long as the owner is already on the heap (like a grid): a lookup only compares the owners
# of keys that have the same hash, by dynamic dispatch on values that need no boxing.
struct ObjectKey
    tag::Symbol
    owner::Any
    type::Type
    index::Int
    hash::UInt
    ObjectKey(tag, owner, type = Nothing, index = 0) =
        new(tag, owner, type, index, hash((tag, owner, type, index)))
end

Base.hash(key::ObjectKey, h::UInt) = hash(key.hash, h)

Base.isequal(key1::ObjectKey, key2::ObjectKey) =
    key1.hash == key2.hash && key1.tag === key2.tag && key1.type === key2.type &&
    key1.index == key2.index && isequal(key1.owner, key2.owner)

const OBJECT_CACHE = Dict{ObjectKey, Any}()

"""
    cached_objects()

Return a vector of all currently cached objects, without duplicates.
"""
function cached_objects()
    unique(values(OBJECT_CACHE))
end


"""
    clean_cache!(object)

Remove `object` from the cache of created objects, along with the objects that
were cached for it (e.g., the buffers of operators on a grid), and return
`nothing`.

This function only needs to be called when constructing many grid objects, for
example during a sweep over grid parameters.
"""
function clean_cache!(object)
    filter!(OBJECT_CACHE) do (cache_key, cache_obj)
        cache_obj !== object && cache_key.owner !== object
    end
    return nothing
end

"""
    clean_cache!()

Remove all objects from the cache of created objects and return `nothing`.

This function only needs to be called when constructing many grid objects, for
example during a sweep over grid parameters.
"""
function clean_cache!()
    empty!(OBJECT_CACHE)
    return nothing
end

end # module
