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


const OBJECT_CACHE = Dict()

"""
    cached_objects()

Return a vector of all currently cached objects, without duplicates.
"""
function cached_objects()
    unique(values(OBJECT_CACHE))
end


"""
    clean_cache!(object)

Remove `object` from the cache of created objects and return `nothing`.

This function only needs to be called when constructing many grid objects, for
example during a sweep over grid parameters.
"""
function clean_cache!(object)
    filter!(OBJECT_CACHE) do (cache_key, cache_obj)
        cache_obj !== object
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
