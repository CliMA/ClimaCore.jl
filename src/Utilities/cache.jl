"""
# `Utilities.Cache`

ClimaCore maintains an internal cache of topology and grid objects: this ensures
that if the constructor with the same arguments is invoked again (e.g. by
reading from a file), the cached object will be returned (also known as
_memoization_). This has two main advantages:

1. topology and metric information can be reused, reducing memory usage.

2. it is easy to check if two fields live on the same grid: we can just check if
   the underlying grid objects are the same (`===`), rather than checking all
   the fields are equal (via `==`).

However this means that objects in the cache will not be removed from the
garbage collector, so we provide an interface to remove these.
"""
module Cache


const OBJECT_CACHE = Dict()

"""
    Utilities.Cache.cached_objects()

List all currently cached objects.
"""
function cached_objects()
    unique(values(OBJECT_CACHE))
end


"""
    Utilities.Cache.clean_cache!(object)

Remove `object` from the cache of created objects. 

In most cases, this function should not need to be called, unless you are
constructing many grid objects, for example when doing a sweep over grid
paramaters.
"""
function clean_cache!(object)
    filter!(OBJECT_CACHE) do (cache_key, cache_obj)
        cache_obj !== object
    end
    return nothing
end

"""
    Utilities.Cache.clean_cache!()

Remove all objects from the cache of created objects. 

In most cases, this function should not need to be called, unless you are
constructing many grid objects, for example when doing a sweep over grid
paramaters.
"""
function clean_cache!()
    empty!(OBJECT_CACHE)
    return nothing
end

end # module
