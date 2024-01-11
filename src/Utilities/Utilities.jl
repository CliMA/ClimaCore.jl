module Utilities

include("plushalf.jl")
include("unrolled_functions.jl")

"""
    Utilities.remove_from_cache!(object)

Remove `object` from the cache of created objects (only topologies and
grids are cached). In most cases, this should not need to be called, unless you
are e.g. doing a sweep over grid paramaters.

Only call this if you are completely finished with this object configuration, and
no other instances of it exist.
"""
function remove_from_cache!(object) # fallback
end


end # module
