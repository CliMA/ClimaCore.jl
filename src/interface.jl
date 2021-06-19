

"""
    slab(data::AbstractData, h::Integer)

A "pancake" view into an underlying
data layout `data` at location `h`.
"""
function slab end


# TODO: this could cause problems when it fails...
#slab(x, inds...) = x
#slab(tup::Tuple, inds...) = map(x -> slab(x, inds...), tup)

function column end
