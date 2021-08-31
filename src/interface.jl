
"""
    slab(data::AbstractData, h::Integer)

A "pancake" view into an underlying
data layout `data` at location `h`.
"""
function slab end
#=
# TODO: this could cause problems when it fails...
#slab(x::Number, inds...) = x
=#
slab(x, inds...) = x
slab(tup::Tuple, inds...) = map(x -> slab(x, inds...), tup)

"""
    column(data::AbstractData, i::Integer)

A contiguous "column" view into an underlying
data layout `data` at nodal point index `i`.
"""
function column end

column(x, inds...) = x
column(tup::Tuple, inds...) = map(x -> column(x, inds...), tup)
