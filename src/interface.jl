# Toplevel interface functions for recurisve broadcast expressions

"""
    slab(data::AbstractData, h::Integer)

A "pancake" view into an underlying
data layout `data` at location `h`.
"""
function slab end

# generic fallback
@inline slab(x, inds...) = x
@inline slab(tup::Tuple, inds...) = slab_args(tup, inds...)

# Recursively call slab() on broadcast arguments in a way that is statically reducible by the optimizer
# see Base.Broadcast.preprocess_args
@inline slab_args(args::Tuple, inds...) =
    (slab(args[1], inds...), slab_args(Base.tail(args), inds...)...)
@inline slab_args(args::Tuple{Any}, inds...) = (slab(args[1], inds...),)
@inline slab_args(args::Tuple{}, inds...) = ()

"""
    column(data::AbstractData, i::Integer)

A contiguous "column" view into an underlying
data layout `data` at nodal point index `i`.
"""
function column end

# generic fallback
@inline column(x, inds...) = x
@inline column(tup::Tuple, inds...) = column_args(tup, inds...)

# Recursively call column() on broadcast arguments in a way that is statically reducible by the optimizer
# see Base.Broadcast.preprocess_args
@inline column_args(args::Tuple, inds...) =
    (column(args[1], inds...), column_args(Base.tail(args), inds...)...)
@inline column_args(args::Tuple{Any}, inds...) = (column(args[1], inds...),)
@inline column_args(args::Tuple{}, inds...) = ()

function level end
