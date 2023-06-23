# Toplevel interface functions for recurisve broadcast expressions

"""
    enable_threading()

By default returns `false` signifying threading is disabled.
Enable the threading runtime by redefining the method at the toplevel your experiment file:

    import ClimaCore: enable_threading
    enable_threading() = true

and running julia with `julia --nthreads=N ...`

This function is deprecated in version `0.10.42`.
Please use the ClimaComms context for threading.
"""
enable_threading() = false
Base.@deprecate enable_threading() deprecated_enable_threading() false
deprecated_enable_threading() = false

"""
    slab(data::AbstractData, h::Integer)

A "pancake" view into an underlying
data layout `data` at location `h`.
"""
function slab end

# generic fallback
Base.@propagate_inbounds slab(x, inds...) = x
Base.@propagate_inbounds slab(tup::Tuple, inds...) = slab_args(tup, inds...)

# Recursively call slab() on broadcast arguments in a way that is statically reducible by the optimizer
# see Base.Broadcast.preprocess_args
Base.@propagate_inbounds slab_args(args::Tuple, inds...) =
    (slab(args[1], inds...), slab_args(Base.tail(args), inds...)...)
Base.@propagate_inbounds slab_args(args::Tuple{Any}, inds...) =
    (slab(args[1], inds...),)
Base.@propagate_inbounds slab_args(args::Tuple{}, inds...) = ()

"""
    column(data::AbstractData, i::Integer)

A contiguous "column" view into an underlying
data layout `data` at nodal point index `i`.
"""
function column end

# generic fallback
Base.@propagate_inbounds column(x, inds...) = x
Base.@propagate_inbounds column(tup::Tuple, inds...) = column_args(tup, inds...)

# Recursively call column() on broadcast arguments in a way that is statically reducible by the optimizer
# see Base.Broadcast.preprocess_args
Base.@propagate_inbounds column_args(args::Tuple, inds...) =
    (column(args[1], inds...), column_args(Base.tail(args), inds...)...)
Base.@propagate_inbounds column_args(args::Tuple{Any}, inds...) =
    (column(args[1], inds...),)
Base.@propagate_inbounds column_args(args::Tuple{}, inds...) = ()

function level end
