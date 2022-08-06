# Toplevel interface functions for recurisve broadcast expressions

"""
    enable_threading()

By default returns `false` signifying threading is disabled.
Enable the threading runtime by redefining the method at the toplevel your experiment file:

    import ClimaCore: enable_threading
    enable_threading() = true

and running julia with `julia --nthreads=N ...`
"""
enable_threading() = false

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
@inline function slab_args(args::Tuple, inds...)
    @inbounds s1 = slab(args[1], inds...)
    (s1, slab_args(Base.tail(args), inds...)...)
end
@inline function slab_args(args::Tuple{Any}, inds...)
    @inbounds s1 = slab(args[1], inds...)
    (s1,)
end
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
@inline function column_args(args::Tuple, inds...)
    @inbounds c1 = column(args[1], inds...)
    (c1, column_args(Base.tail(args), inds...)...)
end
@inline function column_args(args::Tuple{Any}, inds...)
    @inbounds c1 = column(args[1], inds...)
    (c1,)
end
@inline column_args(args::Tuple{}, inds...) = ()

function level end
