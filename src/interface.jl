# Toplevel interface functions for recurisve broadcast expressions
import ..Utilities.Unrolled: unrolled_map_with_inbounds

"""
    slab(data::AbstractData, h::Integer)

A "pancake" view into an underlying
data layout `data` at location `h`.
"""
function slab end

# generic fallback
Base.@propagate_inbounds slab(x, inds...) = x
Base.@propagate_inbounds slab(tup::Tuple, inds...) = slab_args(tup, inds...)

Base.@propagate_inbounds slab_args(args::Tuple, inds...) =
    unrolled_map_with_inbounds(args) do arg
        Base.@_propagate_inbounds_meta
        slab(arg, inds...)
    end
Base.@propagate_inbounds slab_args(args::NamedTuple, inds...) =
    NamedTuple{keys(args)}(slab_args(values(args), inds...))

"""
    column(data::AbstractData, i::Integer)

A contiguous "column" view into an underlying
data layout `data` at nodal point index `i`.
"""
function column end

# generic fallback
Base.@propagate_inbounds column(x, inds...) = x
Base.@propagate_inbounds column(tup::Tuple, inds...) = column_args(tup, inds...)

Base.@propagate_inbounds column_args(args::Tuple, inds...) =
    unrolled_map_with_inbounds(args) do arg
        Base.@_propagate_inbounds_meta
        column(arg, inds...)
    end
Base.@propagate_inbounds column_args(args::NamedTuple, inds...) =
    NamedTuple{keys(args)}(column_args(values(args), inds...))

function level end

Base.@propagate_inbounds level(x, inds...) = x
Base.@propagate_inbounds level_args(args::Tuple, inds...) =
    unrolled_map_with_inbounds(args) do arg
        Base.@_propagate_inbounds_meta
        level(arg, inds...)
    end
