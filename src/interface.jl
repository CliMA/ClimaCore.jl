# Toplevel interface functions for recurisve broadcast expressions
import ..Utilities.Unrolled: unrolled_map_with_inbounds

"""
    level(data, v)

Return a horizontal view of `data` at level `v`, spanning all elements in that level.
"""
function level end

"""
    slab(data, v, h)
    slab(data, h)

Return a horizontal view of `data` at level `v` and horizontal element `h`. If `v` is
omitted, it is assumed to be 1.
"""
function slab end

"""
    column(data, i, j, h)
    column(data, i, h)

Return a vertical view of `data` at nodal point index `(i, j)` of horizontal element
`h`. If `j` is omitted, it is assumed to be 1.
"""
function column end

for op in (:level, :slab, :column)
    @eval $op(n::Number, inds...) = n
    @eval $op(::Nothing, inds...) = nothing
    @eval Base.@propagate_inbounds $op(t::Tuple, inds...) =
        unrolled_map_with_inbounds(t) do x
            Base.@_propagate_inbounds_meta
            $op(x, inds...)
        end
    @eval Base.@propagate_inbounds $op(nt::NamedTuple, inds...) =
        NamedTuple{keys(nt)}($op(values(nt), inds...))
end
