
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
@inline column(tup::Tuple, inds...) = column_args(tup, inds...)


function column_args end

# See Base.Broadcast.preprocess_args
@inline column_args(args::Tuple, inds...) = (column(args[1], inds...), column_args(Base.tail(args), inds...)...)
@inline column_args(args::Tuple{Any}, inds...) = (column(args[1], inds...),)
@inline column_args(args::Tuple{}, inds...) = ()

if VERSION >= v"1.7.0-beta1"
# FIXME(vchuravy/aviatesk): This should not be necessary
# we know the recursion is assured to terminate, tell it to the compiler
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(column_args)
        m.recursion_relation = dont_limit
    end
end
end
