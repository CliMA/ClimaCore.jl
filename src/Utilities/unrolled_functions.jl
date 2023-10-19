"""
    UnrolledFunctions

A collection of generated functions that get unrolled during compilation, which
make it possible to iterate over nonuniform collections without sacrificing
type-stability.

The functions exported by this module are
- `unrolled_map(f, values, [values2])`: alternative to `map`
- `unrolled_any(f, values)`: alternative to `any`
- `unrolled_all(f, values)`: alternative to `all`
- `unrolled_filter(f, values)`: alternative to `filter`
- `unrolled_foreach(f, values)`: alternative to `foreach`
- `unrolled_in(value, values)`: alternative to `in`
- `unrolled_unique(values)`: alternative to `unique`
- `unrolled_flatten(values)`: alternative to `Iterators.flatten`
- `unrolled_flatmap(f, values)`: alternative to `Iterators.flatmap`
- `unrolled_product(values1, values2)`: alternative to `Iterators.product`
- `unrolled_findonly(f, values)`: checks that only one value satisfies `f`, and
  then returns that value
- `unrolled_split(f, values)`: returns a tuple that contains the result of
  calling `unrolled_filter` with `f` and the result of calling it with `!f`
- `unrolled_take(values, ::Val{N})`: alternative to `Iterators.take`, but with
  an `Int` wrapped in a `Val` as the second argument instead of a regular `Int`;
  this usually compiles more quickly than `values[1:N]`
- `unrolled_drop(values, ::Val{N})`: alternative to `Iterators.drop`, but with
  an `Int` wrapped in a `Val` as the second argument instead of a regular `Int`;
  this usually compiles more quickly than `values[(end - N + 1):end]`
"""
module UnrolledFunctions

import Unrolled
import Unrolled: @unroll

export unrolled_map,
    unrolled_any,
    unrolled_all,
    unrolled_filter,
    unrolled_foreach,
    unrolled_in,
    unrolled_unique,
    unrolled_flatten,
    unrolled_flatmap,
    unrolled_product,
    unrolled_findonly,
    unrolled_split,
    unrolled_take,
    unrolled_drop

# The definitions of unrolled_map and unrolled_any are copied over from
# Unrolled.jl, but their recursion limits are disabled here. As of Julia 1.9, we
# need to remove their recursion limits so that we can use them to implement
# recursion in other functions without any type-instabilities. For example, if a
# function f needs to map over some values, and if the computation for each
# value recursively calls f, then the map can be implemented using unrolled_map.

@generated unrolled_map(f, values) =
    :(tuple($((:(f(values[$i])) for i in 1:Unrolled.type_length(values))...)))

@generated function unrolled_map(f, values1, values2)
    N = Unrolled.type_length(values1)
    @assert N == Unrolled.type_length(values2)
    :(tuple($((:(f(values1[$i], values2[$i])) for i in 1:N)...)))
end

@unroll function unrolled_any(f, values)
    @unroll for value in values
        f(value) && return true
    end
    return false
end

if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(unrolled_map)
        m.recursion_relation = dont_limit
    end
    for m in methods(unrolled_any)
        m.recursion_relation = dont_limit
    end
end

const unrolled_all = Unrolled.unrolled_all
const unrolled_filter = Unrolled.unrolled_filter
const unrolled_foreach = Unrolled.unrolled_foreach
const unrolled_in = Unrolled.unrolled_in

# Note: Unrolled.unrolled_reduce passes the arguments to its input function in
# reverse order (as of version 0.1 of Unrolled.jl).

unrolled_unique(values) =
    Unrolled.unrolled_reduce((), values) do value, unique_values
        unrolled_in(value, unique_values) ? unique_values :
        (unique_values..., value)
    end

unrolled_flatten(values) =
    Unrolled.unrolled_reduce((tup2, tup1) -> (tup1..., tup2...), (), values)

unrolled_flatmap(f::F, values) where {F} =
    unrolled_flatten(unrolled_map(f, values))

unrolled_product(values1, values2) =
    unrolled_flatmap(values1) do value1
        unrolled_map(value2 -> (value1, value2), values2)
    end

function unrolled_findonly(f::F, values) where {F}
    filtered_values = unrolled_filter(f, values)
    return length(filtered_values) == 1 ? filtered_values[1] :
           error("unrolled_findonly requires that exactly 1 value makes f true")
end

unrolled_split(f::F, values) where {F} =
    (unrolled_filter(f, values), unrolled_filter(!f, values))

unrolled_take(values, ::Val{N}) where {N} = ntuple(i -> values[i], Val(N))
unrolled_drop(values, ::Val{N}) where {N} =
    ntuple(i -> values[N + i], Val(length(values) - N))

end # module
