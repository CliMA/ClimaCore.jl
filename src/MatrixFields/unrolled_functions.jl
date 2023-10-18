@inline unrolled_zip(values1, values2) =
    isempty(values1) || isempty(values2) ? () :
    (
        (first(values1), first(values2)),
        unrolled_zip(Base.tail(values1), Base.tail(values2))...,
    )

@inline unrolled_map(f::F, values) where {F} =
    isempty(values) ? () :
    (f(first(values)), unrolled_map(f, Base.tail(values))...)

unrolled_foldl(f::F, values) where {F} =
    isempty(values) ?
    error("unrolled_foldl requires init for an empty collection of values") :
    _unrolled_foldl(f, first(values), Base.tail(values))
unrolled_foldl(f::F, values, init) where {F} = _unrolled_foldl(f, init, values)
@inline _unrolled_foldl(f::F, result, values) where {F} =
    isempty(values) ? result :
    _unrolled_foldl(f, f(result, first(values)), Base.tail(values))

# The @inline annotations are needed to avoid allocations when there are a lot
# of values.

# Using first and tail instead of [1] and [2:end] restricts us to Tuples, but it
# also results in less compilation time.

# This is required to make the unrolled functions type-stable, as of Julia 1.9.
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(unrolled_zip)
        m.recursion_relation = dont_limit
    end
    for m in methods(unrolled_map)
        m.recursion_relation = dont_limit
    end
    for m in methods(_unrolled_foldl)
        m.recursion_relation = dont_limit
    end
end

################################################################################

unrolled_foreach(f::F, values) where {F} = (unrolled_map(f, values); nothing)

unrolled_any(f::F, values) where {F} =
    unrolled_foldl(|, unrolled_map(f, values), false)

unrolled_all(f::F, values) where {F} =
    unrolled_foldl(&, unrolled_map(f, values), true)

unrolled_filter(f::F, values) where {F} =
    unrolled_foldl(values, ()) do filtered_values, value
        f(value) ? (filtered_values..., value) : filtered_values
    end

unrolled_unique(values) =
    unrolled_foldl(values, ()) do unique_values, value
        unrolled_any(isequal(value), unique_values) ? unique_values :
        (unique_values..., value)
    end

unrolled_flatten(values) =
    unrolled_foldl(values, ()) do flattened_values, value
        (flattened_values..., value...)
    end

# Non-standard functions:

unrolled_mapflatten(f::F, values) where {F} =
    unrolled_flatten(unrolled_map(f, values))

function unrolled_findonly(f::F, values) where {F}
    filtered_values = unrolled_filter(f, values)
    length(filtered_values) == 1 ||
        error("unrolled_findonly requires that exactly one value makes f true")
    return first(filtered_values)
end

# This is required to make functions defined elsewhere type-stable, as of Julia
# 1.9. Specifically, if an unrolled function is used to implement the recursion
# of another function, it needs to have its recursion limit disabled in order
# for that other function to be type-stable.
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(unrolled_any)
        m.recursion_relation = dont_limit
    end # for is_valid_name
    for m in methods(unrolled_mapflatten)
        m.recursion_relation = dont_limit
    end # for complement_values_in_subtree and value_or_non_overlapping_children
end
