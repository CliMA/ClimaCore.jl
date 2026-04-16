const FieldNamePair = Tuple{FieldName, FieldName}

"""
    FieldNameSet{T}(values, [name_tree])

An `AbstractSet` that contains values of type `T`, which serves as an analogue
of a `KeySet` for a `FieldNameDict`. There are two subtypes of `FieldNameSet`:
- `FieldVectorKeys`, for which `T` is set to `FieldName`
- `FieldMatrixKeys`, for which `T` is set to `Tuple{FieldName, FieldName}`; each
  tuple of type `T` represents a pair of row-column indices

Since `FieldName`s are singleton types, the result of almost any `FieldNameSet`
operation can be inferred during compilation. So, with the exception of `map`,
`foreach`, and `set_string`, functions of `FieldNameSet`s do not have any
performance cost at runtime (as long as their arguments are inferrable).

Unlike other `AbstractSet`s, `FieldNameSet` has special behavior for overlapping
values. For example, the `FieldName`s `@name(a.b)` and `@name(a.b.c)` overlap,
so any set operation needs to first decompose `@name(a.b)` into its child values
before combining it with `@name(a.b.c)`. In order to support this (and also to
support the ability to compute set complements), `FieldNameSet` stores a
`FieldNameTree` `name_tree`, which it uses to infer child values. If `name_tree`
is not specified, it gets set to `nothing` by default, which causes some
`FieldNameSet` operations to become disabled. For binary operations like `union`
or `setdiff`, only one set needs to specify a `name_tree`; if two sets both
specify a `name_tree`, the `name_tree`s must be identical.
"""
struct FieldNameSet{
    T <: Union{FieldName, FieldNamePair},
    V <: NTuple{<:Any, T},
    N <: Union{FieldNameTree, Nothing},
} <: AbstractSet{T}
    values::V
    name_tree::N

    # This needs to be an inner constructor to prevent Julia from automatically
    # generating a constructor that fails Aqua.detect_unbound_args_recursively.
    function FieldNameSet{T}(
        values::NTuple{<:Any, T},
        name_tree::Union{FieldNameTree, Nothing} = nothing,
    ) where {T}
        unrolled_foreach(values) do value
            (isnothing(name_tree) || is_valid_value(value, name_tree)) || error(
                "Invalid FieldNameSet value: $value is incompatible with the \
                 FieldNameTree",
            )
            n_duplicate_values = length(unrolled_filter(isequal(value), values))
            n_duplicate_values == 1 || error(
                "Duplicate FieldNameSet values: $n_duplicate_values copies of \
                $value have been passed to a FieldNameSet constructor",
            )
            overlapping_values = unrolled_filter(values) do value′
                value′ != value && is_overlapping_value(value, value′)
            end
            isempty(overlapping_values) || error(
                "Overlapping FieldNameSet values: $value cannot be in the same \
                FieldNameSet as $(values_string(overlapping_values))",
            )
        end
        return new{T, typeof(values), typeof(name_tree)}(values, name_tree)
    end
end

const FieldVectorKeys = FieldNameSet{FieldName}
const FieldMatrixKeys = FieldNameSet{FieldNamePair}

# Do not print the FieldNameTree, since the current implementation ensures that
# it will be the same across all FieldNameSets that are used together.
function Base.show(io::IO, set::FieldNameSet)
    T = eltype(set)
    name_tree_string = isnothing(set.name_tree) ? "" : "; <FieldNameTree>"
    print(io, "$(FieldNameSet{T})($(join(set.values, ", "))$name_tree_string)")
end

Base.length(set::FieldNameSet) = length(set.values)

Base.iterate(set::FieldNameSet, index = 1) = iterate(set.values, index)

Base.map(f::F, set::FieldNameSet) where {F} = unrolled_map(f, set.values)

Base.foreach(f::F, set::FieldNameSet) where {F} =
    unrolled_foreach(f, set.values)

Base.in(value, set::FieldNameSet) =
    is_value_in_set(value, set.values, set.name_tree)

Base.:(==)(set1::FieldNameSet, set2::FieldNameSet) =
    unrolled_all(value -> unrolled_in(value, set2.values), set1.values) &&
    unrolled_all(value -> unrolled_in(value, set1.values), set2.values)

function Base.issubset(set1::FieldNameSet, set2::FieldNameSet)
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    unrolled_all(set1.values) do value
        is_value_in_set(value, set2.values, name_tree)
    end
end

function Base.union(set1::FieldNameSet, set2::FieldNameSet)
    T = combine_eltypes(eltype(set1), eltype(set2))
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    result_values = union_values(set1.values, set2.values, name_tree)
    return FieldNameSet{T}(result_values, name_tree)
end

function Base.intersect(set1::FieldNameSet, set2::FieldNameSet)
    T = combine_eltypes(eltype(set1), eltype(set2))
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    all_values = union_values(set1.values, set2.values, name_tree)
    result_values = unrolled_filter(all_values) do value
        is_value_in_set(value, set1.values, name_tree) &&
            is_value_in_set(value, set2.values, name_tree)
    end
    return FieldNameSet{T}(result_values, name_tree)
end

function Base.setdiff(set1::FieldNameSet, set2::FieldNameSet)
    T = combine_eltypes(eltype(set1), eltype(set2))
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    all_values = union_values(set1.values, set2.values, name_tree)
    result_values = unrolled_filter(all_values) do value
        !is_value_in_set(value, set2.values, name_tree)
    end
    return FieldNameSet{T}(result_values, name_tree)
end

replace_name_tree(set::FieldNameSet, name_tree) =
    FieldNameSet{eltype(set)}(set.values, name_tree)

set_string(set) = values_string(set.values)

set_complement(set) = setdiff(universal_set(eltype(set), set.name_tree), set)

is_subset_that_covers_set(set1, set2) =
    issubset(set1, set2) && isempty(setdiff(set2, set1))

function corresponding_matrix_keys(set::FieldVectorKeys)
    result_values = unrolled_map(name -> (name, name), set.values)
    return FieldMatrixKeys(result_values, set.name_tree)
end

function cartesian_product(row_set::FieldVectorKeys, col_set::FieldVectorKeys)
    name_tree = combine_name_trees(row_set.name_tree, col_set.name_tree)
    result_values = unrolled_product(row_set.values, col_set.values)
    return FieldMatrixKeys(result_values, name_tree)
end

function corresponding_vector_keys(set::FieldMatrixKeys, ::Val{N}) where {N}
    result_values′ = unrolled_map(name_pair -> name_pair[N], set.values)
    result_values =
        unique_and_non_overlapping_values(result_values′, set.name_tree)
    return FieldVectorKeys(result_values, set.name_tree)
end

matrix_row_keys(set::FieldMatrixKeys) = corresponding_vector_keys(set, Val(1))
matrix_col_keys(set::FieldMatrixKeys) = corresponding_vector_keys(set, Val(2))

function matrix_off_diagonal_keys(set::FieldMatrixKeys)
    result_values =
        unrolled_filter(name_pair -> name_pair[1] != name_pair[2], set.values)
    return FieldMatrixKeys(result_values, set.name_tree)
end

function matrix_diagonal_keys(set::FieldMatrixKeys)
    result_values′ = unrolled_filter(set.values) do name_pair
        is_overlapping_name(name_pair[1], name_pair[2])
    end
    result_values = unrolled_map(result_values′) do name_pair
        if name_pair[1] == name_pair[2]
            name_pair
        elseif is_child_value(name_pair[1], name_pair[2])
            (name_pair[1], name_pair[1])
        else
            (name_pair[2], name_pair[2])
        end
    end
    return FieldMatrixKeys(result_values, set.name_tree)
end

function matrix_inferred_diagonal_keys(set::FieldMatrixKeys)
    row_keys = matrix_row_keys(set)
    col_keys = matrix_col_keys(set)
    diag_keys = matrix_row_keys(matrix_diagonal_keys(set))
    all_keys =
        issubset(row_keys, diag_keys) && issubset(col_keys, diag_keys) ?
        diag_keys : union(row_keys, col_keys) # only compute the union if needed
    return corresponding_matrix_keys(all_keys)
end

#=
There are four cases that we need to support in order to be compatible with
generic data types:
1. (_, name) * name or
   (_, name) * (name, _)
2. (_, name_child) * name      -> (_, name_child) * name_child or
   (_, name_child) * (name, _) -> (_, name_child) * (name_child, _)
   We are able to support this by extracting internal rows from FieldNameDict
   entries. We can only extract an internal row from a ColumnwiseBandMatrixField
   whose values contain internal values that correspond to "name_child".
3. (name, name) * name_child      -> (name_child, name_child) * name_child or
   (name, name) * (name_child, _) -> (name_child, name_child) * (name_child, _)
   We are able to support this by extracting internal diagonal blocks from
   FieldNameDict entries. We can only extract an internal diagonal block from a
   ScalingFieldMatrixEntry or a ColumnwiseBandMatrixField of SingleValues.
4. (name1, name1) * name2      -> (name_child, name_child) * name_child or
   (name1, name1) * (name2, _) -> (name_child, name_child) * (name_child, _)
   This is a combination of cases 2 and 3, where "name_child" is a child name of
   both "name1" and "name2".
We only need to support diagonal matrix blocks of scalar values in cases 3 and 4
because we cannot extract internal columns from FieldNameDict entries.
=#
function matrix_product_keys(set1::FieldMatrixKeys, set2::FieldNameSet)
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    result_values′ = unrolled_flatmap(set1.values) do name_pair1
        overlapping_set2_values = unrolled_filter(set2.values) do value2
            row_name2 = eltype(set2) <: FieldName ? value2 : value2[1]
            is_overlapping_name(name_pair1[2], row_name2)
        end
        unrolled_map(overlapping_set2_values) do value2
            row_name2 = eltype(set2) <: FieldName ? value2 : value2[1]
            if is_child_name(name_pair1[2], row_name2)
                # multiplication case 1 or 2
                eltype(set2) <: FieldName ? name_pair1[1] :
                (name_pair1[1], value2[2])
            elseif name_pair1[1] == name_pair1[2]
                # multiplication case 3
                value2
            else
                error("Cannot extract internal column from an off-diagonal key")
            end
        end
    end
    # Removing the overlaps here can trigger multiplication case 4.
    result_values = unique_and_non_overlapping_values(result_values′, name_tree)
    return FieldNameSet{eltype(set2)}(result_values, name_tree)
end
function summand_names_for_matrix_product(
    product_key,
    set1::FieldMatrixKeys,
    set2::FieldNameSet,
)
    product_row_name = eltype(set2) <: FieldName ? product_key : product_key[1]
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    overlapping_set1_values = unrolled_filter(set1.values) do name_pair1
        is_overlapping_name(product_row_name, name_pair1[1])
    end
    result_values = unrolled_flatmap(overlapping_set1_values) do name_pair1
        overlapping_set2_values = unrolled_filter(set2.values) do value2
            row_name2 = eltype(set2) <: FieldName ? value2 : value2[1]
            is_overlapping_name(name_pair1[2], row_name2) &&
                (
                    eltype(set2) <: FieldName ||
                    is_overlapping_name(product_key[2], value2[2])
                ) &&
                (
                    is_child_name(name_pair1[2], row_name2) ||
                    product_row_name == row_name2 &&
                    name_pair1[1] == name_pair1[2]
                )
        end
        unrolled_map(overlapping_set2_values) do value2
            row_name2 = eltype(set2) <: FieldName ? value2 : value2[1]
            is_child_name(product_row_name, name_pair1[1]) && (
                eltype(set2) <: FieldName || product_key[2] == value2[2]
            ) || error("Invalid matrix product key $product_key")
            if is_child_name(name_pair1[2], row_name2)
                if product_row_name == name_pair1[1]
                    # multiplication case 1 or 2
                    name_pair1[2]
                elseif name_pair1[1] == name_pair1[2]
                    # multiplication case 4
                    product_row_name
                else
                    # multiplication case 1 or 2
                    name_pair1[2]
                end
            else
                # multiplication case 3
                row_name2
            end
        end
    end
    return FieldVectorKeys(result_values, name_tree)
end

################################################################################

# Internal functions:

values_string(values) =
    length(values) == 2 ? join(values, " and ") : join(values, ", ", ", and ")

@noinline combine_eltypes(::T1, ::T2) where {T1, T2} =
    error("Mismatched FieldNameSets: Cannot combine a $T1 with a $T2")

@inline combine_eltypes(::Type{T}, ::Type{T}) where {T} = T

combine_name_trees(::Nothing, ::Nothing) = nothing
combine_name_trees(name_tree1, ::Nothing) = name_tree1
combine_name_trees(::Nothing, name_tree2) = name_tree2
combine_name_trees(name_tree1, name_tree2) =
    name_tree1 == name_tree2 ? name_tree1 :
    error("Mismatched FieldNameTrees: The ability to combine different \
           FieldNameTrees has not been implemented")

function universal_set(::Type{FieldName}, name_tree)
    isnothing(name_tree) && error(
        "Missing FieldNameTree: Cannot compute complement of FieldNameSet \
         without a FieldNameTree",
    )
    return FieldVectorKeys(child_names(@name(), name_tree), name_tree)
end
function universal_set(::Type{FieldNamePair}, name_tree)
    row_set = universal_set(FieldName, name_tree)
    return cartesian_product(row_set, row_set)
end

is_valid_value(name::FieldName, name_tree) = is_valid_name(name, name_tree)
is_valid_value(name_pair::FieldNamePair, name_tree) =
    is_valid_name(name_pair[1], name_tree) &&
    is_valid_name(name_pair[2], name_tree)

is_child_value(name1::FieldName, name2::FieldName) = is_child_name(name1, name2)
is_child_value(name_pair1::FieldNamePair, name_pair2::FieldNamePair) =
    is_child_name(name_pair1[1], name_pair2[1]) &&
    is_child_name(name_pair1[2], name_pair2[2])

is_overlapping_value(name1::FieldName, name2::FieldName) =
    is_overlapping_name(name1, name2)
is_overlapping_value(name_pair1::FieldNamePair, name_pair2::FieldNamePair) =
    is_overlapping_name(name_pair1[1], name_pair2[1]) &&
    is_overlapping_name(name_pair1[2], name_pair2[2])

is_value_in_set(value, values, name_tree) =
    unrolled_in(value, values) ||
    unrolled_any(value′ -> is_child_value(value, value′), values) &&
    (isnothing(name_tree) ? true : is_valid_value(value, name_tree))

function unique_and_non_overlapping_values(values, name_tree)
    unique_values = unrolled_unique(values)
    overlapping_values, non_overlapping_values =
        unrolled_split(unique_values) do value
            unrolled_any(unique_values) do value′
                value != value′ && is_overlapping_value(value, value′)
            end
        end
    isempty(overlapping_values) && return unique_values
    isnothing(name_tree) &&
        error("Missing FieldNameTree: Cannot eliminate overlaps among \
               $(values_string(overlapping_values)) without a FieldNameTree")
    expanded_overlapping_values = unrolled_flatmap(overlapping_values) do value
        values_overlapping_with_value =
            unrolled_filter(overlapping_values) do value′
                value != value′ && is_overlapping_value(value, value′)
            end
        expand_child_values(value, values_overlapping_with_value, name_tree)
    end
    no_longer_overlapping_values = unique_and_non_overlapping_values(
        expanded_overlapping_values,
        name_tree,
    )
    return (non_overlapping_values..., no_longer_overlapping_values...)
end

# The function union_values(values1, values2, name_tree) gives the same result
# as unique_and_non_overlapping_values((values1..., values2...), name_tree), but
# it is slightly more efficient (and faster to compile) because it makes use of
# the fact that values1 == unique_and_non_overlapping_values(values1, name_tree)
# and values2 == unique_and_non_overlapping_values(values2, name_tree).
function union_values(values1, values2, name_tree)
    unique_values2 =
        unrolled_filter(value2 -> !unrolled_in(value2, values1), values2)
    overlapping_values1, non_overlapping_values1 =
        unrolled_split(values1) do value1
            unrolled_any(unique_values2) do value2
                is_overlapping_value(value1, value2)
            end
        end
    isempty(overlapping_values1) && return (values1..., unique_values2...)
    overlapping_values2, non_overlapping_values2 =
        unrolled_split(unique_values2) do value2
            unrolled_any(values1) do value1
                is_overlapping_value(value1, value2)
            end
        end
    isnothing(name_tree) && error(
        "Missing FieldNameTree: Cannot eliminate overlaps between \
         $overlapping_values1 and $overlapping_values2 without a FieldNameTree",
    )
    expanded_overlapping_values1 =
        unrolled_flatmap(overlapping_values1) do value1
            values2_overlapping_value1 =
                unrolled_filter(overlapping_values2) do value2
                    is_overlapping_value(value1, value2)
                end
            expand_child_values(value1, values2_overlapping_value1, name_tree)
        end
    expanded_overlapping_values2 =
        unrolled_flatmap(overlapping_values2) do value2
            values1_overlapping_value2 =
                unrolled_filter(overlapping_values1) do value1
                    is_overlapping_value(value1, value2)
                end
            expand_child_values(value2, values1_overlapping_value2, name_tree)
        end
    union_of_overlapping_values = union_values(
        expanded_overlapping_values1,
        expanded_overlapping_values2,
        name_tree,
    )
    return (
        non_overlapping_values1...,
        non_overlapping_values2...,
        union_of_overlapping_values...,
    )
end

expand_child_values(name::FieldName, overlapping_names, name_tree) =
    unrolled_all(overlapping_names) do name′
        name′ != name && is_child_name(name′, name)
    end ? child_names(name, name_tree) : (name,)
function expand_child_values(
    name_pair::FieldNamePair,
    overlapping_name_pairs,
    name_tree,
)
    row_name, col_name = name_pair
    row_name_children_needed =
        unrolled_all(overlapping_name_pairs) do name_pair′
            name_pair′[1] != row_name && is_child_name(name_pair′[1], row_name)
        end
    col_name_children_needed =
        unrolled_all(overlapping_name_pairs) do name_pair′
            name_pair′[2] != col_name && is_child_name(name_pair′[2], col_name)
        end
    row_name_children =
        row_name_children_needed ? child_names(row_name, name_tree) : ()
    col_name_children =
        col_name_children_needed ? child_names(col_name, name_tree) : ()
    # Note: We need special cases for when either row_name or col_name only has
    # one child name, since automatically expanding that name can generate
    # results with unnecessary expansions. For example, it can lead to a
    # situation in which issubset(set1, set2) && union(set1, set2) != set2
    # evaluates to true because union(set1, set2) has too many expanded values.
    return if length(row_name_children) > 1 && length(col_name_children) > 1 ||
              length(row_name_children) == 1 && length(col_name_children) == 1
        unrolled_product(row_name_children, col_name_children)
    elseif length(row_name_children) > 1 && length(col_name_children) == 1 ||
           length(row_name_children) > 0 && length(col_name_children) == 0
        unrolled_product(row_name_children, (col_name,))
    elseif length(row_name_children) == 1 && length(col_name_children) > 1 ||
           length(row_name_children) == 0 && length(col_name_children) > 0
        unrolled_product((row_name,), col_name_children)
    else # length(row_name_children) == 0 && length(col_name_children) == 0
        (name_pair,)
    end
end

# This is required for type-stability as of Julia 1.9.
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(unique_and_non_overlapping_values)
        m.recursion_relation = dont_limit
    end
    for m in methods(union_values)
        m.recursion_relation = dont_limit
    end
end
