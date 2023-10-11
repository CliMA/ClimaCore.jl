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
        check_values(values, name_tree)
        return new{T, typeof(values), typeof(name_tree)}(values, name_tree)
    end
end

const FieldVectorKeys = FieldNameSet{FieldName}
const FieldMatrixKeys = FieldNameSet{FieldNamePair}

function Base.show(io::IO, set::FieldNameSet{T}) where {T}
    type_string(::Type{FieldName}) = "FieldVectorKeys"
    type_string(::Type{FieldNamePair}) = "FieldMatrixKeys"
    type_string(::Type{T}) where {T} = "FieldNameSet{$T}"
    # Do not print the FieldNameTree, since the current implementation ensures
    # that it will be the same across all FieldNameSets that are used together.
    name_tree_str = isnothing(set.name_tree) ? "" : "; <FieldNameTree>"
    print(io, "$(type_string(T))($(join(set.values, ", "))$name_tree_str)")
end

Base.length(set::FieldNameSet) = length(set.values)

Base.iterate(set::FieldNameSet, index = 1) = iterate(set.values, index)

Base.map(f::F, set::FieldNameSet) where {F} = unrolled_map(f, set.values)

Base.foreach(f::F, set::FieldNameSet) where {F} =
    unrolled_foreach(f, set.values)

Base.in(value, set::FieldNameSet) =
    is_value_in_set(value, set.values, set.name_tree)

function Base.issubset(set1::FieldNameSet, set2::FieldNameSet)
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    unrolled_all(set1.values) do value
        is_value_in_set(value, set2.values, name_tree)
    end
end

Base.:(==)(set1::FieldNameSet, set2::FieldNameSet) =
    issubset(set1, set2) && issubset(set2, set1)

function Base.intersect(set1::FieldNameSet{T}, set2::FieldNameSet{T}) where {T}
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    values1′, values2′ = set1.values, set2.values
    values1, values2 = non_overlapping_values(values1′, values2′, name_tree)
    result_values = unrolled_filter(values2) do value
        unrolled_any(isequal(value), values1)
    end
    return FieldNameSet{T}(result_values, name_tree)
end

function Base.union(set1::FieldNameSet{T}, set2::FieldNameSet{T}) where {T}
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    values1′, values2′ = set1.values, set2.values
    values1, values2 = non_overlapping_values(values1′, values2′, name_tree)
    values2_minus_values1 = unrolled_filter(values2) do value
        !unrolled_any(isequal(value), values1)
    end
    result_values = (values1..., values2_minus_values1...)
    return FieldNameSet{T}(result_values, name_tree)
end

function Base.setdiff(set1::FieldNameSet{T}, set2::FieldNameSet{T}) where {T}
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    set2_complement_values = set_complement_values(T, set2.values, name_tree)
    set2_complement = FieldNameSet{T}(set2_complement_values, name_tree)
    return intersect(set1, set2_complement)
end

set_string(set) =
    length(set) == 2 ? join(set.values, " and ") :
    join(set.values, ", ", ", and ")

is_subset_that_covers_set(set1, set2) =
    issubset(set1, set2) && isempty(setdiff(set2, set1))

function set_complement(set::FieldNameSet{T}) where {T}
    result_values = set_complement_values(T, set.values, set.name_tree)
    return FieldNameSet{T}(result_values, set.name_tree)
end

function corresponding_matrix_keys(set::FieldVectorKeys)
    result_values = unrolled_map(name -> (name, name), set.values)
    return FieldMatrixKeys(result_values, set.name_tree)
end

function cartesian_product(set1::FieldVectorKeys, set2::FieldVectorKeys)
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    result_values = unrolled_mapflatten(set1.values) do row_name
        unrolled_map(col_name -> (row_name, col_name), set2.values)
    end
    return FieldMatrixKeys(result_values, name_tree)
end

function matrix_row_keys(set::FieldMatrixKeys)
    result_values′ = unrolled_map(name_pair -> name_pair[1], set.values)
    result_values =
        unique_and_non_overlapping_values(result_values′, set.name_tree)
    return FieldVectorKeys(result_values, set.name_tree)
end

function matrix_off_diagonal_keys(set::FieldMatrixKeys)
    result_values =
        unrolled_filter(name_pair -> name_pair[1] != name_pair[2], set.values)
    return FieldMatrixKeys(result_values, set.name_tree)
end

function matrix_diagonal_keys(set::FieldMatrixKeys)
    result_values′ = unrolled_filter(set.values) do name_pair
        names_are_overlapping(name_pair[1], name_pair[2])
    end
    result_values = unrolled_map(result_values′) do name_pair
        name_pair[1] == name_pair[2] ? name_pair :
        is_child_value(name_pair[1], name_pair[2]) ?
        (name_pair[1], name_pair[1]) : (name_pair[2], name_pair[2])
    end
    return FieldMatrixKeys(result_values, set.name_tree)
end

#=
There are four cases that we need to support in order to be compatible with
RecursiveApply (i.e., with rmul):
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
   LinearAlgebra.UniformScaling or a ColumnwiseBandMatrixField of SingleValues.
4. (name1, name1) * name2      -> (name_child, name_child) * name_child or
   (name1, name1) * (name2, _) -> (name_child, name_child) * (name_child, _)
   This is a combination of cases 2 and 3, where "name_child" is a child name of
   both "name1" and "name2".
We only need to support diagonal matrix blocks of scalar values in cases 3 and 4
because we cannot extract internal columns from FieldNameDict entries.
=#
function matrix_product_keys(set1::FieldMatrixKeys, set2::FieldNameSet)
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    result_values′ = unrolled_mapflatten(set1.values) do name_pair1
        overlapping_set2_values = unrolled_filter(set2.values) do value2
            row_name2 = eltype(set2) <: FieldName ? value2 : value2[1]
            names_are_overlapping(name_pair1[2], row_name2)
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
    result_values = unique_and_non_overlapping_values(result_values′, name_tree)
    # Note: the modification of result_values may trigger multiplication case 4.
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
        names_are_overlapping(product_row_name, name_pair1[1])
    end
    result_values = unrolled_mapflatten(overlapping_set1_values) do name_pair1
        overlapping_set2_values = unrolled_filter(set2.values) do value2
            row_name2 = eltype(set2) <: FieldName ? value2 : value2[1]
            names_are_overlapping(name_pair1[2], row_name2) && (
                eltype(set2) <: FieldName ||
                names_are_overlapping(product_key[2], value2[2])
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
                product_row_name == row_name2 &&
                    name_pair1[1] == name_pair1[2] ||
                    error("Invalid matrix product key $product_key")
                row_name2
            end
        end
    end
    return FieldVectorKeys(result_values, name_tree)
end

################################################################################

# Internal functions:

check_values(values, name_tree) =
    unrolled_foreach(values) do value
        (isnothing(name_tree) || is_valid_value(value, name_tree)) || error(
            "Invalid FieldNameSet value: $value is incompatible with name_tree",
        )
        duplicate_values = unrolled_filter(isequal(value), values)
        length(duplicate_values) == 1 || error(
            "Duplicate FieldNameSet values: $(length(duplicate_values)) copies \
             of $value have been passed to a FieldNameSet constructor",
        )
        overlapping_values = unrolled_filter(values) do value′
            value != value′ && values_are_overlapping(value, value′)
        end
        if !isempty(overlapping_values)
            overlapping_values_string =
                length(overlapping_values) == 2 ?
                join(overlapping_values, " or ") :
                join(overlapping_values, ", ", ", or ")
            error("Overlapping FieldNameSet values: $value cannot be in the \
                   same FieldNameSet as $overlapping_values_string")
        end
    end

combine_name_trees(::Nothing, ::Nothing) = nothing
combine_name_trees(name_tree1, ::Nothing) = name_tree1
combine_name_trees(::Nothing, name_tree2) = name_tree2
combine_name_trees(name_tree1, name_tree2) =
    name_tree1 == name_tree2 ? name_tree1 :
    error("Mismatched FieldNameTrees: The ability to combine different \
           FieldNameTrees has not been implemented")

is_valid_value(name::FieldName, name_tree) = is_valid_name(name, name_tree)
is_valid_value(name_pair::FieldNamePair, name_tree) =
    is_valid_name(name_pair[1], name_tree) &&
    is_valid_name(name_pair[2], name_tree)

values_are_overlapping(name1::FieldName, name2::FieldName) =
    names_are_overlapping(name1, name2)
values_are_overlapping(name_pair1::FieldNamePair, name_pair2::FieldNamePair) =
    names_are_overlapping(name_pair1[1], name_pair2[1]) &&
    names_are_overlapping(name_pair1[2], name_pair2[2])

is_child_value(name1::FieldName, name2::FieldName) = is_child_name(name1, name2)
is_child_value(name_pair1::FieldNamePair, name_pair2::FieldNamePair) =
    is_child_name(name_pair1[1], name_pair2[1]) &&
    is_child_name(name_pair1[2], name_pair2[2])

is_value_in_set(value, values, name_tree) =
    if unrolled_any(isequal(value), values)
        true
    elseif unrolled_any(value′ -> is_child_value(value, value′), values)
        isnothing(name_tree) && error(
            "Cannot check if $value is in FieldNameSet without a FieldNameTree",
        )
        is_valid_value(value, name_tree)
    else
        false
    end

function non_overlapping_values(values1, values2, name_tree)
    new_values1 = unrolled_mapflatten(values1) do value
        value_or_non_overlapping_children(value, values2, name_tree)
    end
    new_values2 = unrolled_mapflatten(values2) do value
        value_or_non_overlapping_children(value, values1, name_tree)
    end
    if eltype(values1) <: FieldName
        new_values1, new_values2
    else
        # Repeat the above operation to handle complex matrix key overlaps.
        new_values1′ = unrolled_mapflatten(new_values1) do value
            value_or_non_overlapping_children(value, new_values2, name_tree)
        end
        new_values2′ = unrolled_mapflatten(new_values2) do value
            value_or_non_overlapping_children(value, new_values1, name_tree)
        end
        return new_values1′, new_values2′
    end
end

function unique_and_non_overlapping_values(values, name_tree)
    new_values = unrolled_mapflatten(values) do value
        value_or_non_overlapping_children(value, values, name_tree)
    end
    return unrolled_unique(new_values)
end

function value_or_non_overlapping_children(name::FieldName, names, name_tree)
    need_child_names = unrolled_any(names) do name′
        is_child_value(name′, name) && name′ != name
    end
    need_child_names || return (name,)
    isnothing(name_tree) &&
        error("Cannot compute child names of $name without a FieldNameTree")
    return unrolled_mapflatten(child_names(name, name_tree)) do child_name
        value_or_non_overlapping_children(child_name, names, name_tree)
    end
end
function value_or_non_overlapping_children(
    name_pair::FieldNamePair,
    name_pairs,
    name_tree,
)
    need_row_child_names = unrolled_any(name_pairs) do name_pair′
        is_child_value(name_pair′, name_pair) && name_pair′[1] != name_pair[1]
    end
    need_col_child_names = unrolled_any(name_pairs) do name_pair′
        is_child_value(name_pair′, name_pair) && name_pair′[2] != name_pair[2]
    end
    need_row_child_names || need_col_child_names || return (name_pair,)
    isnothing(name_tree) && error(
        "Cannot compute child name pairs of $name_pair without a FieldNameTree",
    )
    row_name_children =
        need_row_child_names ? child_names(name_pair[1], name_tree) :
        (name_pair[1],)
    col_name_children =
        need_col_child_names ? child_names(name_pair[2], name_tree) :
        (name_pair[2],)
    return unrolled_mapflatten(row_name_children) do row_name_child
        unrolled_mapflatten(col_name_children) do col_name_child
            child_pair = (row_name_child, col_name_child)
            value_or_non_overlapping_children(child_pair, name_pairs, name_tree)
        end
    end
end

set_complement_values(_, _, ::Nothing) =
    error("Cannot compute complement of a FieldNameSet without a FieldNameTree")
set_complement_values(::Type{<:FieldName}, names, name_tree::FieldNameTree) =
    complement_values_in_subtree(names, name_tree)
set_complement_values(
    ::Type{<:FieldNamePair},
    name_pairs,
    name_tree::FieldNameTree,
) = complement_values_in_subtree_pair(name_pairs, (name_tree, name_tree))

function complement_values_in_subtree(names, subtree)
    name = subtree.name
    unrolled_all(name′ -> !is_child_value(name, name′), names) || return ()
    unrolled_any(name′ -> is_child_value(name′, name), names) || return (name,)
    return unrolled_mapflatten(subtree.subtrees) do subsubtree
        complement_values_in_subtree(names, subsubtree)
    end
end

function complement_values_in_subtree_pair(name_pairs, subtree_pair)
    name_pair = (subtree_pair[1].name, subtree_pair[2].name)
    is_name_pair_in_complement = unrolled_all(name_pairs) do name_pair′
        !is_child_value(name_pair, name_pair′)
    end
    is_name_pair_in_complement || return ()
    need_row_subsubtrees = unrolled_any(name_pairs) do name_pair′
        is_child_value(name_pair′, name_pair) && name_pair′[1] != name_pair[1]
    end
    need_col_subsubtrees = unrolled_any(name_pairs) do name_pair′
        is_child_value(name_pair′, name_pair) && name_pair′[2] != name_pair[2]
    end
    need_row_subsubtrees || need_col_subsubtrees || return (name_pair,)
    row_subsubtrees =
        need_row_subsubtrees ? subtree_pair[1].subtrees : (subtree_pair[1],)
    col_subsubtrees =
        need_col_subsubtrees ? subtree_pair[2].subtrees : (subtree_pair[2],)
    return unrolled_mapflatten(row_subsubtrees) do row_subsubtree
        unrolled_mapflatten(col_subsubtrees) do col_subsubtree
            subsubtree_pair = (row_subsubtree, col_subsubtree)
            complement_values_in_subtree_pair(name_pairs, subsubtree_pair)
        end
    end
end

################################################################################

# This is required for type-stability as of Julia 1.9.
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(value_or_non_overlapping_children)
        m.recursion_relation = dont_limit
    end
    for m in methods(complement_values_in_subtree)
        m.recursion_relation = dont_limit
    end
    for m in methods(complement_values_in_subtree_pair)
        m.recursion_relation = dont_limit
    end
end
