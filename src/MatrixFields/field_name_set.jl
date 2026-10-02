const FieldNamePair = Tuple{FieldName, FieldName}

"""
    FieldNameSet{T}(values, [name_tree])

`AbstractSet` that contains values of type `T`, serving as an analogue of a
`KeySet` for a [`FieldNameDict`](@ref). There are two aliases of `FieldNameSet`:

  - `FieldVectorKeys`, for which `T` is `FieldName`.
  - `FieldMatrixKeys`, for which `T` is `Tuple{FieldName, FieldName}`; each tuple
    of type `T` represents a pair of row-column indices.

Since `FieldName`s are singleton types, the result of almost any `FieldNameSet`
operation can be inferred during compilation. So, with the exception of `map`,
`foreach`, and `set_string`, functions of `FieldNameSet`s have no performance cost
at runtime (as long as their arguments are inferrable).

Unlike other `AbstractSet`s, `FieldNameSet` has special behavior for overlapping
values. For example, the `FieldName`s `@name(a.b)` and `@name(a.b.c)` overlap, so
any set operation needs to first decompose `@name(a.b)` into its child values
before combining it with `@name(a.b.c)`. To support this (and to support set
complements), `FieldNameSet` stores a [`FieldNameTree`](@ref) `name_tree`, which it
uses to infer child values. If `name_tree` is not specified, it defaults to
`nothing`, which disables some `FieldNameSet` operations. For binary operations
like `union` or `setdiff`, only one set needs to specify a `name_tree`; if both
sets specify a `name_tree`, the `name_tree`s must be identical.
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

#=
A name chain is the tuple of `Symbol`s and `Int`s inside a `FieldName`. Two
names of equal depth share a chain type, while every name has its own type, so
predicates that close over a chain compile to one method instance per depth
rather than one per name. The operations that compare every value against
every other work over chains and convert back at the end.
=#

const NameChain = Tuple{Vararg{Union{Symbol, Int}}}
const NamePairChain = Tuple{NameChain, NameChain}

name_chain(::FieldName{chain}) where {chain} = chain
name_chain(name_pair::FieldNamePair) =
    (name_chain(name_pair[1]), name_chain(name_pair[2]))

field_name(chain::NameChain) = FieldName{chain}()
field_name(pair_chain::NamePairChain) =
    (field_name(pair_chain[1]), field_name(pair_chain[2]))

is_child_chain(child::NameChain, parent::NameChain) =
    length(child) >= length(parent) &&
    unrolled_take(child, Val(length(parent))) == parent
is_child_chain(child::NamePairChain, parent::NamePairChain) =
    is_child_chain(child[1], parent[1]) && is_child_chain(child[2], parent[2])

is_overlapping_chain(chain1::NameChain, chain2::NameChain) =
    is_child_chain(chain1, chain2) || is_child_chain(chain2, chain1)
is_overlapping_chain(chain1::NamePairChain, chain2::NamePairChain) =
    is_overlapping_chain(chain1[1], chain2[1]) &&
    is_overlapping_chain(chain1[2], chain2[2])

chain_in(chain, chains::Vector) = any(c -> c == chain, chains)

is_valid_chain(chain::NameChain, valid_chains::Vector) =
    chain_in(chain, valid_chains)
is_valid_chain(pair_chain::NamePairChain, valid_chains::Vector) =
    is_valid_chain(pair_chain[1], valid_chains) &&
    is_valid_chain(pair_chain[2], valid_chains)

# Every chain in a FieldNameTree, so that validity is a membership test.
tree_chain_vector(::Nothing) = nothing
tree_chain_vector(tree::FieldNameTree) = push_tree_chains!(Any[], tree)
function push_tree_chains!(chains::Vector, @nospecialize(tree))
    push!(chains, name_chain(tree.name))
    tree isa FieldNameTreeNode &&
        foreach(subtree -> push_tree_chains!(chains, subtree), tree.subtrees)
    return chains
end

"""
    FieldVectorKeys(values, [name_tree])

Alias for `FieldNameSet{FieldName}`: the key set of a `FieldVectorView`, i.e. a
set of `FieldName`s such as `(@name(c.ρ), @name(f.u₃))`, that serves as the
analogue of a `KeySet` for a [`FieldNameDict`](@ref).
"""
const FieldVectorKeys = FieldNameSet{FieldName}

"""
    FieldMatrixKeys(values, [name_tree])

Alias for `FieldNameSet{Tuple{FieldName, FieldName}}`: the key set of a
`FieldMatrix`, i.e. a set of `(row_name, col_name)` pairs of `FieldName`s such as
`((@name(c.ρ), @name(c.ρ)), (@name(c.ρ), @name(f.u₃)))`, that serves as the
analogue of a `KeySet` for a [`FieldNameDict`](@ref).
"""
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

is_child_value(name1::FieldName, name2::FieldName) = is_child_name(name1, name2)
is_child_value(name_pair1::FieldNamePair, name_pair2::FieldNamePair) =
    is_child_name(name_pair1[1], name_pair2[1]) &&
    is_child_name(name_pair1[2], name_pair2[2])

instance(@nospecialize(T)) =
    isdefined(T, :instance) ? T.instance :
    error("$T is not a singleton type, so its value cannot be read from its type")
constant(value) = Expr(:block, value)

chain_vector(values::Tuple) = Any[name_chain(values[i]) for i in 1:length(values)]
name_tuple(chains::Vector) = Tuple(Any[field_name(c) for c in chains])
# Children of a name are the valid chains one level deeper that start with it.
function child_chain_vector(chain, valid_chains::Vector)
    isnothing(valid_chains) && error("Missing FieldNameTree")
    children = Any[
        c for c in valid_chains if
        length(c) == length(chain) + 1 && is_child_chain(c, chain)
    ]
    isempty(children) &&
        error("$(field_name(chain)) does not have child names")
    return children
end

chain_product(chains1::Vector, chains2::Vector) =
    Any[(c1, c2) for c2 in chains2 for c1 in chains1]

expand_child_chains(chain::NameChain, overlapping::Vector, valid_chains) =
    all(c -> c != chain && is_child_chain(c, chain), overlapping) ?
    child_chain_vector(chain, valid_chains) : Any[chain]
function expand_child_chains(
    pair_chain::NamePairChain,
    overlapping::Vector,
    valid_chains,
)
    row, col = pair_chain
    rows =
        all(p -> p[1] != row && is_child_chain(p[1], row), overlapping) ?
        child_chain_vector(row, valid_chains) : Any[]
    cols =
        all(p -> p[2] != col && is_child_chain(p[2], col), overlapping) ?
        child_chain_vector(col, valid_chains) : Any[]
    n_rows, n_cols = length(rows), length(cols)
    return if n_rows > 1 && n_cols > 1 || n_rows == 1 && n_cols == 1
        chain_product(rows, cols)
    elseif n_rows > 1 && n_cols == 1 || n_rows > 0 && n_cols == 0
        chain_product(rows, Any[col])
    elseif n_rows == 1 && n_cols > 1 || n_rows == 0 && n_cols > 0
        chain_product(Any[row], cols)
    else
        Any[pair_chain]
    end
end

function unique_and_non_overlapping_chains(chains::Vector, valid_chains)
    unique_chains =
        Any[c for (i, c) in enumerate(chains) if !any(j -> chains[j] == c, 1:(i - 1))]
    overlaps(c) =
        any(c2 -> c != c2 && is_overlapping_chain(c, c2), unique_chains)
    overlapping = Any[c for c in unique_chains if overlaps(c)]
    isempty(overlapping) && return unique_chains
    isnothing(valid_chains) && error(
        "Missing FieldNameTree: Cannot eliminate overlaps among \
         $(values_string(name_tuple(overlapping))) without a FieldNameTree",
    )
    expanded = Any[]
    for c in overlapping
        with_c = Any[c2 for c2 in overlapping if c != c2 && is_overlapping_chain(c, c2)]
        append!(expanded, expand_child_chains(c, with_c, valid_chains))
    end
    non_overlapping = Any[c for c in unique_chains if !overlaps(c)]
    return vcat(
        non_overlapping,
        unique_and_non_overlapping_chains(expanded, valid_chains),
    )
end

function union_chains(chains1::Vector, chains2::Vector, valid_chains)
    unique2 = Any[c for c in chains2 if !chain_in(c, chains1)]
    overlaps1(c1) = any(c2 -> is_overlapping_chain(c1, c2), unique2)
    overlapping1 = Any[c for c in chains1 if overlaps1(c)]
    isempty(overlapping1) && return vcat(chains1, unique2)
    overlaps2(c2) = any(c1 -> is_overlapping_chain(c1, c2), chains1)
    overlapping2 = Any[c for c in unique2 if overlaps2(c)]
    isnothing(valid_chains) && error(
        "Missing FieldNameTree: Cannot eliminate overlaps between \
         $(name_tuple(overlapping1)) and $(name_tuple(overlapping2)) without a \
         FieldNameTree",
    )
    expanded1 = Any[]
    for c1 in overlapping1
        with_c1 = Any[c2 for c2 in overlapping2 if is_overlapping_chain(c1, c2)]
        append!(expanded1, expand_child_chains(c1, with_c1, valid_chains))
    end
    expanded2 = Any[]
    for c2 in overlapping2
        with_c2 = Any[c1 for c1 in overlapping1 if is_overlapping_chain(c1, c2)]
        append!(expanded2, expand_child_chains(c2, with_c2, valid_chains))
    end
    return vcat(
        Any[c for c in chains1 if !overlaps1(c)],
        Any[c for c in unique2 if !overlaps2(c)],
        union_chains(expanded1, expanded2, valid_chains),
    )
end

is_chain_in_set(chain, chains::Vector, valid_chains) =
    chain_in(chain, chains) ||
    any(c -> is_child_chain(chain, c), chains) &&
    (isnothing(valid_chains) ? true : is_valid_chain(chain, valid_chains))

"""
    check_chains(chains, valid_chains)

Throw an error if any chain is incompatible with `valid_chains`, is duplicated,
or overlaps another chain.
"""
function check_chains(chains::Vector, valid_chains)
    for (i, chain) in enumerate(chains)
        (isnothing(valid_chains) || is_valid_chain(chain, valid_chains)) ||
            error(
                "Invalid FieldNameSet value: $(field_name(chain)) is \
                 incompatible with the FieldNameTree",
            )
        n_duplicate_values = count(c -> c == chain, chains)
        n_duplicate_values == 1 || error(
            "Duplicate FieldNameSet values: $n_duplicate_values copies of \
            $(field_name(chain)) have been passed to a FieldNameSet constructor",
        )
        overlapping = Any[
            c for c in chains if c != chain && is_overlapping_chain(chain, c)
        ]
        isempty(overlapping) || error(
            "Overlapping FieldNameSet values: $(field_name(chain)) cannot be \
            in the same FieldNameSet as $(values_string(name_tuple(overlapping)))",
        )
    end
end

@generated function check_values(::V, ::N) where {V <: Tuple, N}
    check_chains(chain_vector(instance(V)), tree_chain_vector(instance(N)))
    return nothing
end

@generated is_value_in_set(::W, ::V, ::N) where {W, V <: Tuple, N} = constant(
    is_chain_in_set(
        name_chain(instance(W)),
        chain_vector(instance(V)),
        tree_chain_vector(instance(N)),
    ),
)

@generated unique_and_non_overlapping_values(::V, ::N) where {V <: Tuple, N} =
    constant(
        name_tuple(
            unique_and_non_overlapping_chains(
                chain_vector(instance(V)),
                tree_chain_vector(instance(N)),
            ),
        ),
    )

@generated union_values(::V1, ::V2, ::N) where {V1 <: Tuple, V2 <: Tuple, N} =
    constant(
        name_tuple(
            union_chains(
                chain_vector(instance(V1)),
                chain_vector(instance(V2)),
                tree_chain_vector(instance(N)),
            ),
        ),
    )
