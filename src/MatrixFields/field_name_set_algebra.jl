#=
Set algebra on name chains stored as runtime data.

Every `FieldName` is a singleton, so a `FieldNameSet` and a `FieldNameTree` are
fully determined by their types. The generated functions in field_name_set.jl
call the functions below with the singleton values reconstructed from the
argument types, and return the result as a constant. Each set operation is
therefore computed once per type signature, with a fixed set of method
instances, and costs nothing at run time.

A name is a `Vector{Any}` chain of `Symbol`s and `Int`s. A value is a chain
(vector case) or a `Tuple` of two chains (matrix case). A tree node is a
`(chain, Vector{node})` pair with an empty vector for leaves. Element order
matches the unrolled `Tuple` implementation this replaces.
=#

const FieldNamePair = Tuple{FieldName, FieldName}

values_string(values) =
    length(values) == 2 ? join(values, " and ") : join(values, ", ", ", and ")

const NameChain = Vector{Any}
const NameTreeNode = Tuple{NameChain, Vector{Any}}

name_chain(::FieldName{chain}) where {chain} = Any[chain...]
chain_value(name::FieldName) = name_chain(name)
chain_value(name_pair::FieldNamePair) =
    (name_chain(name_pair[1]), name_chain(name_pair[2]))
chain_values(values::Tuple) = Any[chain_value(values[i]) for i in 1:length(values)]

name_from_chain(chain::NameChain) = FieldName(chain...)
value_from_chain(chain::NameChain) = name_from_chain(chain)
value_from_chain(pair::Tuple) = (name_from_chain(pair[1]), name_from_chain(pair[2]))
values_from_chains(chains::Vector) =
    Tuple(Any[value_from_chain(chains[i]) for i in 1:length(chains)])

function tree_node(@nospecialize(tree))
    subtrees =
        tree isa FieldNameTreeNode ?
        Any[tree_node(tree.subtrees[i]) for i in 1:length(tree.subtrees)] : Any[]
    return (name_chain(tree.name), subtrees)
end
tree_node(::Nothing) = nothing

is_child_chain(child::NameChain, parent::NameChain) =
    length(child) >= length(parent) &&
    view(child, 1:length(parent)) == parent
is_overlapping_chain(chain1::NameChain, chain2::NameChain) =
    is_child_chain(chain1, chain2) || is_child_chain(chain2, chain1)

is_child_chain_value(value1::NameChain, value2::NameChain) =
    is_child_chain(value1, value2)
is_child_chain_value(pair1::Tuple, pair2::Tuple) =
    is_child_chain(pair1[1], pair2[1]) && is_child_chain(pair1[2], pair2[2])
is_overlapping_chain_value(value1::NameChain, value2::NameChain) =
    is_overlapping_chain(value1, value2)
is_overlapping_chain_value(pair1::Tuple, pair2::Tuple) =
    is_overlapping_chain(pair1[1], pair2[1]) &&
    is_overlapping_chain(pair1[2], pair2[2])

is_valid_chain(name::NameChain, tree::NameTreeNode) =
    name == tree[1] || any(subtree -> is_valid_chain(name, subtree), tree[2])
is_valid_chain_value(name::NameChain, tree) = is_valid_chain(name, tree)
is_valid_chain_value(pair::Tuple, tree) =
    is_valid_chain(pair[1], tree) && is_valid_chain(pair[2], tree)

function subtree_at_chain(name::NameChain, tree::NameTreeNode)
    name == tree[1] && return tree
    subtrees = filter(subtree -> is_valid_chain(name, subtree), tree[2])
    @assert length(subtrees) == 1
    return subtree_at_chain(name, subtrees[1])
end
function child_chains(name::NameChain, tree::NameTreeNode)
    is_valid_chain(name, tree) || error("$(name_from_chain(name)) is not a valid name")
    subtree = subtree_at_chain(name, tree)
    isempty(subtree[2]) &&
        error("$(name_from_chain(name)) does not have child names")
    return Any[node[1] for node in subtree[2]]
end

chain_in(value, values::Vector) = any(value′ -> value′ == value, values)
is_chain_value_in_set(value, values::Vector, tree) =
    chain_in(value, values) ||
    any(value′ -> is_child_chain_value(value, value′), values) &&
    (isnothing(tree) ? true : is_valid_chain_value(value, tree))

# First occurrence of each value is kept, as in unrolled_unique.
unique_chains(values::Vector) =
    Any[
        value for
        (i, value) in enumerate(values) if !any(j -> values[j] == value, 1:(i - 1))
    ]

# First iterator varies fastest, as in unrolled_product.
chain_product(values1::Vector, values2::Vector) =
    Any[(value1, value2) for value2 in values2 for value1 in values1]

function unique_and_non_overlapping_chains(values::Vector, tree)
    unique_values = unique_chains(values)
    overlaps(value) = any(
        value′ -> value != value′ && is_overlapping_chain_value(value, value′),
        unique_values,
    )
    overlapping = Any[value for value in unique_values if overlaps(value)]
    non_overlapping = Any[value for value in unique_values if !overlaps(value)]
    isempty(overlapping) && return unique_values
    isnothing(tree) && error(
        "Missing FieldNameTree: Cannot eliminate overlaps among \
         $(values_string(values_from_chains(overlapping))) without a FieldNameTree",
    )
    expanded = Any[]
    for value in overlapping
        with_value = Any[
            value′ for value′ in overlapping if
            value != value′ && is_overlapping_chain_value(value, value′)
        ]
        append!(expanded, expand_child_chains(value, with_value, tree))
    end
    return vcat(non_overlapping, unique_and_non_overlapping_chains(expanded, tree))
end

function union_chains(values1::Vector, values2::Vector, tree)
    unique2 = Any[value for value in values2 if !chain_in(value, values1)]
    overlaps1(value1) =
        any(value2 -> is_overlapping_chain_value(value1, value2), unique2)
    overlapping1 = Any[value for value in values1 if overlaps1(value)]
    non_overlapping1 = Any[value for value in values1 if !overlaps1(value)]
    isempty(overlapping1) && return vcat(values1, unique2)
    overlaps2(value2) =
        any(value1 -> is_overlapping_chain_value(value1, value2), values1)
    overlapping2 = Any[value for value in unique2 if overlaps2(value)]
    non_overlapping2 = Any[value for value in unique2 if !overlaps2(value)]
    isnothing(tree) && error(
        "Missing FieldNameTree: Cannot eliminate overlaps between \
         $(values_from_chains(overlapping1)) and \
         $(values_from_chains(overlapping2)) without a FieldNameTree",
    )
    expanded1 = Any[]
    for value1 in overlapping1
        with_value1 = Any[
            value2 for value2 in overlapping2 if
            is_overlapping_chain_value(value1, value2)
        ]
        append!(expanded1, expand_child_chains(value1, with_value1, tree))
    end
    expanded2 = Any[]
    for value2 in overlapping2
        with_value2 = Any[
            value1 for value1 in overlapping1 if
            is_overlapping_chain_value(value1, value2)
        ]
        append!(expanded2, expand_child_chains(value2, with_value2, tree))
    end
    return vcat(
        non_overlapping1,
        non_overlapping2,
        union_chains(expanded1, expanded2, tree),
    )
end

expand_child_chains(name::NameChain, overlapping::Vector, tree) =
    all(name′ -> name′ != name && is_child_chain(name′, name), overlapping) ?
    child_chains(name, tree) : Any[name]
function expand_child_chains(pair::Tuple, overlapping::Vector, tree)
    row_name, col_name = pair
    row_children_needed = all(
        pair′ -> pair′[1] != row_name && is_child_chain(pair′[1], row_name),
        overlapping,
    )
    col_children_needed = all(
        pair′ -> pair′[2] != col_name && is_child_chain(pair′[2], col_name),
        overlapping,
    )
    row_children = row_children_needed ? child_chains(row_name, tree) : Any[]
    col_children = col_children_needed ? child_chains(col_name, tree) : Any[]
    n_rows, n_cols = length(row_children), length(col_children)
    return if n_rows > 1 && n_cols > 1 || n_rows == 1 && n_cols == 1
        chain_product(row_children, col_children)
    elseif n_rows > 1 && n_cols == 1 || n_rows > 0 && n_cols == 0
        chain_product(row_children, Any[col_name])
    elseif n_rows == 1 && n_cols > 1 || n_rows == 0 && n_cols > 0
        chain_product(Any[row_name], col_children)
    else
        Any[pair]
    end
end

intersect_chains(values1::Vector, values2::Vector, tree) = Any[
    value for value in union_chains(values1, values2, tree) if
    is_chain_value_in_set(value, values1, tree) &&
        is_chain_value_in_set(value, values2, tree)
]
setdiff_chains(values1::Vector, values2::Vector, tree) = Any[
    value for value in union_chains(values1, values2, tree) if
    !is_chain_value_in_set(value, values2, tree)
]
issubset_chains(values1::Vector, values2::Vector, tree) =
    all(value -> is_chain_value_in_set(value, values2, tree), values1)
equal_chains(values1::Vector, values2::Vector) =
    all(value -> chain_in(value, values2), values1) &&
    all(value -> chain_in(value, values1), values2)

corresponding_matrix_chains(names::Vector) = Any[(name, name) for name in names]
corresponding_vector_chains(pairs::Vector, n::Int, tree) =
    unique_and_non_overlapping_chains(Any[pair[n] for pair in pairs], tree)
off_diagonal_chains(pairs::Vector) = Any[pair for pair in pairs if pair[1] != pair[2]]
function diagonal_chains(pairs::Vector)
    overlapping = Any[pair for pair in pairs if is_overlapping_chain(pair[1], pair[2])]
    return Any[
        pair[1] == pair[2] ? pair :
        is_child_chain(pair[1], pair[2]) ? (pair[1], pair[1]) : (pair[2], pair[2])
        for pair in overlapping
    ]
end
function inferred_diagonal_chains(pairs::Vector, tree)
    rows = corresponding_vector_chains(pairs, 1, tree)
    cols = corresponding_vector_chains(pairs, 2, tree)
    diag = corresponding_vector_chains(diagonal_chains(pairs), 1, tree)
    all_names =
        issubset_chains(rows, diag, tree) && issubset_chains(cols, diag, tree) ?
        diag : union_chains(rows, cols, tree)
    return corresponding_matrix_chains(all_names)
end

row_chain(value2, is_vector::Bool) = is_vector ? value2 : value2[1]
function product_chains(pairs1::Vector, values2::Vector, is_vector::Bool, tree)
    result = Any[]
    for pair1 in pairs1
        overlapping2 = Any[
            value2 for value2 in values2 if
            is_overlapping_chain(pair1[2], row_chain(value2, is_vector))
        ]
        for value2 in overlapping2
            row2 = row_chain(value2, is_vector)
            if is_child_chain(pair1[2], row2)
                push!(result, is_vector ? pair1[1] : (pair1[1], value2[2]))
            elseif pair1[1] == pair1[2]
                push!(result, value2)
            else
                error("Cannot extract internal column from an off-diagonal key")
            end
        end
    end
    return unique_and_non_overlapping_chains(result, tree)
end

function summand_chains(product_key, pairs1::Vector, values2::Vector, is_vector::Bool)
    product_row = is_vector ? product_key : product_key[1]
    overlapping1 = Any[
        pair1 for pair1 in pairs1 if is_overlapping_chain(product_row, pair1[1])
    ]
    result = Any[]
    for pair1 in overlapping1
        overlapping2 = Any[
            value2 for value2 in values2 if
            is_overlapping_chain(pair1[2], row_chain(value2, is_vector)) &&
                (is_vector || is_overlapping_chain(product_key[2], value2[2])) &&
                (
                    is_child_chain(pair1[2], row_chain(value2, is_vector)) ||
                    product_row == row_chain(value2, is_vector) && pair1[1] == pair1[2]
                )
        ]
        for value2 in overlapping2
            row2 = row_chain(value2, is_vector)
            is_child_chain(product_row, pair1[1]) &&
            (is_vector || product_key[2] == value2[2]) || error(
                "Invalid matrix product key $(value_from_chain(product_key))",
            )
            if is_child_chain(pair1[2], row2)
                push!(
                    result,
                    product_row == pair1[1] ? pair1[2] :
                    pair1[1] == pair1[2] ? product_row : pair1[2],
                )
            else
                push!(result, row2)
            end
        end
    end
    return result
end
