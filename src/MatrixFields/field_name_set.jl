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

@noinline function check_chain_values(
    chains::Vector,
    @nospecialize(values::Tuple),
    tree,
)
    n_values = length(chains)
    for i in 1:n_values
        value, chain = values[i], chains[i]
        (isnothing(tree) || is_valid_chain_value(chain, tree)) || error(
            "Invalid FieldNameSet value: $value is incompatible with the \
             FieldNameTree",
        )
        n_duplicate_values = count(j -> chains[j] == chain, 1:n_values)
        n_duplicate_values == 1 || error(
            "Duplicate FieldNameSet values: $n_duplicate_values copies of \
            $value have been passed to a FieldNameSet constructor",
        )
        overlapping_values = Any[
            values[j] for j in 1:n_values if
            chains[j] != chain && is_overlapping_chain_value(chain, chains[j])
        ]
        isempty(overlapping_values) || error(
            "Overlapping FieldNameSet values: $value cannot be in the same \
            FieldNameSet as $(values_string(overlapping_values))",
        )
    end
    return nothing
end

"""
    check_values(values, name_tree)

Throw an error if any value in `values` is incompatible with `name_tree`, is
duplicated, or overlaps with another value. Every `FieldName` is a singleton, so
`values` and `name_tree` are fully determined by their types, and the checks
run once per type while this method is being generated. The compiled method
body is `nothing`.
"""
@generated function check_values(
    ::V,
    ::N,
) where {V <: Tuple, N <: Union{FieldNameTree, Nothing}}
    values = instance(V)
    check_chain_values(
        chain_values(values),
        values,
        tree_node(instance(N)),
    )
    return nothing
end

"""
    FieldVectorKeys(values, [name_tree])

Alias for `FieldNameSet{FieldName}`: the key set of a `FieldVectorView`, i.e. a
set of `FieldName`s such as `(@name(c.ρ), @name(f.u₃))`, that serves as the
analogue of a `KeySet` for a [`FieldNameDict`](@ref).
"""
const FieldVectorKeys = FieldNameSet{FieldName}

chain_values(set::FieldNameSet) = chain_values(set.values)

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

#=
The set operations below are generated functions following one pattern.

A `FieldName` is a singleton, so a `FieldNameSet` and a `FieldNameTree` are
singletons too, and `instance(T)` recovers the one value of type `T`. Each
generator therefore reads its arguments out of their types, computes the answer
with ordinary runtime code over name chains (field_name_set_algebra.jl), and
emits that answer as a constant. The compiled method is a single `return`, and
the work happens once per type signature rather than once per value.

Two rules apply to everything in this section.

A generated function may only call functions defined before it, so the algebra
file is included first (MatrixFields.jl) and `check_chain_values` is defined
above `check_values`. Moving a helper below its caller fails to precompile with
a "method may be too new" error that does not name the cause.

The values embedded by `constant` must be immutable. `FieldNameSet`s, `Bool`s
and tuples of `FieldName`s are. Embedding a mutable object would share one
instance across every call site.
=#

"""
    instance(T)

Return the single value of the singleton type `T`. Every argument of the
generated functions below is a singleton, and this reports a usable error if
that ever stops holding.
"""
function instance(@nospecialize(T))
    isdefined(T, :instance) || error(
        "$T is not a singleton type, so its value cannot be read from its type \
         inside a generated function",
    )
    return T.instance
end

"""
    constant(value)

Return an expression that evaluates to `value`. Used as the return value of a
generated function, so that the compiled method body is `value` itself rather
than code that computes it.
"""
constant(value) = Expr(:block, value)

# Called from inside the generators, where `check_values` (itself generated)
# runs on the result.
@noinline set_from_chains(@nospecialize(T), chains::Vector, @nospecialize(name_tree)) =
    FieldNameSet{T}(values_from_chains(chains), name_tree)

@generated Base.in(value::Union{FieldName, FieldNamePair}, set::FieldNameSet) =
    constant(
        is_chain_value_in_set(
            chain_value(instance(value)),
            chain_values(instance(set)),
            tree_node(instance(set).name_tree),
        ),
    )

@generated Base.:(==)(set1::FieldNameSet, set2::FieldNameSet) =
    constant(equal_chains(chain_values(instance(set1)), chain_values(instance(set2))))

@generated function Base.issubset(set1::FieldNameSet, set2::FieldNameSet)
    s1, s2 = instance(set1), instance(set2)
    name_tree = combine_name_trees(s1.name_tree, s2.name_tree)
    return constant(
        issubset_chains(chain_values(s1), chain_values(s2), tree_node(name_tree)),
    )
end

# `union`, `intersect` and `setdiff` differ only in which chain operation they
# call.
function combined_set(set1, set2, chains_op::F) where {F}
    T = combine_eltypes(eltype(set1), eltype(set2))
    name_tree = combine_name_trees(set1.name_tree, set2.name_tree)
    chains =
        chains_op(chain_values(set1), chain_values(set2), tree_node(name_tree))
    return set_from_chains(T, chains, name_tree)
end

@generated Base.union(set1::FieldNameSet, set2::FieldNameSet) =
    constant(combined_set(instance(set1), instance(set2), union_chains))

@generated Base.intersect(set1::FieldNameSet, set2::FieldNameSet) =
    constant(combined_set(instance(set1), instance(set2), intersect_chains))

@generated Base.setdiff(set1::FieldNameSet, set2::FieldNameSet) =
    constant(combined_set(instance(set1), instance(set2), setdiff_chains))

replace_name_tree(set::FieldNameSet, name_tree) =
    FieldNameSet{eltype(set)}(set.values, name_tree)

set_string(set) = values_string(set.values)

set_complement(set) = setdiff(universal_set(eltype(set), set.name_tree), set)

is_subset_that_covers_set(set1, set2) =
    issubset(set1, set2) && isempty(setdiff(set2, set1))

@generated corresponding_matrix_keys(set::FieldVectorKeys) = constant(
    set_from_chains(
        FieldNamePair,
        corresponding_matrix_chains(chain_values(instance(set))),
        instance(set).name_tree,
    ),
)

@generated function cartesian_product(row_set::FieldVectorKeys, col_set::FieldVectorKeys)
    rows, cols = instance(row_set), instance(col_set)
    name_tree = combine_name_trees(rows.name_tree, cols.name_tree)
    chains = chain_product(chain_values(rows), chain_values(cols))
    return constant(set_from_chains(FieldNamePair, chains, name_tree))
end

@generated function corresponding_vector_keys(set::FieldMatrixKeys, ::Val{N}) where {N}
    s = instance(set)
    chains = corresponding_vector_chains(chain_values(s), N, tree_node(s.name_tree))
    return constant(set_from_chains(FieldName, chains, s.name_tree))
end

matrix_row_keys(set::FieldMatrixKeys) = corresponding_vector_keys(set, Val(1))
matrix_col_keys(set::FieldMatrixKeys) = corresponding_vector_keys(set, Val(2))

@generated matrix_off_diagonal_keys(set::FieldMatrixKeys) = constant(
    set_from_chains(
        FieldNamePair,
        off_diagonal_chains(chain_values(instance(set))),
        instance(set).name_tree,
    ),
)

@generated matrix_diagonal_keys(set::FieldMatrixKeys) = constant(
    set_from_chains(
        FieldNamePair,
        diagonal_chains(chain_values(instance(set))),
        instance(set).name_tree,
    ),
)

@generated function matrix_inferred_diagonal_keys(set::FieldMatrixKeys)
    s = instance(set)
    chains = inferred_diagonal_chains(chain_values(s), tree_node(s.name_tree))
    return constant(set_from_chains(FieldNamePair, chains, s.name_tree))
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
@generated function matrix_product_keys(set1::FieldMatrixKeys, set2::FieldNameSet)
    s1, s2 = instance(set1), instance(set2)
    name_tree = combine_name_trees(s1.name_tree, s2.name_tree)
    is_vector = eltype(s2) <: FieldName
    chains = product_chains(
        chain_values(s1),
        chain_values(s2),
        is_vector,
        tree_node(name_tree),
    )
    return constant(set_from_chains(eltype(s2), chains, name_tree))
end

@generated function summand_names_for_matrix_product(
    product_key::Union{FieldName, FieldNamePair},
    set1::FieldMatrixKeys,
    set2::FieldNameSet,
)
    s1, s2 = instance(set1), instance(set2)
    name_tree = combine_name_trees(s1.name_tree, s2.name_tree)
    is_vector = eltype(s2) <: FieldName
    chains = summand_chains(
        chain_value(instance(product_key)),
        chain_values(s1),
        chain_values(s2),
        is_vector,
    )
    return constant(set_from_chains(FieldName, chains, name_tree))
end

################################################################################

# Internal functions:

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
