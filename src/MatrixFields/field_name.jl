"""
    FieldName(name_chain...)

A singleton type that represents a chain of `getproperty` calls, which can be
used to access a property or sub-property of an object `x` using the function
`get_field(x, name)`. The entire object `x` can also be accessed with the empty
`FieldName()`.
"""
struct FieldName{name_chain} end
FieldName() = FieldName{()}() # This is required for type stability.
FieldName(name_chain...) = FieldName{name_chain}()

"""
    @name(expr)

Shorthand for constructing a `FieldName`. Some examples include
- `name = @name()`, in which case `get_field(x, name)` returns `x`
- `name = @name(a)`, in which case `get_field(x, name)` returns `x.a`
- `name = @name(a.b.c)`, in which case `get_field(x, name)` returns `x.a.b.c`
- `name = @name(a.b.c.:(1).d)`, in which case `get_field(x, name)` returns
  `x.a.b.c.:(1).d`

This macro is preferred over the `FieldName` constructor because it checks
whether `expr` is a syntactically valid chain of `getproperty` calls before
calling the constructor.
"""
macro name()
    return :(FieldName())
end
macro name(expr)
    return :(FieldName($(expr_to_name_chain(expr))...))
end

expr_to_name_chain(value) = error("$(repr(value)) is not a valid property name")
expr_to_name_chain(value::Union{Symbol, Integer}) = (value,)
expr_to_name_chain(quote_node::QuoteNode) = expr_to_name_chain(quote_node.value)
function expr_to_name_chain(expr::Expr)
    expr.head == :. || error("$expr is not a valid property name")
    arg1, arg2 = expr.args
    return (expr_to_name_chain(arg1)..., expr_to_name_chain(arg2)...)
end

# Show a FieldName with @name syntax, instead of the default constructor syntax.
function Base.show(io::IO, ::FieldName{name_chain}) where {name_chain}
    quoted_names = map(name -> name isa Integer ? ":($name)" : name, name_chain)
    print(io, "@name($(join(quoted_names, '.')))")
end

extract_first(::FieldName{name_chain}) where {name_chain} = first(name_chain)
drop_first(::FieldName{name_chain}) where {name_chain} =
    FieldName(Base.tail(name_chain)...)

has_field(x, ::FieldName{()}) = true
has_field(x, name::FieldName) =
    extract_first(name) in propertynames(x) &&
    has_field(getproperty(x, extract_first(name)), drop_first(name))

get_field(x, ::FieldName{()}) = x
get_field(x, name::FieldName) =
    get_field(getproperty(x, extract_first(name)), drop_first(name))

broadcasted_has_field(::Type{X}, ::FieldName{()}) where {X} = true
broadcasted_has_field(::Type{X}, name::FieldName) where {X} =
    extract_first(name) in fieldnames(X) &&
    broadcasted_has_field(fieldtype(X, extract_first(name)), drop_first(name))

broadcasted_get_field(x, ::FieldName{()}) = x
broadcasted_get_field(x, name::FieldName) =
    broadcasted_get_field(getfield(x, extract_first(name)), drop_first(name))

is_child_name(
    ::FieldName{child_name_chain},
    ::FieldName{parent_name_chain},
) where {child_name_chain, parent_name_chain} =
    length(child_name_chain) >= length(parent_name_chain) &&
    child_name_chain[1:length(parent_name_chain)] == parent_name_chain

names_are_overlapping(name1, name2) =
    is_child_name(name1, name2) || is_child_name(name2, name1)

extract_internal_name(
    child_name::FieldName{child_name_chain},
    parent_name::FieldName{parent_name_chain},
) where {child_name_chain, parent_name_chain} =
    is_child_name(child_name, parent_name) ?
    FieldName(child_name_chain[(length(parent_name_chain) + 1):end]...) :
    error("$child_name is not a child name of $parent_name")

append_internal_name(
    ::FieldName{name_chain},
    ::FieldName{internal_name_chain},
) where {name_chain, internal_name_chain} =
    FieldName(name_chain..., internal_name_chain...)

top_level_names(x) = wrapped_prop_names(Val(propertynames(x)))
wrapped_prop_names(::Val{()}) = ()
wrapped_prop_names(::Val{prop_names}) where {prop_names} = (
    FieldName(first(prop_names)),
    wrapped_prop_names(Val(Base.tail(prop_names)))...,
)

################################################################################

"""
    FieldNameTree(x)

Tree of `FieldName`s that can be used to access `x` with `get_field(x, name)`.
Check whether a `name` is valid by calling `is_valid_name(name, tree)`,
and extract the children of `name` by calling `child_names(name, tree)`.
"""
abstract type FieldNameTree end
struct FieldNameTreeLeaf{V <: FieldName} <: FieldNameTree
    name::V
end
struct FieldNameTreeNode{V <: FieldName, S <: NTuple{<:Any, FieldNameTree}} <:
       FieldNameTree
    name::V
    subtrees::S
end

FieldNameTree(x) = make_subtree_at_name(x, @name())
function make_subtree_at_name(x, name)
    internal_names = top_level_names(get_field(x, name))
    isempty(internal_names) && return FieldNameTreeLeaf(name)
    subsubtrees = unrolled_map(internal_names) do internal_name
        make_subtree_at_name(x, append_internal_name(name, internal_name))
    end
    return FieldNameTreeNode(name, subsubtrees)
end

is_valid_name(name, tree::FieldNameTreeLeaf) = name == tree.name
is_valid_name(name, tree::FieldNameTreeNode) =
    name == tree.name ||
    is_child_name(name, tree.name) &&
    unrolled_any(subtree -> is_valid_name(name, subtree), tree.subtrees)

function child_names(name, tree)
    subtree = get_subtree_at_name(name, tree)
    subtree isa FieldNameTreeNode ||
        error("FieldNameTree does not contain any child names for $name")
    return unrolled_map(subsubtree -> subsubtree.name, subtree.subtrees)
end
get_subtree_at_name(name, tree::FieldNameTreeLeaf) =
    name == tree.name ? tree :
    error("FieldNameTree does not contain the name $name")
get_subtree_at_name(name, tree::FieldNameTreeNode) =
    if name == tree.name
        tree
    elseif is_valid_name(name, tree)
        subtree_that_contains_name = unrolled_findonly(tree.subtrees) do subtree
            is_child_name(name, subtree.name)
        end
        get_subtree_at_name(name, subtree_that_contains_name)
    else
        error("FieldNameTree does not contain the name $name")
    end

################################################################################

# This is required for type-stability as of Julia 1.9.
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(has_field)
        m.recursion_relation = dont_limit
    end
    for m in methods(get_field)
        m.recursion_relation = dont_limit
    end
    for m in methods(broadcasted_has_field)
        m.recursion_relation = dont_limit
    end
    for m in methods(broadcasted_get_field)
        m.recursion_relation = dont_limit
    end
    for m in methods(wrapped_prop_names)
        m.recursion_relation = dont_limit
    end
    for m in methods(make_subtree_at_name)
        m.recursion_relation = dont_limit
    end
    for m in methods(is_valid_name)
        m.recursion_relation = dont_limit
    end
    for m in methods(get_subtree_at_name)
        m.recursion_relation = dont_limit
    end
end
