"""
    FieldNameDict(keys, entries)
    FieldNameDict{T}(key_entry_pairs...)

An `AbstractDict` with keys of type `T` that are stored as a `FieldNameSet{T}`.
There are two subtypes of `FieldNameDict`:
- `FieldMatrix`, which maps a set of `FieldMatrixKeys` to either
  `ColumnwiseBandMatrixField`s or multiples of `LinearAlgebra.I`; this is the
  only user-facing subtype of `FieldNameDict`
- `FieldVectorView`, which maps a set of `FieldVectorKeys` to `Field`s; this
  subtype is automatically generated when a `FieldVector` is used in the same
  operation as a `FieldMatrix` (e.g., when both appear in the same broadcast
  expression, or when both are passed to a `FieldMatrixSolver`)

A `FieldNameDict` can also be "lazy", which means that it can store
`AbstractBroadcasted` objects that become `Field`s when they are materialized.
Many internal operations generate lazy `FieldNameDict`s to reduce the number of
calls to `materialize!`, since each call comes with a small performance penalty.

The entry at a specific key can be extracted by calling `dict[key]`, and the
entries that correspond to all the keys in a `FieldNameSet` can be extracted by
calling `dict[set]`. If `dict` is a `FieldMatrix`, the corresponding identity
matrix can be computed by calling `one(dict)`.

When broadcasting over `FieldNameDict`s, the following operations are supported:
- Addition and subtraction
- Multiplication, where the first argument must be a `FieldMatrix`
- Inversion, where the argument must be a diagonal `FieldMatrix`, i.e., one in
  which every entry is either a `ColumnwiseBandMatrixField` of
  `DiagonalMatrixRow`s or a multiple of `LinearAlgebra.I`
"""
struct FieldNameDict{
    T <: Union{FieldName, FieldNamePair},
    K <: FieldNameSet{T},
    E,
} <: AbstractDict{T, Any}
    keys::K
    entries::E

    # This needs to be an inner constructor to prevent Julia from automatically
    # generating a constructor that fails Aqua.detect_unbound_args_recursively.
    function FieldNameDict(keys::FieldNameSet{T}, entries) where {T}
        length(keys) == length(entries) || error(
            "FieldNameDict cannot have different numbers of keys and entries",
        )
        unrolled_foreach(entries) do entry
            check_entry(T, entry) ||
                error("Invalid $(FieldNameDict{T}) entry: $entry")
        end
        return new{T, typeof(keys), typeof(entries)}(keys, entries)
    end
end
function FieldNameDict{T}(key_entry_pairs::Pair{<:T}...) where {T}
    keys = unrolled_map(pair -> pair[1], key_entry_pairs)
    entries = unrolled_map(pair -> pair[2], key_entry_pairs)
    return FieldNameDict(FieldNameSet{T}(keys), entries)
end

const FieldVectorView = FieldNameDict{FieldName}
const FieldMatrix = FieldNameDict{FieldNamePair}

check_entry(_, _) = false
check_entry(::Type{FieldName}, ::Fields.Field) = true
check_entry(::Type{FieldNamePair}, ::UniformScaling) = true
check_entry(::Type{FieldNamePair}, ::ColumnwiseBandMatrixField) = true
check_entry(_, entry::Base.AbstractBroadcasted) =
    Base.Broadcast.BroadcastStyle(typeof(entry)) isa Fields.AbstractFieldStyle

function Base.show(io::IO, dict::FieldNameDict)
    T = eltype(keys(dict))
    print(io, "$(FieldNameDict{T}) with $(length(dict)) entries:")
    for (key, entry) in dict
        print(io, "\n  $key => ")
        if entry isa Fields.Field
            print(io, eltype(entry), "-valued Field:")
            Fields._show_compact_field(io, entry, "    ", true)
        elseif entry isa UniformScaling
            if entry.λ == 1
                print(io, "I")
            elseif entry.λ == -1
                print(io, "-I")
            else
                print(io, "$(entry.λ) * I")
            end
        else
            print(io, entry)
        end
    end
end

Base.keys(dict::FieldNameDict) = dict.keys

Base.values(dict::FieldNameDict) = dict.entries

Base.pairs(dict::FieldNameDict) =
    unrolled_map((key, value) -> key => value, keys(dict).values, values(dict))

Base.length(dict::FieldNameDict) = length(keys(dict))

Base.iterate(dict::FieldNameDict, index = 1) = iterate(pairs(dict), index)

Base.:(==)(dict1::FieldNameDict, dict2::FieldNameDict) =
    keys(dict1) == keys(dict2) && values(dict1) == values(dict2)

function Base.getindex(dict::FieldNameDict, key)
    key in keys(dict) || throw(KeyError(key))
    key′, entry′ =
        unrolled_findonly(pair -> is_child_value(key, pair[1]), pairs(dict))
    return get_internal_entry(entry′, get_internal_key(key, key′))
end

get_internal_key(child_name::FieldName, name::FieldName) =
    extract_internal_name(child_name, name)
get_internal_key(child_name_pair::FieldNamePair, name_pair::FieldNamePair) = (
    extract_internal_name(child_name_pair[1], name_pair[1]),
    extract_internal_name(child_name_pair[2], name_pair[2]),
)

unsupported_internal_entry_error(entry, key) =
    error("Unsupported FieldNameDict operation: \
           get_internal_entry(<$(typeof(entry).name.name)>, $key)")

get_internal_entry(entry, name::FieldName) = get_field(entry, name)
get_internal_entry(entry, name_pair::FieldNamePair) =
    name_pair == (@name(), @name()) ? entry :
    unsupported_internal_entry_error(entry, name_pair)
get_internal_entry(entry::UniformScaling, name_pair::FieldNamePair) =
    name_pair[1] == name_pair[2] ? entry :
    unsupported_internal_entry_error(entry, name_pair)
function get_internal_entry(
    entry::ColumnwiseBandMatrixField,
    name_pair::FieldNamePair,
)
    # Ensure compatibility with RecursiveApply (i.e., with rmul).
    # See note above matrix_product_keys in field_name_set.jl for more details.
    T = eltype(eltype(entry))
    if name_pair == (@name(), @name())
        entry
    elseif name_pair[1] == name_pair[2]
        # multiplication case 3 or 4, first argument
        @assert T <: SingleValue && !broadcasted_has_field(T, name_pair[1])
        entry
    elseif name_pair[2] == @name() && broadcasted_has_field(T, name_pair[1])
        # multiplication case 2 or 4, second argument
        Base.broadcasted(entry) do matrix_row
            map(matrix_row) do matrix_row_entry
                broadcasted_get_field(matrix_row_entry, name_pair[1])
            end
        end # Note: This assumes that the entry is in a FieldMatrixBroadcasted.
    else
        unsupported_internal_entry_error(entry, name_pair)
    end
end

# Similar behavior to indexing an array with a slice.
function Base.getindex(dict::FieldNameDict, new_keys::FieldNameSet)
    common_keys = intersect(keys(dict), new_keys)
    return FieldNameDict(common_keys, map(key -> dict[key], common_keys))
end

function Base.similar(dict::FieldNameDict)
    entries = unrolled_map(values(dict)) do entry
        entry isa UniformScaling ? entry : similar(entry)
    end
    return FieldNameDict(keys(dict), entries)
end

# Note: This assumes that the matrix has the same row and column units, since I
# cannot be multiplied by anything other than a scalar.
function Base.one(matrix::FieldMatrix)
    diagonal_keys = matrix_diagonal_keys(keys(matrix))
    return FieldNameDict(diagonal_keys, map(_ -> I, diagonal_keys))
end

replace_name_tree(dict::FieldNameDict, name_tree) =
    FieldNameDict(replace_name_tree(keys(dict), name_tree), values(dict))

function check_block_diagonal_matrix(matrix, error_message_start = "The matrix")
    off_diagonal_keys = matrix_off_diagonal_keys(keys(matrix))
    isempty(off_diagonal_keys) || error(
        "$error_message_start has entries at the following off-diagonal keys: \
         $(set_string(off_diagonal_keys))",
    )
end

function check_diagonal_matrix(matrix, error_message_start = "The matrix")
    check_block_diagonal_matrix(matrix, error_message_start)
    non_diagonal_entry_pairs = unrolled_filter(pairs(matrix)) do pair
        !(pair[2] isa UniformScaling || eltype(pair[2]) <: DiagonalMatrixRow)
    end
    non_diagonal_entry_keys =
        FieldMatrixKeys(unrolled_map(pair -> pair[1], non_diagonal_entry_pairs))
    isempty(non_diagonal_entry_keys) || error(
        "$error_message_start has non-diagonal entries at the following keys: \
         $(set_string(non_diagonal_entry_keys))",
    )
end

"""
    is_lazy(dict)

Checks whether the `FieldNameDict` `dict` contains any un-materialized
`AbstractBroadcasted` entries.
"""
is_lazy(dict) =
    unrolled_any(entry -> entry isa Base.AbstractBroadcasted, values(dict))

"""
    lazy_main_diagonal(matrix)

Constructs a lazy `FieldMatrix` that contains the main diagonal of `matrix`.
"""
function lazy_main_diagonal(matrix)
    diagonal_keys = matrix_diagonal_keys(keys(matrix))
    entries = map(diagonal_keys) do key
        entry = matrix[key]
        entry isa UniformScaling || eltype(entry) <: DiagonalMatrixRow ?
        entry :
        Base.Broadcast.broadcasted(row -> DiagonalMatrixRow(row[0]), entry)
    end
    return FieldNameDict(diagonal_keys, entries)
end

"""
    field_vector_view(x, [name_tree])

Constructs a `FieldVectorView` that contains all of the `Field`s in the
`FieldVector` `x`. The default `name_tree` is `FieldNameTree(x)`, but this can
be modified if needed.
"""
function field_vector_view(x, name_tree = FieldNameTree(x))
    keys_of_fields = FieldVectorKeys(names_of_fields(x, name_tree), name_tree)
    entries = map(name -> get_field(x, name), keys_of_fields)
    return FieldNameDict(keys_of_fields, entries)
end
names_of_fields(x, name_tree) =
    unrolled_flatmap(top_level_names(x)) do name
        entry = get_field(x, name)
        if entry isa Fields.Field
            (name,)
        elseif entry isa Fields.FieldVector
            unrolled_map(names_of_fields(entry, name_tree)) do internal_name
                append_internal_name(name, internal_name)
            end
        else
            error("field_vector_view does not support entries of type \
                   $(typeof(entry).name.name)")
        end
    end

"""
    concrete_field_vector(vector)

Converts the `FieldVectorView` `vector` back into a `FieldVector`.
"""
concrete_field_vector(vector) =
    concrete_field_vector_within_subtree(keys(vector).name_tree, vector)
concrete_field_vector_within_subtree(tree, vector) =
    if tree.name in keys(vector)
        vector[tree.name]
    else
        subtrees = unrolled_filter(tree.subtrees) do subtree
            unrolled_any(keys(vector).values) do key
                is_child_name(key, subtree.name)
            end
        end
        internal_names = unrolled_map(subtrees) do subtree
            extract_first(extract_internal_name(subtree.name, tree.name))
        end
        internal_entries = unrolled_map(subtrees) do subtree
            concrete_field_vector_within_subtree(subtree, vector)
        end
        entry_eltypes = unrolled_map(recursive_bottom_eltype, internal_entries)
        T = promote_type(entry_eltypes...)
        Fields.FieldVector{T}(NamedTuple{internal_names}(internal_entries))
    end

# This is required for type-stability as of Julia 1.9.
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(names_of_fields)
        m.recursion_relation = dont_limit
    end
    for m in methods(concrete_field_vector_within_subtree)
        m.recursion_relation = dont_limit
    end
end

################################################################################

struct FieldNameDictStyle <: Base.Broadcast.BroadcastStyle end

const FieldVectorStyleType = Union{
    Fields.FieldVector,
    Base.Broadcast.Broadcasted{<:Fields.FieldVectorStyle},
}

Base.Broadcast.broadcastable(vector_or_matrix::FieldNameDict) = vector_or_matrix

Base.Broadcast.BroadcastStyle(::Type{<:FieldNameDict}) = FieldNameDictStyle()
Base.Broadcast.BroadcastStyle(::FieldNameDictStyle, ::Fields.FieldVectorStyle) =
    FieldNameDictStyle()

function field_matrix_broadcast_error(f, args...)
    arg_string(::FieldVectorView) = "<vector>"
    arg_string(::FieldMatrix) = "<matrix>"
    arg_string(::FieldVectorStyleType) = "<FieldVector>"
    arg_string(::T) where {T} = error(
        "Unsupported FieldNameDict broadcast argument type: $(T.name.name)",
    )
    args_string = join(map(arg_string, args), ", ")
    error("Unsupported FieldNameDict broadcast operation: $f.($args_string)")
end

Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    f::F, # This should be restricted to a Function to avoid a method ambiguity.
    args...,
) where {F <: Function} = field_matrix_broadcast_error(f, args...)

# When a broadcast expression with + or * has more than two arguments, split it
# up into a chain of two-argument broadcast expressions. This simplifies the
# remaining methods for Base.Broadcast.broadcasted, since it allows us to assume
# that they will have at most two arguments.
Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    f::Union{typeof(+), typeof(*)},
    arg1,
    arg2,
    arg3,
    args...,
) =
    foldl((arg1, arg2, arg3, args...)) do arg1′, arg2′
        Base.Broadcast.broadcasted(f, arg1′, arg2′)
    end

# Add support for broadcast expressions of the form dict1 .= dict2.
Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    ::typeof(identity),
    arg::FieldNameDict,
) = arg

function Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    ::typeof(zero),
    vector_or_matrix::FieldNameDict,
)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        entry isa UniformScaling ? zero(entry) :
        Base.Broadcast.broadcasted(value -> rzero(typeof(value)), entry)
    end
    return FieldNameDict(keys(vector_or_matrix), entries)
end

function Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    ::typeof(-),
    vector_or_matrix::FieldNameDict,
)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        entry isa UniformScaling ? -entry : Base.Broadcast.broadcasted(-, entry)
    end
    return FieldNameDict(keys(vector_or_matrix), entries)
end

function Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    f::Union{typeof(+), typeof(-)},
    vector_or_matrix1::FieldNameDict,
    vector_or_matrix2::FieldNameDict,
)
    all_keys = union(keys(vector_or_matrix1), keys(vector_or_matrix2))
    entries = map(all_keys) do key
        if key in intersect(keys(vector_or_matrix1), keys(vector_or_matrix2))
            entry1 = vector_or_matrix1[key]
            entry2 = vector_or_matrix2[key]
            if entry1 isa UniformScaling && entry2 isa UniformScaling
                f(entry1, entry2)
            elseif entry1 isa UniformScaling
                Base.Broadcast.broadcasted(f, (entry1,), entry2)
            elseif entry2 isa UniformScaling
                Base.Broadcast.broadcasted(f, entry1, (entry2,))
            else
                Base.Broadcast.broadcasted(f, entry1, entry2)
            end
        elseif key in keys(vector_or_matrix1)
            vector_or_matrix1[key]
        else
            if f isa typeof(+)
                vector_or_matrix2[key]
            else
                entry = vector_or_matrix2[key]
                entry isa UniformScaling ? -entry :
                Base.Broadcast.broadcasted(-, entry)
            end
        end
    end
    return FieldNameDict(all_keys, entries)
end

function Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    ::typeof(*),
    matrix::FieldMatrix,
    vector_or_matrix::FieldNameDict,
)
    product_keys = matrix_product_keys(keys(matrix), keys(vector_or_matrix))
    entries = map(product_keys) do product_key
        summand_names = summand_names_for_matrix_product(
            product_key,
            keys(matrix),
            keys(vector_or_matrix),
        )
        summand_bcs = map(summand_names) do summand_name
            key1, key2 = matrix_product_argument_keys(product_key, summand_name)
            entry1 = matrix[key1]
            entry2 = vector_or_matrix[key2]
            if entry1 isa UniformScaling && entry2 isa UniformScaling
                entry1 * entry2
            elseif entry1 isa UniformScaling
                Base.Broadcast.broadcasted(*, entry1.λ, entry2)
            elseif entry2 isa UniformScaling
                Base.Broadcast.broadcasted(*, entry1, entry2.λ)
            else
                Base.Broadcast.broadcasted(⋅, entry1, entry2)
            end
        end
        length(summand_bcs) == 1 ? summand_bcs[1] :
        Base.Broadcast.broadcasted(+, summand_bcs...)
    end
    return FieldNameDict(product_keys, entries)
end

matrix_product_argument_keys(product_name::FieldName, summand_name) =
    ((product_name, summand_name), summand_name)
matrix_product_argument_keys(product_name_pair::FieldNamePair, summand_name) =
    ((product_name_pair[1], summand_name), (summand_name, product_name_pair[2]))

function Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    ::typeof(inv),
    matrix::FieldMatrix,
)
    check_diagonal_matrix(
        matrix,
        "inv.(<matrix>) cannot be computed because the matrix",
    )
    entries = unrolled_map(values(matrix)) do entry
        entry isa UniformScaling ? inv(entry) :
        Base.Broadcast.broadcasted(inv, entry)
    end
    return FieldNameDict(keys(matrix), entries)
end

# Convert every FieldVectorStyle object to a FieldNameDict. This makes it
# possible to directly use a FieldVector in the same broadcast expression as a
# FieldMatrix, without needing to convert it to a FieldVectorView first.
Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    f::F,
    arg::FieldVectorStyleType,
) where {F <: Function} =
    Base.Broadcast.broadcasted(f, convert_to_field_name_dict(arg))
Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    f::F,
    arg1::FieldVectorStyleType,
    arg2,
) where {F <: Function} =
    Base.Broadcast.broadcasted(f, convert_to_field_name_dict(arg1), arg2)
Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    f::F,
    arg1,
    arg2::FieldVectorStyleType,
) where {F <: Function} =
    Base.Broadcast.broadcasted(f, arg1, convert_to_field_name_dict(arg2))
Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    f::F,
    arg1::FieldVectorStyleType,
    arg2::FieldVectorStyleType,
) where {F <: Function} = Base.Broadcast.broadcasted(
    f,
    convert_to_field_name_dict(arg1),
    convert_to_field_name_dict(arg2),
)

convert_to_field_name_dict(x::Fields.FieldVector) = field_vector_view(x)
convert_to_field_name_dict(
    bc::Base.Broadcast.Broadcasted{<:Fields.FieldVectorStyle},
) = Base.broadcast.broadcasted(FieldNameDictStyle(), bc.f, bc.args...)

################################################################################

function Base.Broadcast.materialize(vector_or_matrix::FieldNameDict)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        Base.Broadcast.materialize(entry)
    end
    return FieldNameDict(keys(vector_or_matrix), entries)
end

Base.Broadcast.materialize!(
    dest::Fields.FieldVector,
    vector_or_matrix::FieldNameDict,
) = Base.Broadcast.materialize!(field_vector_view(dest), vector_or_matrix)
function Base.Broadcast.materialize!(
    dest::FieldNameDict,
    vector_or_matrix::FieldNameDict,
)
    !is_lazy(dest) || error("Cannot materialize into a lazy FieldNameDict")
    is_subset_that_covers_set(keys(vector_or_matrix), keys(dest)) || error(
        "Broadcast result and destination keys are incompatible: \
         $(set_string(keys(vector_or_matrix))) vs. $(set_string(keys(dest)))",
    ) # It is not always the case that keys(vector_or_matrix) == keys(dest).
    foreach(keys(vector_or_matrix)) do key
        entry = vector_or_matrix[key]
        if dest[key] isa UniformScaling
            dest[key] == entry || error("UniformScaling is immutable")
        elseif entry isa UniformScaling
            dest[key] .= (entry,)
        else
            dest[key] .= entry
        end
    end
end

#=
For debugging, uncomment the function below and put the following lines into the
loop in materialize!:
    println()
    println(key)
    println(summary_string(vector_or_matrix[key]))
    println(dest[key])
    println()

summary_string(entry) = summary_string(entry, 0)
summary_string(entry, indent_level) = "$("    "^indent_level)$entry"
function summary_string(field::Fields.Field, indent_level)
    staggering_string =
        hasproperty(axes(field), :staggering) ?
        string(typeof(axes(field).staggering).name.name) : "Single Level"
    return "$("    "^indent_level)Field{$(eltype(field)), $staggering_string}"
end
function summary_string(bc::Base.AbstractBroadcasted, indent_level)
    func = bc isa Operators.OperatorBroadcasted ? bc.op : bc.f
    arg_strings = map(arg -> summary_string(arg, indent_level + 1), bc.args)
    tab = "    "^indent_level
    return "$(tab)Broadcasted{$func}(\n$(join(arg_strings, ",\n")),\n$tab)"
end
=#
