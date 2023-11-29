"""
    FieldNameDict{T1, T2}(keys, entries)
    FieldNameDict{T1, T2}(key_entry_pairs...)

An `AbstractDict` that contains keys of type `T1` and entries of type `T2`,
where the keys are stored as a `FieldNameSet{T1}`. There are four commonly used
subtypes of `FieldNameDict`:
- `FieldMatrix`, which maps a set of `FieldMatrixKeys` to either
  `ColumnwiseBandMatrixField`s or multiples of `LinearAlgebra.I`; this is the
  only user-facing subtype of `FieldNameDict`
- `FieldVectorView`, which maps a set of `FieldVectorKeys` to `Field`s; this
  subtype is automatically generated when a `FieldVector` is used in the same
  operation as a `FieldMatrix` (e.g., when both appear in the same broadcast
  expression or are passed to a `FieldMatrixSolver`)
- `FieldMatrixBroadcasted` and `FieldVectorViewBroadcasted`, which are the same
  as `FieldMatrix` and `FieldVectorView`, except that they can also store
  unevaluated broadcast expressions; these subtypes are automatically generated
  when a `FieldMatrix` or a `FieldVectorView` is used in a broadcast expression

The entry at a specific key can be extracted by calling `dict[key]`, and the
entries that correspond to all the keys in a `FieldNameSet` can be extracted by
calling `dict[set]`. If `dict` is a `FieldMatrix`, the corresponding identity
matrix can be computed by calling `one(dict)`.

When broadcasting over `FieldNameDict`s, the following operations are supported:
- Addition and subtraction
- Multiplication, where the first argument must be a `FieldMatrix` (or
  `FieldMatrixBroadcasted`)
- Inversion, where the argument must be a diagonal `FieldMatrix` (or
  `FieldMatrixBroadcasted`), i.e., one in which every entry is either a
  `ColumnwiseBandMatrixField` of `DiagonalMatrixRow`s or a multiple of
  `LinearAlgebra.I`
"""
struct FieldNameDict{T1, T2, K <: FieldNameSet{T1}, E <: NTuple{<:Any, T2}} <:
       AbstractDict{T1, T2}
    keys::K
    entries::E

    # This needs to be an inner constructor to prevent Julia from automatically
    # generating a constructor that fails Aqua.detect_unbound_args_recursively.
    FieldNameDict{T1, T2}(
        keys::FieldNameSet{T1, <:NTuple{N, T1}},
        entries::NTuple{N, T2},
    ) where {T1, T2, N} =
        new{T1, T2, typeof(keys), typeof(entries)}(keys, entries)
end
function FieldNameDict{T1, T2}(
    key_entry_pairs::Pair{<:T1, <:T2}...;
    name_tree = nothing,
) where {T1, T2}
    keys = unrolled_map(pair -> pair[1], key_entry_pairs)
    entries = unrolled_map(pair -> pair[2], key_entry_pairs)
    return FieldNameDict{T1, T2}(FieldNameSet{T1}(keys, name_tree), entries)
end
FieldNameDict{T1}(args...) where {T1} = FieldNameDict{T1, Any}(args...)

const FieldVectorView = FieldNameDict{FieldName, Fields.Field}
const FieldVectorViewBroadcasted =
    FieldNameDict{FieldName, Union{Fields.Field, Base.AbstractBroadcasted}}
const FieldMatrix = FieldNameDict{
    FieldNamePair,
    Union{UniformScaling, ColumnwiseBandMatrixField},
}
const FieldMatrixBroadcasted = FieldNameDict{
    FieldNamePair,
    Union{UniformScaling, ColumnwiseBandMatrixField, Base.AbstractBroadcasted},
}

dict_type(::FieldNameDict{T1, T2}) where {T1, T2} = FieldNameDict{T1, T2}

function Base.show(io::IO, dict::FieldNameDict)
    print(io, "$(dict_type(dict)) with $(length(dict)) entries:")
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
    unrolled_map(unrolled_zip(keys(dict).values, values(dict))) do key_entry_tup
        key_entry_tup[1] => key_entry_tup[2]
    end

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

get_internal_key(name1::FieldName, name2::FieldName) =
    extract_internal_name(name1, name2)
get_internal_key(name_pair1::FieldNamePair, name_pair2::FieldNamePair) = (
    extract_internal_name(name_pair1[1], name_pair2[1]),
    extract_internal_name(name_pair1[2], name_pair2[2]),
)

unsupported_internal_entry_error(::T, key) where {T} =
    error("Unsupported call to get_internal_entry(<$(T.name.name)>, $key)")

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
    FieldNameDictType = dict_type(dict)
    common_keys = intersect(keys(dict), new_keys)
    return FieldNameDictType(common_keys, map(key -> dict[key], common_keys))
end

function Base.similar(dict::FieldNameDict)
    FieldNameDictType = dict_type(dict)
    entries = unrolled_map(values(dict)) do entry
        entry isa UniformScaling ? entry : similar(entry)
    end
    return FieldNameDictType(keys(dict), entries)
end

# Note: This assumes that the matrix has the same row and column units, since I
# cannot be multiplied by anything other than a scalar.
function Base.one(matrix::FieldMatrix)
    diagonal_keys = matrix_diagonal_keys(keys(matrix))
    return FieldMatrix(diagonal_keys, map(_ -> I, diagonal_keys))
end

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
    lazy_main_diagonal(matrix)

Creates an un-materialized `FieldMatrixBroadcasted` that extracts the main
diagonal of the `FieldMatrix`/`FieldMatrixBroadcasted` `matrix`.
"""
function lazy_main_diagonal(matrix)
    diagonal_keys = matrix_diagonal_keys(keys(matrix))
    entries = map(diagonal_keys) do key
        entry = matrix[key]
        entry isa UniformScaling || eltype(entry) <: DiagonalMatrixRow ?
        entry :
        Base.Broadcast.broadcasted(row -> DiagonalMatrixRow(row[0]), entry)
    end
    return FieldMatrixBroadcasted(diagonal_keys, entries)
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
    return FieldVectorView(keys_of_fields, entries)
end
names_of_fields(x, name_tree) =
    unrolled_mapflatten(top_level_names(x)) do name
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

struct FieldMatrixStyle <: Base.Broadcast.BroadcastStyle end

const FieldMatrixStyleType =
    Union{FieldVectorViewBroadcasted, FieldMatrixBroadcasted}

const FieldVectorStyleType = Union{
    Fields.FieldVector,
    Base.Broadcast.Broadcasted{<:Fields.FieldVectorStyle},
}

Base.Broadcast.broadcastable(vector_or_matrix::FieldMatrixStyleType) =
    vector_or_matrix
Base.Broadcast.broadcastable(vector::FieldVectorView) =
    FieldVectorViewBroadcasted(keys(vector), values(vector))
Base.Broadcast.broadcastable(matrix::FieldMatrix) =
    FieldMatrixBroadcasted(keys(matrix), values(matrix))

Base.Broadcast.BroadcastStyle(::Type{<:FieldMatrixStyleType}) =
    FieldMatrixStyle()
Base.Broadcast.BroadcastStyle(::FieldMatrixStyle, ::Fields.FieldVectorStyle) =
    FieldMatrixStyle()

function field_matrix_broadcast_error(f, args...)
    arg_string(::FieldVectorViewBroadcasted) = "<vector>"
    arg_string(::FieldMatrixBroadcasted) = "<matrix>"
    arg_string(::FieldVectorStyleType) = "<FieldVector>"
    arg_string(::T) where {T} = error(
        "Unsupported FieldMatrixStyle broadcast argument type: $(T.name.name)",
    )
    args_string = join(map(arg_string, args), ", ")
    error("Unsupported FieldMatrixStyle broadcast operation: $f.($args_string)")
end

Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    f::F, # This should be restricted to a Function to avoid a method ambiguity.
    args...,
) where {F <: Function} = field_matrix_broadcast_error(f, args...)

# When a broadcast expression with + or * has more than two arguments, split it
# up into a chain of two-argument broadcast expressions. This simplifies the
# remaining methods for Base.Broadcast.broadcasted, since it allows us to assume
# that they will have at most two arguments.
Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    f::Union{typeof(+), typeof(*)},
    arg1,
    arg2,
    arg3,
    args...,
) =
    unrolled_foldl((arg1, arg2, arg3, args...)) do arg1′, arg2′
        Base.Broadcast.broadcasted(f, arg1′, arg2′)
    end

# Add support for broadcast expressions of the form dict1 .= dict2.
Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    ::typeof(identity),
    arg::FieldMatrixStyleType,
) = arg

function Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    ::typeof(zero),
    vector_or_matrix::FieldMatrixStyleType,
)
    FieldNameDictType = dict_type(vector_or_matrix)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        entry isa UniformScaling ? zero(entry) :
        Base.Broadcast.broadcasted(value -> rzero(typeof(value)), entry)
    end
    return FieldNameDictType(keys(vector_or_matrix), entries)
end

function Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    ::typeof(-),
    vector_or_matrix::FieldMatrixStyleType,
)
    FieldNameDictType = dict_type(vector_or_matrix)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        entry isa UniformScaling ? -entry : Base.Broadcast.broadcasted(-, entry)
    end
    return FieldNameDictType(keys(vector_or_matrix), entries)
end

function Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    f::Union{typeof(+), typeof(-)},
    vector_or_matrix1::FieldMatrixStyleType,
    vector_or_matrix2::FieldMatrixStyleType,
)
    dict_type(vector_or_matrix1) == dict_type(vector_or_matrix2) ||
        field_matrix_broadcast_error(f, vector_or_matrix1, vector_or_matrix2)
    FieldNameDictType = dict_type(vector_or_matrix1)
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
    return FieldNameDictType(all_keys, entries)
end

function Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    ::typeof(*),
    matrix::FieldMatrixBroadcasted,
    vector_or_matrix::FieldMatrixStyleType,
)
    FieldNameDictType = dict_type(vector_or_matrix)
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
    return FieldNameDictType(product_keys, entries)
end

matrix_product_argument_keys(product_name::FieldName, summand_name) =
    ((product_name, summand_name), summand_name)
matrix_product_argument_keys(product_name_pair::FieldNamePair, summand_name) =
    ((product_name_pair[1], summand_name), (summand_name, product_name_pair[2]))

function Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    ::typeof(inv),
    matrix::FieldMatrixBroadcasted,
)
    check_diagonal_matrix(
        matrix,
        "inv.(<matrix>) cannot be computed because the matrix",
    )
    entries = unrolled_map(values(matrix)) do entry
        entry isa UniformScaling ? inv(entry) :
        Base.Broadcast.broadcasted(inv, entry)
    end
    return FieldMatrixBroadcasted(keys(matrix), entries)
end

# Convert every FieldVectorStyle object to a FieldMatrixStyle object. This makes
# it possible to directly use a FieldVector in the same broadcast expression as
# a FieldMatrix, without needing to convert it to a FieldVectorView first.
Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    f::F,
    arg::FieldVectorStyleType,
) where {F <: Function} =
    Base.Broadcast.broadcasted(f, convert_to_field_matrix_style(arg))
Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    f::F,
    arg1::FieldVectorStyleType,
    arg2,
) where {F <: Function} =
    Base.Broadcast.broadcasted(f, convert_to_field_matrix_style(arg1), arg2)
Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    f::F,
    arg1,
    arg2::FieldVectorStyleType,
) where {F <: Function} =
    Base.Broadcast.broadcasted(f, arg1, convert_to_field_matrix_style(arg2))
Base.Broadcast.broadcasted(
    ::FieldMatrixStyle,
    f::F,
    arg1::FieldVectorStyleType,
    arg2::FieldVectorStyleType,
) where {F <: Function} = Base.Broadcast.broadcasted(
    f,
    convert_to_field_matrix_style(arg1),
    convert_to_field_matrix_style(arg2),
)

convert_to_field_matrix_style(x::Fields.FieldVector) = field_vector_view(x)
convert_to_field_matrix_style(
    bc::Base.Broadcast.Broadcasted{<:Fields.FieldVectorStyle},
) = Base.broadcast.broadcasted(FieldMatrixStyle(), bc.f, bc.args...)

################################################################################

materialized_dict_type(::FieldVectorViewBroadcasted) = FieldVectorView
materialized_dict_type(::FieldMatrixBroadcasted) = FieldMatrix

function Base.Broadcast.materialize(vector_or_matrix::FieldMatrixStyleType)
    FieldNameDictType = materialized_dict_type(vector_or_matrix)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        Base.Broadcast.materialize(entry)
    end
    return FieldNameDictType(keys(vector_or_matrix), entries)
end

Base.Broadcast.materialize!(
    dest::Fields.FieldVector,
    vector_or_matrix::FieldMatrixStyleType,
) = Base.Broadcast.materialize!(field_vector_view(dest), vector_or_matrix)
function Base.Broadcast.materialize!(
    dest::Union{FieldVectorView, FieldMatrix},
    vector_or_matrix::FieldMatrixStyleType,
)
    FieldNameDictType = materialized_dict_type(vector_or_matrix)
    dest isa FieldNameDictType ||
        error("Broadcast result and destination types are incompatible:
               $FieldNameDictType vs. $(typeof(dest).name.name)")
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
            Base.Broadcast.materialize!(dest[key], entry)
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
