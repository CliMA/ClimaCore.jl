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
    keys = unrolled_map(first, key_entry_pairs)
    entries = unrolled_map(last, key_entry_pairs)
    return FieldNameDict(FieldNameSet{T}(keys), entries)
end

const FieldVectorView = FieldNameDict{FieldName}
const FieldMatrix = FieldNameDict{FieldNamePair}

const ScalingFieldMatrixEntry{T} =
    Union{UniformScaling{T}, DiagonalMatrixRow{T}}

scaling_value(entry::UniformScaling) = entry.λ
scaling_value(entry::DiagonalMatrixRow) = entry[0]

check_entry(_, _) = false
check_entry(::Type{FieldName}, ::Fields.Field) = true
check_entry(::Type{FieldNamePair}, ::ScalingFieldMatrixEntry) = true
check_entry(::Type{FieldNamePair}, ::ColumnwiseBandMatrixField) = true

is_field_broadcasted(bc) =
    Base.Broadcast.BroadcastStyle(typeof(bc)) isa Fields.AbstractFieldStyle
check_entry(::Type{FieldName}, entry::Base.AbstractBroadcasted) =
    is_field_broadcasted(entry)
check_entry(::Type{FieldNamePair}, entry::Base.AbstractBroadcasted) =
    is_field_broadcasted(entry) && eltype(entry) <: BandMatrixRow

is_diagonal_matrix_entry(::ScalingFieldMatrixEntry) = true
is_diagonal_matrix_entry(entry) = eltype(entry) <: DiagonalMatrixRow

function Base.show(io::IO, dict::FieldNameDict)
    T = eltype(keys(dict))
    print(io, "$(FieldNameDict{T}) with $(length(dict)) entries:")
    for (key, entry) in dict
        print(io, "\n  $key => ")
        if entry isa Fields.Field
            print(io, eltype(entry), "-valued Field:")
            Fields._show_compact_field(io, entry, "    ", true)
        elseif entry isa ScalingFieldMatrixEntry
            if scaling_value(entry) == 1
                print(io, "I")
            elseif scaling_value(entry) == -1
                print(io, "-I")
            else
                print(io, "$(scaling_value(entry)) * I")
            end
        else
            print(io, entry)
        end
    end
end

function Operators.strip_space(dict::FieldNameDict)
    vals = unrolled_map(values(dict)) do val
        if val isa Fields.Field
            Fields.Field(Fields.field_values(val), Operators.PlaceholderSpace())
        else
            val
        end
    end
    FieldNameDict(keys(dict), vals)
end

function Adapt.adapt_structure(to, dict::FieldNameDict)
    vals = unrolled_map(v -> Adapt.adapt_structure(to, v), values(dict))
    FieldNameDict(keys(dict), vals)
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
        unrolled_filter(pair -> is_child_value(key, pair[1]), pairs(dict))[1]
    internal_key = get_internal_key(key, key′)
    return get_internal_entry(entry′, internal_key)
end

get_internal_key(child_name::FieldName, name::FieldName) =
    extract_internal_name(child_name, name)
get_internal_key(child_name_pair::FieldNamePair, name_pair::FieldNamePair) = (
    extract_internal_name(child_name_pair[1], name_pair[1]),
    extract_internal_name(child_name_pair[2], name_pair[2]),
)
"""
    get_internal_entry(entry, name::FieldName)

Returns the field indexed to by `name` from `entry`
"""
get_internal_entry(entry, name::FieldName) = get_field(entry, name)
# call get_internal_entry on scaling value, and rebuild entry container
"""
    get_internal_entry(entry, name_pair::FieldNamePair)

Returns the field indexed to by `name_pair` from `entry`. Indexing behavior is described
in the MatrixFields section of the documentation. If `entry` is a `ColumnwiseBandMatrixField`,
and the field indexed to by `name_pair` is not a field of scalars, a broadcasted object
is returned. This also happens when indexing off diagonal with the implicit tensor structure
optimization (see MatrixFields documentation).
"""
get_internal_entry(entry::UniformScaling, name_pair::FieldNamePair) =
    UniformScaling(get_internal_entry(scaling_value(entry), name_pair))
get_internal_entry(entry::DiagonalMatrixRow, name_pair::FieldNamePair) =
    DiagonalMatrixRow(get_internal_entry(scaling_value(entry), name_pair))
get_internal_entry(entry, name_pair) =
    get_internal_entry(entry, name_pair, name_pair)
# get_internal_entry to be used on the values held inside a `BandMatrixRow`
function get_internal_entry(
    entry::T,
    name_pair::FieldNamePair,
    full_key::FieldNamePair,
) where {T}
    if name_pair == (@name(), @name())
        return entry
    elseif T <: Geometry.Axis2Tensor &&
           all(n -> is_child_name(n, @name(components.data)), name_pair)
        # two indices needed to index into a 2d tensor (one can be Colon())
        internal_row_name =
            extract_internal_name(name_pair[1], @name(components.data))
        internal_col_name =
            extract_internal_name(name_pair[2], @name(components.data))
        row_index = extract_first(internal_row_name)
        col_index = extract_first(internal_col_name)
        return get_internal_entry(
            entry[row_index, col_index],
            (drop_first(internal_row_name), drop_first(internal_col_name)),
            full_key,
        )
    elseif T <: Geometry.Axis2Tensor && # slicing a 2d tensor
           is_child_name(name_pair[1], @name(components.data))
        internal_row_name =
            extract_internal_name(name_pair[1], @name(components.data))
        return get_internal_entry(
            entry[extract_first(internal_row_name), :],
            (drop_first(internal_row_name), name_pair[2]),
            full_key,
        )
    elseif T <: Geometry.Axis2Tensor && # slicing a 2d tensor
           is_child_name(name_pair[2], @name(components.data))
        internal_col_name =
            extract_internal_name(name_pair[2], @name(components.data))
        return get_internal_entry(
            entry[:, extract_first(internal_col_name)],
            (name_pair[1], drop_first(internal_col_name)),
            full_key,
        )
    elseif T <: Geometry.AdjointAxisVector # bypass parent for adjoint vectors
        return get_internal_entry(getfield(entry, :parent), name_pair, full_key)
    elseif name_pair[1] != @name() &&
           extract_first(name_pair[1]) in fieldnames(T)
        return get_internal_entry(
            getfield(entry, extract_first(name_pair[1])),
            (drop_first(name_pair[1]), name_pair[2]),
            full_key,
        )
    elseif name_pair[2] != @name() &&
           extract_first(name_pair[2]) in fieldnames(T)
        return get_internal_entry(
            getfield(entry, extract_first(name_pair[2])),
            (name_pair[1], drop_first(name_pair[2])),
            full_key,
        )
    elseif !any(isequal(@name()), name_pair) # implicit tensor structure
        return get_internal_entry(
            extract_first(name_pair[1]) == extract_first(name_pair[2]) ? entry :
            zero(entry),
            (drop_first(name_pair[1]), drop_first(name_pair[2])),
            full_key,
        )
    else
        throw(KeyError(full_key))
    end
end
function get_internal_entry(
    entry::ColumnwiseBandMatrixField,
    name_pair::FieldNamePair,
)
    name_pair == (@name(), @name()) && return entry
    S = eltype(eltype(entry))
    T = eltype(parent(entry))
    (start_offset, target_type, index_method) =
        field_offset_and_type(name_pair, T, S, name_pair)
    if isa(index_method, Val{:view})
        @assert target_type <: T
        band_element_size = DataLayouts.typesize(T, S)
        singleton_datalayout = DataLayouts.singleton(Fields.field_values(entry))
        scalar_band_type =
            band_matrix_row_type(outer_diagonals(eltype(entry))..., target_type)
        field_dim_size = DataLayouts.ncomponents(Fields.field_values(entry))
        parent_indices = DataLayouts.to_data_specific_field(
            singleton_datalayout,
            (:, :, (start_offset + 1):band_element_size:field_dim_size, :, :),
        )
        scalar_data = view(parent(entry), parent_indices...)
        values = DataLayouts.union_all(singleton_datalayout){
            scalar_band_type,
            Base.tail(DataLayouts.type_params(Fields.field_values(entry)))...,
        }(
            scalar_data,
        )
        return Fields.Field(values, axes(entry))
    elseif isa(index_method, Val{:broadcasted_zero})
        # implicit tensor structure optimization, off diagonal
        zero_value = zero(target_type)
        return Base.broadcasted(entry) do matrix_row
            map(x -> zero_value, matrix_row)
        end
    elseif target_type == S && isa(index_method, Val{:view_of_blocks})
        return entry
    else # fallback to broadcasted indexing on each element, currently no support for view_of_blocks
        return Base.broadcasted(entry) do matrix_row
            map(matrix_row) do matrix_row_entry
                get_internal_entry(
                    disable_auto_broadcasting(matrix_row_entry),
                    name_pair,
                )
            end
        end
    end
end
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(get_internal_entry)
        m.recursion_relation = dont_limit
    end
end

# Similar behavior to indexing an array with a slice.
function Base.getindex(dict::FieldNameDict, new_keys::FieldNameSet)
    common_keys = intersect(keys(dict), new_keys)
    return FieldNameDict(common_keys, map(key -> dict[key], common_keys))
end

function Base.similar(dict::FieldNameDict)
    entries = unrolled_map(values(dict)) do entry
        entry isa ScalingFieldMatrixEntry ? entry : similar(entry)
    end
    return FieldNameDict(keys(dict), entries)
end

# TODO: The behavior of this method is extremely counterintuitive---it is
# zeroing out mutable values, but leaving nonzero immutable values unchanged.
# We should probably use a different function name for this method.
function Base.zero(dict::FieldNameDict)
    entries = unrolled_map(values(dict)) do entry
        entry isa ScalingFieldMatrixEntry ? entry : zero(entry)
    end
    return FieldNameDict(keys(dict), entries)
end

function Base.one(matrix::FieldMatrix)
    inferred_diagonal_keys = matrix_inferred_diagonal_keys(keys(matrix))
    entries = map(inferred_diagonal_keys) do key
        if !(key in keys(matrix))
            I # default value for missing diagonal entries in a sparse matrix
        else
            # TODO: Add method for one(::Axis2Tensor) to simplify this.
            T =
                matrix[key] isa ScalingFieldMatrixEntry ?
                eltype(matrix[key]) : eltype(eltype(matrix[key]))
            if T <: Number
                UniformScaling(one(T))
            elseif T <: Geometry.Axis2Tensor
                tensor_data = UniformScaling(one(eltype(T)))
                DiagonalMatrixRow(Geometry.AxisTensor(axes(T), tensor_data))
            else
                error("Unsupported diagonal FieldMatrix entry type: $T")
            end
        end
    end
    return FieldNameDict(inferred_diagonal_keys, entries)
end

"""
    field_offset_and_type(name_pair::FieldNamePair, ::Type{T}, ::Type{S}, full_key::FieldNamePair)

Returns the offset of the field with name `name_pair` in an object of type `S` in
multiples of `sizeof(T)`, the type of the field with name `name_pair`, and a `Val` indicating
what method can index a ClimaCore `Field` of `S` with `name_pair`.

The third return value is one of the following:
- `Val(:view)`: indexing with a view is possible
-  `Val(:view_of_blocks)`: indexing with a view of non-unfiform stride length is possible.\
 This is not implemented, and currently treated the same as `Val(:broadcasted_fallback)`
- `Val(:broadcasted_fallback)`: indexing with a view is not possible
- `Val(:broadcasted_zero)`: indexing with a view is not possible, and the `name_pair` indexes
off diagonal with implicit tensor structure optimization (see MatrixFields docs)

When `S` is a `Geometry.Axis2Tensor`, and the name pair indexes to a slice of
the tensor, an offset of `-1` is returned . In other words, the name pair cannot index into a slice.

If neither element of `name_pair` is `@name()`, the first name in the pair is indexed with
first, and then the second name is used to index the result of the first.

This is an internal funtion designed to be used with `get_internal_entry(::ColumnwiseBandMatrixField)`
"""
function field_offset_and_type(
    name_pair::FieldNamePair,
    ::Type{T},
    ::Type{S},
    full_key::FieldNamePair,
) where {S, T}
    if name_pair == (@name(), @name()) # recursion base case
        # if S <: T, then its possible to construct a strided view in the indexing function
        return (0, S, S <: T ? Val(:view) : Val(:view_of_blocks))
    elseif S <: Geometry.Axis2Tensor &&
           any(n -> is_child_name(n, @name(components.data)), name_pair) # special case to calculate index
        all(n -> is_child_name(n, @name(components.data)), name_pair) ||
            return (0, S, Val{:broadcasted_fallback}())
        internal_row_name =
            extract_internal_name(name_pair[1], @name(components.data))
        internal_col_name =
            extract_internal_name(name_pair[2], @name(components.data))
        row_index = extract_first(internal_row_name)
        col_index = extract_first(internal_col_name)
        ((row_index isa Number) && (col_index isa Number)) ||
            throw(KeyError(full_key))
        (n_rows, n_cols) = map(length, axes(S))
        (remaining_offset, end_type, index_method) = field_offset_and_type(
            (drop_first(internal_row_name), drop_first(internal_col_name)),
            T,
            eltype(S),
            full_key,
        )
        (row_index <= n_rows && col_index <= n_cols) ||
            throw(KeyError(full_key))
        return (
            (n_rows * (col_index - 1) + row_index - 1) + remaining_offset,
            end_type,
            index_method,
        )
    elseif S <: Geometry.AdjointAxisVector # bypass adjoint because indexing parent is equivalent
        return field_offset_and_type(name_pair, T, fieldtype(S, 1), full_key)
    elseif name_pair[1] != @name() &&
           extract_first(name_pair[1]) in fieldnames(S) # index with first part of name_pair[1]
        remaining_field_chain = (drop_first(name_pair[1]), name_pair[2])
        child_type = fieldtype(S, extract_first(name_pair[1]))
        field_index = unrolled_filter(
            i -> fieldname(S, i) == extract_first(name_pair[1]),
            1:fieldcount(S),
        )[1]
        (remaining_offset, end_type, index_method) = field_offset_and_type(
            remaining_field_chain,
            T,
            child_type,
            full_key,
        )
        return (
            DataLayouts.fieldtypeoffset(T, S, field_index) + remaining_offset,
            end_type,
            index_method,
        )
    elseif name_pair[2] != @name() &&
           extract_first(name_pair[2]) in fieldnames(S) # index with first part of name_pair[2]
        remaining_field_chain = name_pair[1], drop_first(name_pair[2])
        child_type = fieldtype(S, extract_first(name_pair[2]))
        field_index = unrolled_filter(
            i -> fieldname(S, i) == extract_first(name_pair[2]),
            1:fieldcount(S),
        )[1]
        (remaining_offset, end_type, index_method) = field_offset_and_type(
            remaining_field_chain,
            T,
            child_type,
            full_key,
        )
        return (
            DataLayouts.fieldtypeoffset(T, S, field_index) + remaining_offset,
            end_type,
            index_method,
        )
    elseif !any(isequal(@name()), name_pair) # implicit tensor structure optimization
        (remaining_offset, end_type, index_method) = field_offset_and_type(
            (drop_first(name_pair[1]), drop_first(name_pair[2])),
            T,
            S,
            full_key,
        )
        return (
            remaining_offset,
            end_type,
            extract_first(name_pair[1]) == extract_first(name_pair[2]) ?
            index_method : Val(:broadcasted_zero), # zero if off diagonal
        )
    else
        throw(KeyError(full_key))
    end
end
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(field_offset_and_type)
        m.recursion_relation = dont_limit
    end
end

"""
    get_scalar_keys(dict::FieldMatrix)

Returns a `FieldMatrixKeys` object that contains the keys that result in
a `ScalingFieldMatrixEntry{<: target_type}` or a `ColumnwiseBandMatrixField` with bands of
eltype `<: target_type` when indexing `dict`. `target_type` is determined by the eltype of the
parent of the first entry in `dict` that is a `Fields.Field`. If no such entry
is found, `target_type` defaults to `Number`.
"""
function get_scalar_keys(dict::FieldMatrix)
    first_field_idx = unrolled_findfirst(Base.Fix2(isa, Fields.Field), dict.entries)
    target_type = Val(
        isnothing(first_field_idx) ? Number :
        eltype(parent(dict.entries[first_field_idx])),
    )
    keys_tuple = unrolled_flatmap(keys(dict).values) do outer_key
        unrolled_map(
            get_scalar_keys(eltype(dict[outer_key]), target_type),
        ) do inner_key
            (
                append_internal_name(outer_key[1], inner_key[1]),
                append_internal_name(outer_key[2], inner_key[2]),
            )
        end
    end
    return FieldMatrixKeys(keys_tuple, dict.keys.name_tree)
end

"""
    get_scalar_keys(T::Type, ::Val{FT})

Returns a tuple of `FieldNamePair` objects that correspond to any children
of `T` that are of type `<: FT`.
"""
function get_scalar_keys(::Type{T}, ::Val{FT}) where {T, FT}
    if T <: FT
        return ((@name(), @name()),)
    elseif T <: BandMatrixRow
        return get_scalar_keys(eltype(T), Val(FT))
    elseif T <: Geometry.Axis2Tensor
        return unrolled_flatmap(1:length(axes(T)[1])) do row_component
            unrolled_map(1:length(axes(T)[2])) do col_component
                append_internal_name.(
                    Ref(@name(components.data)),
                    (FieldName(row_component), FieldName(col_component)),
                )
            end
        end
    elseif T <: Geometry.AdjointAxisVector
        return unrolled_map(
            get_scalar_keys(fieldtype(T, :parent), Val(FT)),
        ) do inner_key
            (inner_key[2], inner_key[1]) # assumes that adjoints only appear with d/dvec
        end
    elseif T <: Geometry.AxisVector # special case to avoid recursing into the axis field
        return unrolled_map(
            get_scalar_keys(fieldtype(T, :components), Val(FT)),
        ) do inner_key
            (
                append_internal_name(@name(components), inner_key[1]),
                inner_key[2],
            )
        end
    else
        return unrolled_flatmap(fieldnames(T)) do inner_name
            unrolled_map(
                get_scalar_keys(fieldtype(T, inner_name), Val(FT)),
            ) do inner_key
                (
                    append_internal_name(FieldName(inner_name), inner_key[1]),
                    inner_key[2],
                )
            end
        end
    end
end
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(get_scalar_keys)
        m.recursion_relation = dont_limit
    end
end


"""
    scalar_field_matrix(field_matrix::FieldMatrix)

Constructs a `FieldNameDict` where the keys and entries are views
of the entries of `field_matrix`, which corresponding to the
`FT` typed components of entries of `field_matrix`.

# Example usage
```julia
e¹² = Geometry.Covariant12Vector(1.6, 0.7)
e₃ = Geometry.Contravariant3Vector(1.0)
e³ = Geometry.Covariant3Vector(1)
ᶜᶜmat3 = fill(TridiagonalMatrixRow(2.0, 3.2, 2.1), center_space)
ᶜᶠmat2 = fill(BidiagonalMatrixRow(4.3, 1.7), center_space)
ᶜᶜmat3_uₕ_scalar = ᶜᶜmat3 .* (e¹²,)
ρχ_unit = (;ρq_liq = 1.0, ρq_ice = 1.0)
ᶜᶠmat2_ρχ_u₃ = map(Base.Fix1(map, Base.Fix2(*, ρχ_unit * e₃')), ᶜᶠmat2)

A = MatrixFields.FieldMatrix(
    (@name(c.ρχ), @name(f.u₃)) => ᶜᶠmat2_ρχ_u₃,
    (@name(c.uₕ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_uₕ_scalar,
)

A_scalar = MatrixFields.scalar_field_matrix(A)
keys(A_scalar)
# Output:
# (@name(c.ρχ.ρq_liq), @name(f.u₃.:(1)))
# (@name(c.ρχ.ρq_ice), @name(f.u₃.:(1)))
# (@name(c.uₕ.:(1)), @name(c.sgsʲs.:(1).ρa))
# (@name(c.uₕ.:(2)), @name(c.sgsʲs.:(1).ρa))
```
"""
function scalar_field_matrix(field_matrix::FieldMatrix)
    scalar_keys = get_scalar_keys(field_matrix)
    entries = unrolled_map(scalar_keys.values) do key
        field_matrix[key]
    end
    return FieldNameDict(scalar_keys, entries)
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
        !is_diagonal_matrix_entry(pair[2])
    end
    non_diagonal_entry_keys =
        FieldMatrixKeys(unrolled_map(first, non_diagonal_entry_pairs))
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
    unrolled_any(Base.Fix2(isa, Base.AbstractBroadcasted), values(dict))

"""
    lazy_main_diagonal(matrix)

Constructs a lazy `FieldMatrix` that contains the main diagonal of `matrix`.
"""
function lazy_main_diagonal(matrix)
    diagonal_keys = matrix_diagonal_keys(keys(matrix))
    entries = map(diagonal_keys) do key
        entry = matrix[key]
        is_diagonal_matrix_entry(entry) ? entry :
        Base.Broadcast.broadcasted(row -> DiagonalMatrixRow(row[0]), entry)
    end
    return FieldNameDict(diagonal_keys, entries)
end

"""
    identity_field_matrix(x)

Constructs a `FieldMatrix` that represents the identity operator for the
`FieldVector` `x`. The keys of this `FieldMatrix` correspond to single values,
such as numbers and vectors.

This offers an alternative to `one(matrix)`, which is not guaranteed to have all
the entries required to solve `matrix * x = b` for `x` if `matrix` is sparse.
"""
function identity_field_matrix(x::Fields.FieldVector)
    single_field_names = filtered_names(x) do field
        field isa Fields.Field && eltype(field) <: SingleValue
    end
    single_field_keys = FieldVectorKeys(single_field_names, FieldNameTree(x))
    entries = map(single_field_keys) do name
        # This must be consistent with the definition of one(::FieldMatrix).
        T = eltype(get_field(x, name))
        if T <: Number
            UniformScaling(one(T))
        elseif T <: Geometry.AxisVector
            # TODO: Add methods for +(::UniformScaling, ::Axis2Tensor) and
            # -(::UniformScaling, ::Axis2Tensor) to simplify this.
            tensor_axes = (axes(T)[1], Geometry.dual(axes(T)[1]))
            tensor_data = UniformScaling(one(eltype(T)))
            DiagonalMatrixRow(Geometry.AxisTensor(tensor_axes, tensor_data))
        else
            I # default value for elements that are neither scalars nor vectors
        end
    end
    return FieldNameDict(corresponding_matrix_keys(single_field_keys), entries)
end

"""
    field_vector_view(x, [name_tree])

Constructs a `FieldVectorView` that contains all of the `Field`s in the
`FieldVector` `x`. The default `name_tree` is `FieldNameTree(x)`, but this can
be modified if needed.
"""
function field_vector_view(x, name_tree = FieldNameTree(x))
    field_names = filtered_names(field -> field isa Fields.Field, x)
    field_keys = FieldVectorKeys(field_names, name_tree)
    entries = map(name -> get_field(x, name), field_keys)
    return FieldNameDict(field_keys, entries)
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

const SingleValueStyle =
    Union{Base.Broadcast.DefaultArrayStyle{0}, Base.Broadcast.Style{Tuple}}

const SingleValueStyleType = Union{
    Number,
    Ref{SingleValue},
    Tuple{SingleValue},
    Base.Broadcast.Broadcasted{<:SingleValueStyle},
}

Base.Broadcast.broadcastable(vector_or_matrix::FieldNameDict) = vector_or_matrix

Base.Broadcast.BroadcastStyle(::Type{<:FieldNameDict}) = FieldNameDictStyle()
Base.Broadcast.BroadcastStyle(::FieldNameDictStyle, ::Fields.FieldVectorStyle) =
    FieldNameDictStyle()
Base.Broadcast.BroadcastStyle(::FieldNameDictStyle, ::SingleValueStyle) =
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
    unrolled_reduce((arg1, arg2, arg3, args...)) do arg1′, arg2′
        Base.Broadcast.broadcasted(f, arg1′, arg2′)
    end

# Add support for broadcast expressions of the form dict1 .= dict2.
Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    ::typeof(identity),
    arg::FieldNameDict,
) = arg

# Add support for multiplication and division by single values.
function Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    f::Union{typeof(*), typeof(/), typeof(\)},
    single_value_or_bc::SingleValueStyleType,
    vector_or_matrix::FieldNameDict,
)
    single_value = Base.Broadcast.materialize(single_value_or_bc)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        entry isa ScalingFieldMatrixEntry ? f(single_value, entry) :
        Base.Broadcast.broadcasted(f, single_value, entry)
    end
    return FieldNameDict(keys(vector_or_matrix), entries)
end
function Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    f::Union{typeof(*), typeof(/), typeof(\)},
    vector_or_matrix::FieldNameDict,
    single_value_or_bc::SingleValueStyleType,
)
    single_value = Base.Broadcast.materialize(single_value_or_bc)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        entry isa ScalingFieldMatrixEntry ? f(entry, single_value) :
        Base.Broadcast.broadcasted(f, entry, single_value)
    end
    return FieldNameDict(keys(vector_or_matrix), entries)
end

function Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    ::typeof(zero),
    vector_or_matrix::FieldNameDict,
)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        entry isa ScalingFieldMatrixEntry ? zero(entry) :
        Base.Broadcast.broadcasted(zero, entry)
    end
    return FieldNameDict(keys(vector_or_matrix), entries)
end

function Base.Broadcast.broadcasted(
    ::FieldNameDictStyle,
    ::typeof(-),
    vector_or_matrix::FieldNameDict,
)
    entries = unrolled_map(values(vector_or_matrix)) do entry
        entry isa ScalingFieldMatrixEntry ? -entry :
        Base.Broadcast.broadcasted(-, entry)
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
            if (
                entry1 isa ScalingFieldMatrixEntry &&
                entry2 isa ScalingFieldMatrixEntry
            )
                f(entry1, entry2)
            elseif entry1 isa ScalingFieldMatrixEntry
                Base.Broadcast.broadcasted(f, (entry1,), entry2)
            elseif entry2 isa ScalingFieldMatrixEntry
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
                entry isa ScalingFieldMatrixEntry ? -entry :
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
            if (
                entry1 isa ScalingFieldMatrixEntry &&
                entry2 isa ScalingFieldMatrixEntry
            )
                product_value = scaling_value(entry1) * scaling_value(entry2)
                product_value isa Number ?
                (UniformScaling(product_value),) :
                (DiagonalMatrixRow(product_value),)
            elseif entry1 isa ScalingFieldMatrixEntry
                Base.Broadcast.broadcasted(*, (scaling_value(entry1),), entry2)
            elseif entry2 isa ScalingFieldMatrixEntry
                Base.Broadcast.broadcasted(*, entry1, (scaling_value(entry2),))
            else
                Base.Broadcast.broadcasted(*, entry1, entry2)
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
        entry isa ScalingFieldMatrixEntry ? inv(entry) :
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

NVTX.@annotate function copyto_foreach!(
    dest::FieldNameDict,
    vector_or_matrix::FieldNameDict,
)
    foreach(keys(vector_or_matrix)) do key
        entry = vector_or_matrix[key]
        if dest[key] isa ScalingFieldMatrixEntry
            dest[key] == entry || error("matrix entry at $key is immutable")
        elseif entry isa ScalingFieldMatrixEntry
            dest[key] .= (entry,)
        else
            dest[key] .= entry
        end
    end
end

NVTX.@annotate function Base.Broadcast.materialize!(
    dest::FieldNameDict,
    vector_or_matrix::FieldNameDict,
)
    !is_lazy(dest) || error("Cannot materialize into a lazy FieldNameDict")
    is_subset_that_covers_set(keys(vector_or_matrix), keys(dest)) || error(
        "Broadcast result and destination keys are incompatible: \
         $(set_string(keys(vector_or_matrix))) vs. $(set_string(keys(dest)))",
    ) # It is not always the case that keys(vector_or_matrix) == keys(dest).
    copyto_foreach!(dest, vector_or_matrix)
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
