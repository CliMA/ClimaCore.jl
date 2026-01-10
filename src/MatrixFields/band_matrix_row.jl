"""
    BandMatrixRow{ld}(entries...)

Stores the nonzero entries in a row of a band matrix, starting with the lowest
diagonal, which has index `ld`. Supported operations include accessing the entry
on the diagonal with index `d` by calling `row[d]`, taking linear combinations
with other band matrix rows (and with `LinearAlgebra.I`), and checking for
equality with other band matrix rows (and with `LinearAlgebra.I`). There are
several aliases for commonly used subtypes of `BandMatrixRow`:
- `DiagonalMatrixRow(entry_1)`
- `BidiagonalMatrixRow(entry_1, entry_2)`
- `TridiagonalMatrixRow(entry_1, entry_2, entry_3)`
- `QuaddiagonalMatrixRow(entry_1, entry_2, entry_3, entry_4)`
- `PentadiagonalMatrixRow(entry_1, entry_2, entry_3, entry_4, entry_5)`
"""
struct BandMatrixRow{ld, bw, T} # bw is the bandwidth (the number of diagonals)
    entries::NTuple{bw, T}
    BandMatrixRow{ld, bw, T}(entries::NTuple{bw, Any}) where {ld, bw, T} =
        new{ld, bw, T}(entries)
end
BandMatrixRow{ld}(entries::Vararg{Any, bw}) where {ld, bw} =
    BandMatrixRow{ld, bw}(entries...)
BandMatrixRow{ld, bw}(entries::Vararg{Any, bw}) where {ld, bw} =
    BandMatrixRow{ld, bw, promote_type(map(typeof, entries)...)}(entries)

const DiagonalMatrixRow{T} = BandMatrixRow{0, 1, T}
const BidiagonalMatrixRow{T} = BandMatrixRow{-1 + half, 2, T}
const TridiagonalMatrixRow{T} = BandMatrixRow{-1, 3, T}
const QuaddiagonalMatrixRow{T} = BandMatrixRow{-2 + half, 4, T}
const PentadiagonalMatrixRow{T} = BandMatrixRow{-2, 5, T}

"""
    outer_diagonals(::Type{<:BandMatrixRow})

Gets the indices of the lower and upper diagonals, `ld` and `ud`, of the given
subtype of `BandMatrixRow`.
"""
outer_diagonals(::Type{<:BandMatrixRow{ld, bw}}) where {ld, bw} =
    (ld, ld + bw - 1)

@inline lower_diagonal(::Tuple{<:BandMatrixRow{ld}}) where {ld} = ld
@inline lower_diagonal(t::Tuple) = lower_diagonal(t...)
@inline lower_diagonal(::BandMatrixRow{ld}, ::BandMatrixRow{ld}...) where {ld} =
    ld

"""
    band_matrix_row_type(ld, ud, T)

A shorthand for getting the subtype of `BandMatrixRow` that has entries of type
`T` on the diagonals with indices in the range `ld:ud`.
"""
band_matrix_row_type(ld, ud, T) = BandMatrixRow{ld, ud - ld + 1, T}

Base.eltype(::Type{BandMatrixRow{ld, bw, T}}) where {ld, bw, T} = T

Base.map(f::F, rows::BandMatrixRow...) where {F} =
    BandMatrixRow{lower_diagonal(rows)}(
        map(f, map(row -> row.entries, rows)...)...,
    )

Base.@propagate_inbounds Base.getindex(row::BandMatrixRow{ld}, d) where {ld} =
    row.entries[d - ld + 1]

function Base.promote_rule(
    ::Type{BMR1},
    ::Type{BMR2},
) where {BMR1 <: BandMatrixRow, BMR2 <: BandMatrixRow}
    ld1, ud1 = outer_diagonals(BMR1)
    ld2, ud2 = outer_diagonals(BMR2)
    typeof(ld1) == typeof(ld2) || error(
        "Cannot promote the $(ld1 isa PlusHalf ? "non-" : "")square matrix \
         row type $BMR1 and the $(ld2 isa PlusHalf ? "non-" : "")square matrix \
         row type $BMR2 to a common type",
    )
    T = promote_type(eltype(BMR1), eltype(BMR2))
    return band_matrix_row_type(min(ld1, ld2), max(ud1, ud2), T)
end

Base.promote_rule(
    ::Type{BMR},
    ::Type{US},
) where {BMR <: BandMatrixRow, US <: UniformScaling} =
    promote_rule(BMR, DiagonalMatrixRow{eltype(US)})

function Base.convert(
    ::Type{BMR},
    row::BandMatrixRow,
) where {BMR <: BandMatrixRow}
    old_ld, old_ud = outer_diagonals(typeof(row))
    new_ld, new_ud = outer_diagonals(BMR)
    typeof(old_ld) == typeof(new_ld) || error(
        "Cannot convert a $(old_ld isa PlusHalf ? "non-" : "")square matrix \
         row of type $(typeof(row)) to the \
         $(new_ld isa PlusHalf ? "non-" : "")square matrix row type $BMR",
    )
    new_ld <= old_ld && new_ud >= old_ud || error(
        "Cannot convert a $(typeof(row)) to a $BMR, since that would require \
         dropping potentially non-zero row entries",
    )
    first_zeros = ntuple(_ -> zero(eltype(BMR)), Val(old_ld - new_ld))
    last_zeros = ntuple(_ -> zero(eltype(BMR)), Val(new_ud - old_ud))
    return BMR((first_zeros..., row.entries..., last_zeros...))
end

Base.convert(::Type{BMR}, row::UniformScaling) where {BMR <: BandMatrixRow} =
    convert(BMR, DiagonalMatrixRow(row.λ))

Base.:(==)(row1::BMR, row2::BMR) where {BMR <: BandMatrixRow} =
    row1.entries == row2.entries
Base.:(==)(row1::BandMatrixRow, row2::BandMatrixRow) =
    ==(promote(row1, row2)...)
Base.:(==)(row1::BandMatrixRow, row2::UniformScaling) =
    ==(promote(row1, row2)...)
Base.:(==)(row1::UniformScaling, row2::BandMatrixRow) =
    ==(promote(row1, row2)...)

Base.:+(row::BandMatrixRow) = map(+, row)
Base.:+(row1::BandMatrixRow, row2::BandMatrixRow) =
    map(+, promote(row1, row2)...)
Base.:+(row1::BandMatrixRow, row2::UniformScaling) =
    map(+, promote(row1, row2)...)
Base.:+(row1::UniformScaling, row2::BandMatrixRow) =
    map(+, promote(row1, row2)...)

Base.:-(row::BandMatrixRow) = map(-, row)
Base.:-(row1::BandMatrixRow, row2::BandMatrixRow) =
    map(-, promote(row1, row2)...)
Base.:-(row1::BandMatrixRow, row2::UniformScaling) =
    map(-, promote(row1, row2)...)
Base.:-(row1::UniformScaling, row2::BandMatrixRow) =
    map(-, promote(row1, row2)...)

Base.:*(row::BandMatrixRow, value::Union{SingleValue, AutoBroadcaster}) =
    map(Base.Fix2(*, value), row)
Base.:*(value::Union{SingleValue, AutoBroadcaster}, row::BandMatrixRow) =
    map(Base.Fix1(*, value), row)

Base.:/(row::BandMatrixRow, value::Union{SingleValue, AutoBroadcaster}) =
    map(Base.Fix2(/, value), row)

Base.zero(row::BandMatrixRow) = zero(typeof(row))
Base.zero(::Type{BandMatrixRow{ld, bw, T}}) where {ld, bw, T} =
    BandMatrixRow{ld}(ntuple(Returns(zero(T)), Val(bw))...)

Base.one(row::BandMatrixRow) = one(typeof(row))
Base.one(::Type{DiagonalMatrixRow{T}}) where {T} = DiagonalMatrixRow(one(T))
Base.one(::Type{BandMatrixRow{ld, bw, T}}) where {ld, bw, T} =
    ld isa PlusHalf ?
    error("A non-square matrix does not have a corresponding identity matrix") :
    one(DiagonalMatrixRow{T})

inv(row::DiagonalMatrixRow) = DiagonalMatrixRow(inv(row[0]))
inv(::BandMatrixRow{ld, bw}) where {ld, bw} = error(
    "The inverse of a matrix with $bw diagonals is generally a dense matrix, \
     so it cannot be represented using BandMatrixRows",
)

# Allow row entries to be wrapped in AutoBroadcasters, but not the row itself.
is_auto_broadcastable(::Type{BMR}) where {BMR <: BandMatrixRow} =
    is_auto_broadcastable(eltype(BMR))
enable_auto_broadcasting(row::BandMatrixRow) =
    map(enable_auto_broadcasting, row)
disable_auto_broadcasting(row::BandMatrixRow) =
    map(disable_auto_broadcasting, row)
