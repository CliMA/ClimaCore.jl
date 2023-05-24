import LinearAlgebra
import ClimaCore.Utilities: PlusHalf, half

"""
    BandMatrixRow{ld}(entries...)

Stores the nonzero entries in a row of a band matrix, starting with the lowest
diagonal, which has index `ld`. Supported operations include accessing the entry
on the diagonal with index `d` by calling `row[d]`, as well as taking linear
combinations with other band matrix rows.
"""
struct BandMatrixRow{ld, bw, T}
    entries::NTuple{bw, T}
end
# The parameter bw is the "bandwidth", i.e., the number of nonzero entries.

const DiagonalMatrixRow{T} = BandMatrixRow{0, 1, T}
const BidiagonalMatrixRow{T} = BandMatrixRow{-1 + half, 2, T}
const TridiagonalMatrixRow{T} = BandMatrixRow{-1, 3, T}
const QuaddiagonalMatrixRow{T} = BandMatrixRow{-2 + half, 4, T}
const PentadiagonalMatrixRow{T} = BandMatrixRow{-2, 5, T}

BandMatrixRow{ld}(entries::Vararg{Any, bw}) where {ld, bw} =
    BandMatrixRow{ld, bw}(entries...)
BandMatrixRow{ld, bw}(entries::Vararg{Any, bw}) where {ld, bw} =
    BandMatrixRow{ld, bw}(promote(entries...)...)
BandMatrixRow{ld, bw}(entries::Vararg{T, bw}) where {ld, bw, T} =
    BandMatrixRow{ld, bw, T}(entries)

"""
    outer_diagonals(::Type{<:BandMatrixRow})

Gets the indices of the lower and upper diagonals, `ld` and `ud`, of the given
`BandMatrixRow` type.
"""
outer_diagonals(::Type{<:BandMatrixRow{ld, bw}}) where {ld, bw} =
    (ld, ld + bw - 1)

"""
    band_matrix_row_type(ld, ud, T)

Returns the element type of a band matrix that has entries of type `T` on the
diagonals with indices in the range `ld:ud`.
"""
band_matrix_row_type(ld, ud, T) = BandMatrixRow{ld, ud - ld + 1, T}

Base.eltype(::Type{<:BandMatrixRow{ld, bw, T}}) where {ld, bw, T} = T

function Base.promote_rule(
    ::Type{BMR1},
    ::Type{BMR2},
) where {BMR1 <: BandMatrixRow, BMR2 <: BandMatrixRow}
    ld1, ud1 = outer_diagonals(BMR1)
    ld2, ud2 = outer_diagonals(BMR2)
    typeof(ld1) == typeof(ld2) || error(
        "Cannot promote the $(ld1 isa PlusHalf ? "non-" : "")square matrix \
         row type $BMR1 and the $(ld2 isa PlusHalf ? "non-" : "")square \
         matrix row type $BMR2 to a common type",
    )
    T = promote_type(eltype(BMR1), eltype(BMR2))
    return band_matrix_row_type(min(ld1, ld2), max(ud1, ud2), T)
end

function Base.convert(
    ::Type{BMR},
    row::BandMatrixRow,
) where {BMR <: BandMatrixRow}
    old_ld, old_ud = outer_diagonals(typeof(row))
    new_ld, new_ud = outer_diagonals(BMR)
    typeof(old_ld) == typeof(new_ld) ||
        error("Cannot convert a $(old_ld isa PlusHalf ? "non-" : "")square \
               matrix row of type $(typeof(row)) to the \
               $(new_ld isa PlusHalf ? "non-" : "")square matrix row type $BMR")
    new_ld <= old_ld && new_ud >= old_ud ||
        error("Cannot convert a $(typeof(row)) to a $BMR, since that would \
               require dropping potentially non-zero row entries")
    first_zeros = ntuple(_ -> zero(eltype(BMR)), Val(old_ld - new_ld))
    entries = map(entry -> convert(eltype(BMR), entry), row.entries)
    last_zeros = ntuple(_ -> zero(eltype(BMR)), Val(new_ud - old_ud))
    return BandMatrixRow{new_ld}(first_zeros..., entries..., last_zeros...)
end

Base.promote_rule(
    ::Type{BMR},
    ::Type{<:LinearAlgebra.UniformScaling{T}},
) where {BMR <: BandMatrixRow, T} =
    band_matrix_row_type(outer_diagonals(BMR)..., promote_type(eltype(BMR), T))

Base.convert(
    ::Type{BMR},
    row::LinearAlgebra.UniformScaling,
) where {BMR <: BandMatrixRow} = convert(BMR, DiagonalMatrixRow(row.λ))

Base.@propagate_inbounds Base.getindex(row::BandMatrixRow{ld}, d) where {ld} =
    row.entries[d - ld + 1]

Base.:(==)(row1::BMR, row2::BMR) where {BMR <: BandMatrixRow} =
    row1.entries == row2.entries
Base.:(==)(row1::BandMatrixRow, row2::BandMatrixRow) =
    ==(promote(row1, row2)...)
Base.:(==)(row1::BandMatrixRow, row2::LinearAlgebra.UniformScaling) =
    ==(promote(row1, row2)...)
Base.:(==)(row1::LinearAlgebra.UniformScaling, row2::BandMatrixRow) =
    ==(promote(row1, row2)...)

# Define all necessary operations for computing linear combinations of matrices.
const ScalarQuantity = Union{Number, Geometry.AxisTensor}
Base.map(f, rows::BMR...) where {ld, BMR <: BandMatrixRow{ld}} =
    BandMatrixRow{ld}(map(f, map(row -> row.entries, rows)...)...)
for op in (:+, :-)
    @eval begin
        Base.$op(row::BandMatrixRow) = map($op, row)
        Base.$op(row1::BandMatrixRow, row2::BandMatrixRow) =
            map($op, promote(row1, row2)...)
        Base.$op(row1::BandMatrixRow, row2::LinearAlgebra.UniformScaling) =
            map($op, promote(row1, row2)...)
        Base.$op(row1::LinearAlgebra.UniformScaling, row2::BandMatrixRow) =
            map($op, promote(row1, row2)...)
    end
end
for op in (:*, :/)
    @eval begin
        Base.$op(row::BandMatrixRow, scalar::ScalarQuantity) =
            map(Base.Fix2($op, scalar), row)
    end
end
for op in (:*, :\)
    @eval begin
        Base.$op(scalar::ScalarQuantity, row::BandMatrixRow) =
            map(Base.Fix1($op, scalar), row)
    end
end

# Don't implement the Hadamard product in order to avoid accidental ⋅/* typos.
Base.:*(::BandMatrixRow, ::BandMatrixRow) =
    error("Band matrices must be multiplied using ⋅, not *")


################################################################################
################################################################################
################################################################################


"""
    Times_eᵀ(value)

Represents the quantity `value * eᵀ`, where `eᵀ` denotes the transpose of a
basis vector. For example, if `e₃` denotes the basis vector along the
`Contravariant3Axis`, then we can express `Contravariant3Vector(u³)` as
`u³ * e₃`. Since `dot(e³, e₃) = 1`, this means that
`e³ᵀ * Contravariant3Vector(u³) = u³ * e³ᵀ * e₃ = u³`. More precisely,
`Times_e³ᵀ(value)` represents a quantity that, when multiplied by some
`AxisVector` `x` defined on a `LocalGeometry` `lg`, returns the quantity
`value * contravariant3(x, lg)`. There is a subtype of `Times_eᵀ` for each basis
vector: `Times_e₁ᵀ`, `Times_e₂ᵀ`, `Times_e₃ᵀ`, `Times_e¹ᵀ`, `Times_e²ᵀ`, and
`Times_e³ᵀ`.
"""
struct Times_eᵀ{A, T}
    value::T
end
Times_eᵀ{A}() where {A} = Times_eᵀ{A}(true)

const Times_e₁ᵀ{T} = Times_eᵀ{Geometry.Covariant1Axis, T}
const Times_e₂ᵀ{T} = Times_eᵀ{Geometry.Covariant2Axis, T}
const Times_e₃ᵀ{T} = Times_eᵀ{Geometry.Covariant3Axis, T}
const Times_e¹ᵀ{T} = Times_eᵀ{Geometry.Contravariant1Axis, T}
const Times_e²ᵀ{T} = Times_eᵀ{Geometry.Contravariant2Axis, T}
const Times_e³ᵀ{T} = Times_eᵀ{Geometry.Contravariant3Axis, T}

Base.eltype(::Type{Times_eᵀ{A, T}}) where {A, T} = T

"""
    mul(x, y, lg)

Computes `x * y`, using the `LocalGeometry` `lg` for any `AxisVector`
projections required by this computation. For example, if `x` is a `Times_e³ᵀ`
and `y` is an `AxisVector`, then `y` will be projected onto the
`Contravariant3Axis` for the multiplication. In order to simplify the
implementation, projecting higher-dimensional `AxisTensor`s is not supported.
"""
mul(x, y, _) = x * y
mul(x, y::Times_eᵀ, lg) = Times_eᵀ(mul(x, y.value, lg))
mul(x::Times_eᵀ, y, lg) = Times_eᵀ(mul(x.value, y, lg))
mul(x::Times_eᵀ, y::Times_eᵀ, lg) = Times_eᵀ(mul(x.value, y, lg))
mul(x::Times_e₁ᵀ, y::Geometry.AxisVector, lg) =
    mul(x.value, Geometry.covariant1(y, lg), lg)
mul(x::Times_e₂ᵀ, y::Geometry.AxisVector, lg) =
    mul(x.value, Geometry.covariant2(y, lg), lg)
mul(x::Times_e₃ᵀ, y::Geometry.AxisVector, lg) =
    mul(x.value, Geometry.covariant3(y, lg), lg)
mul(x::Times_e¹ᵀ, y::Geometry.AxisVector, lg) =
    mul(x.value, Geometry.contravariant1(y, lg), lg)
mul(x::Times_e²ᵀ, y::Geometry.AxisVector, lg) =
    mul(x.value, Geometry.contravariant2(y, lg), lg)
mul(x::Times_e³ᵀ, y::Geometry.AxisVector, lg) =
    mul(x.value, Geometry.contravariant3(y, lg), lg)
mul(_::Times_eᵀ, _::Geometry.AxisTensor, _) =
    error("Multiplying a generic AxisTensor by a Times_eᵀ is not yet supported")

"""
    mul_type(X, Y)

Computes the result type of `mul(x::X, y::Y, lg)`.
"""
mul_type(::Type{X}, ::Type{Y}) where {X, Y} = promote_type(X, Y)
mul_type(::Type{X}, ::Type{Y}) where {X, Y <: Times_eᵀ} =
    Times_eᵀ{mul_type(X, eltype(Y))}
mul_type(::Type{X}, ::Type{Y}) where {X <: Times_eᵀ, Y} =
    Times_eᵀ{mul_type(eltype(X), Y)}
mul_type(::Type{X}, ::Type{Y}) where {X <: Times_eᵀ, Y <: Times_eᵀ} =
    Times_eᵀ{mul_type(eltype(X), Y)}
mul_type(::Type{X}, ::Type{Y}) where {X <: Times_eᵀ, Y <: Geometry.AxisVector} =
    mul_type(eltype(X), eltype(Y))
mul_type(::Type{X}, ::Type{Y}) where {X <: Times_eᵀ, Y <: Geometry.AxisTensor} =
    error("Multiplying a generic AxisTensor by a Times_eᵀ is not yet supported")


################################################################################
################################################################################
################################################################################


"""
    MultiplyColumnwiseBandMatrixField

An operator that multiplies a columnwise band matrix field (a field of
`BandMatrixRow`s) by a regular field or by another columnwise band matrix field,
i.e., matrix-vector or matrix-matrix multiplication. The `⋅` symbol is an alias
for `MultiplyColumnwiseBandMatrixField()`.
"""
struct MultiplyColumnwiseBandMatrixField <: FiniteDifferenceOperator end
const ⋅ = MultiplyColumnwiseBandMatrixField()

#=
TODO: Rewrite the following derivation in LaTeX and move it into the ClimaCore
documentation.

Notation:

For any single-column field F, let F[idx] denote the value of F at level idx.
For any single-column BandMatrixRow field M, let
    M[idx, idx′] = M[idx][idx′ - idx].
If there are multiple columns, the following equations apply per column.

Matrix-Vector Multiplication:

Consider a BandMatrixRow field M and a scalar (non-BandMatrixRow) field V.
From the definition of matrix-vector multiplication,
    (M ⋅ V)[idx] = ∑_{idx′} M[idx, idx′] * V[idx′].
If V[idx] is only defined when left_idx ≤ idx ≤ right_idx, this becomes
    (M ⋅ V)[idx] = ∑_{idx′ ∈ left_idx:right_idx} M[idx, idx′] * V[idx′].
If M[idx, idx′] is only defined when idx + ld ≤ idx′ ≤ idx + ud, this becomes
    (M ⋅ V)[idx] =
        ∑_{idx′ ∈ max(left_idx, idx + ld):min(right_idx, idx + ud)}
            M[idx, idx′] * V[idx′].
Replacing the variable idx′ with the variable d = idx′ - idx gives us
    (M ⋅ V)[idx] =
        ∑_{d ∈ max(left_idx - idx, ld):min(right_idx - idx, ud)}
            M[idx, idx + d] * V[idx + d].
This can be rewritten using the standard indexing notation as
    (M ⋅ V)[idx] =
        ∑_{d ∈ max(left_idx - idx, ld):min(right_idx - idx, ud)}
            M[idx][d] * V[idx + d].
Finally, we can express this in terms of left/right boundaries and an interior:
    (M ⋅ V)[idx] =
        ∑_{
            d ∈
                if idx < left_idx - ld
                    (left_idx - idx):ud
                elseif idx > right_idx - ud
                    ld:(right_idx - idx)
                else
                    ld:ud
                end
        } M[idx][d] * V[idx + d].

Matrix-Matrix Multiplication:

Consider a BandMatrixRow field M1 and another BandMatrixRow field M2.
From the definition of matrix-matrix multiplication,
    (M1 ⋅ M2)[idx, idx′] = ∑_{idx′′} M1[idx, idx′′] * M2[idx′′, idx′].
If M2[idx′′] is only defined when left_idx ≤ idx′′ ≤ right_idx, this becomes
    (M1 ⋅ M2)[idx, idx′] =
        ∑_{idx′′ ∈ left_idx:right_idx} M1[idx, idx′′] * M2[idx′′, idx′].
If M1[idx, idx′′] is only defined when idx + ld1 ≤ idx′′ ≤ idx + ud1, this becomes
    (M1 ⋅ M2)[idx, idx′] =
        ∑_{idx′′ ∈ max(left_idx, idx + ld1):min(right_idx, idx + ud1)}
            M1[idx, idx′′] * M2[idx′′, idx′].
If M2[idx′′, idx′] is only defined when idx′′ + ld2 ≤ idx′ ≤ idx′′ + ud2, or,
equivalently, when idx′ - ud2 ≤ idx′′ ≤ idx′ - ld2, this becomes
    (M1 ⋅ M2)[idx, idx′] =
        ∑_{
            idx′′ ∈
                max(left_idx, idx + ld1, idx′ - ud2):
                min(right_idx, idx + ud1, idx′ - ld2)
        } M1[idx, idx′′] * M2[idx′′, idx′].
Replacing the variable idx′ with the variable prod_d = idx′ - idx gives us
    (M1 ⋅ M2)[idx, idx + prod_d] =
        ∑_{
            idx′′ ∈
                max(left_idx, idx + ld1, idx + prod_d - ud2):
                min(right_idx, idx + ud1, idx + prod_d - ld2)
        } M1[idx, idx′′] * M2[idx′′, idx + prod_d].
Replacing the variable idx′′ with the variable d = idx′′ - idx gives us
    (M1 ⋅ M2)[idx, idx + prod_d] =
        ∑_{
            d ∈
                max(left_idx - idx, ld1, prod_d - ud2):
                min(right_idx - idx, ud1, prod_d - ld2)
        } M1[idx, idx + d] * M2[idx + d, idx + prod_d].
This can be rewritten using the standard indexing notation as
    (M1 ⋅ M2)[idx][prod_d] =
        ∑_{
            d ∈
                max(left_idx - idx, ld1, prod_d - ud2):
                min(right_idx - idx, ud1, prod_d - ld2)
        } M1[idx][d] * M2[idx + d][prod_d - d].
Finally, we can express this in terms of left/right boundaries and an interior:
    (M1 ⋅ M2)[idx][prod_d] =
        ∑_{
            d ∈
                if idx < left_idx - ld1
                    max(left_idx - idx, prod_d - ud2):min(ud1, prod_d - ld2)
                elseif idx > right_idx - ud1
                    max(ld1, prod_d - ud2):min(right_idx - idx, prod_d - ld2)
                else
                    max(ld1, prod_d - ud2):min(ud1, prod_d - ld2)
                end
        } M1[idx][d] * M2[idx + d][prod_d - d].

We only need to define (M1 ⋅ M2)[idx][prod_d] when it has a nonzero value in the
interior, which will be the case when
    max(ld1, prod_d - ud2) ≤ min(ud1, prod_d - ld2).
This can be rewritten as a system of four inequalities:
    ld1 ≤ ud1,
    ld1 ≤ prod_d - ld2,
    prod_d - ud2 ≤ ud1, and
    prod_d - ud2 ≤ prod_d - ld2.
By definition, ld1 ≤ ud1 and ld2 ≤ ud2, so the first and last inequality are
always true. Rearranging the remaining two inequalities gives us
    ld1 + ld2 ≤ prod_d ≤ ud1 + ud2.
=#

struct TopLeftMatrixCorner <: BoundaryCondition end
struct BottomRightMatrixCorner <: BoundaryCondition end

has_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::LeftBoundaryWindow{name},
) where {name} = true
has_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::RightBoundaryWindow{name},
) where {name} = true

get_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::LeftBoundaryWindow{name},
) where {name} = TopLeftMatrixCorner()
get_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::RightBoundaryWindow{name},
) where {name} = BottomRightMatrixCorner()

stencil_interior_width(
    ::MultiplyColumnwiseBandMatrixField,
    matrix,
    mat_or_vec,
) = ((0, 0), outer_diagonals(eltype(matrix)))

function boundary_width(
    ::MultiplyColumnwiseBandMatrixField,
    ::TopLeftMatrixCorner,
    matrix,
    mat_or_vec,
)
    ld = outer_diagonals(eltype(matrix))[1]
    return max((left_idx(axes(mat_or_vec)) - ld) - left_idx(axes(matrix)), 0)
end
function boundary_width(
    ::MultiplyColumnwiseBandMatrixField,
    ::BottomRightMatrixCorner,
    matrix,
    mat_or_vec,
)
    ud = outer_diagonals(eltype(matrix))[2]
    return max(right_idx(axes(matrix)) - (right_idx(axes(mat_or_vec)) - ud), 0)
end

function product_matrix_outer_diagonals(matrix1, matrix2)
    ld1, ud1 = outer_diagonals(eltype(matrix1))
    ld2, ud2 = outer_diagonals(eltype(matrix2))
    return (ld1 + ld2, ud1 + ud2)
end

function return_eltype(::MultiplyColumnwiseBandMatrixField, matrix, mat_or_vec)
    eltype(matrix) <: BandMatrixRow ||
        error("The first argument of ⋅ must be a band matrix field, but the \
               given argument is a field with element type $(eltype(matrix))")
    if eltype(mat_or_vec) <: BandMatrixRow # matrix-matrix multiplication
        matrix2 = mat_or_vec
        prod_ld, prod_ud = product_matrix_outer_diagonals(matrix, matrix2)
        prod_eltype = mul_type(eltype(eltype(matrix)), eltype(eltype(matrix2)))
        return band_matrix_row_type(prod_ld, prod_ud, prod_eltype)
    else # matrix-vector multiplication
        vector = mat_or_vec
        return mul_type(eltype(eltype(matrix)), eltype(vector))
    end
end

return_space(::MultiplyColumnwiseBandMatrixField, matrix_space, _) =
    matrix_space

Base.@propagate_inbounds function multiply_columnwise_band_matrix_at_index(
    loc,
    idx,
    hidx,
    matrix,
    mat_or_vec;
    ld = nothing,
    ud = nothing,
)
    if isnothing(ld)
        ld = outer_diagonals(eltype(matrix))[1]
    end
    if isnothing(ud)
        ud = outer_diagonals(eltype(matrix))[2]
    end
    if eltype(mat_or_vec) <: BandMatrixRow # matrix-matrix multiplication
        matrix2 = mat_or_vec
        ld2, ud2 = outer_diagonals(eltype(matrix2))
        prod_ld, prod_ud = product_matrix_outer_diagonals(matrix, matrix2)
        prod_entries = map(prod_ld:prod_ud) do prod_d
            Base.@_propagate_inbounds_meta
            mapreduce(⊞, max(ld, prod_d - ud2):min(ud, prod_d - ld2)) do d
                Base.@_propagate_inbounds_meta
                lg = Geometry.LocalGeometry(axes(matrix2), idx, hidx)
                rmap(
                    (x, y) -> mul(x, y, lg),
                    getidx(matrix, loc, idx, hidx)[d],
                    getidx(matrix2, loc, idx + d, hidx)[prod_d - d],
                )
            end
        end
        return BandMatrixRow{prod_ld}(prod_entries...)
    else # matrix-vector multiplication
        vector = mat_or_vec
        return mapreduce(⊞, ld:ud) do d
            Base.@_propagate_inbounds_meta
            lg = Geometry.LocalGeometry(axes(vector), idx, hidx)
            rmap(
                (x, y) -> mul(x, y, lg),
                getidx(matrix, loc, idx, hidx)[d],
                getidx(vector, loc, idx + d, hidx),
            )
        end
    end
end

Base.@propagate_inbounds stencil_interior(
    ::MultiplyColumnwiseBandMatrixField,
    loc,
    idx,
    hidx,
    matrix,
    mat_or_vec,
) = multiply_columnwise_band_matrix_at_index(loc, idx, hidx, matrix, mat_or_vec)

Base.@propagate_inbounds stencil_left_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::TopLeftMatrixCorner,
    loc,
    idx,
    hidx,
    matrix,
    mat_or_vec,
) = multiply_columnwise_band_matrix_at_index(
    loc,
    idx,
    hidx,
    matrix,
    mat_or_vec;
    ld = left_idx(axes(mat_or_vec)) - idx,
)

Base.@propagate_inbounds stencil_right_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::BottomRightMatrixCorner,
    loc,
    idx,
    hidx,
    matrix,
    mat_or_vec,
) = multiply_columnwise_band_matrix_at_index(
    loc,
    idx,
    hidx,
    matrix,
    mat_or_vec;
    ud = right_idx(axes(mat_or_vec)) - idx,
)


################################################################################
################################################################################
################################################################################


"""
    FiniteDifferenceOperatorTermsMatrix

A wrapper for a `FiniteDifferenceOperator` that converts its output from a
linear combination of values to a band matrix whose nonzero elements are the
terms of the linear combination.

That is, given an operator `op`, a field `F`, and optionally some additional
fields `args`, there are some values of `ld`, `ud`, and `C[idx, idx′]` (the last
of which may be a function of `args`) such that
    `op.(args..., F)[idx] =
        ∑_{idx′ ∈ (idx + ld):(idx + ud)} C[idx, idx′] * F[idx′]`.
If `op_matrix = FiniteDifferenceOperatorTermsMatrix(op)`, then
    `op_matrix.(args..., F)[idx] = BandMatrixRow{ld}(
        map(idx′ -> C[idx, idx′] * F[idx′], (idx + ld):(idx + ud))...
    )`.
Using the notation `matrix_field[idx, idx′] = matrix_field[idx][idx′ - idx]`,
this means that
    `op_matrix.(args..., F)[idx, idx′] = C[idx, idx′] * F[idx′]`.

This property of `op_matrix` means that it can be used to specify derivatives of
expressions that involve `op`. For example,
    `(∂(op.(args..., F))/∂F)[idx, idx′] =
        ∂(op.(args..., F)[idx])/∂(F[idx′]) =
        C[idx, idx′] =
        op_matrix.(args..., one.(F))[idx, idx′]`.
More generally, if `f` is some function that can act on each element of `F` and
`∂f∂F` is a function that returns the partial derivative of `f` with respect to
each element of `F`, then
    `(∂(op.(args..., f.(F)))/∂F)[idx, idx′] =
        ∂(op.(args..., f.(F))[idx])/∂(F[idx′]) =
        C[idx, idx′] * ∂f∂F.(F)[idx′] =
        op_matrix.(args..., ∂f∂F.(F))[idx, idx′]`.

This can be further generalized to compositions of multiple operators. Given two
operators `op₁` and `op₂`, each of which has corresponding values of `ld`, `ud`,
and `C`, and given two functions `f₁` and `f₂`, we have that
"""
struct FiniteDifferenceOperatorTermsMatrix{O <: FiniteDifferenceOperator} <:
       FiniteDifferenceOperator
    op::O
end

has_boundary(
    matrix_op::FiniteDifferenceOperatorTermsMatrix,
    bw::LeftBoundaryWindow{name},
) where {name} = has_boundary(matrix_op.op, bw)
has_boundary(
    matrix_op::FiniteDifferenceOperatorTermsMatrix,
    bw::RightBoundaryWindow{name},
) where {name} = has_boundary(matrix_op.op, bw)

get_boundary(
    matrix_op::FiniteDifferenceOperatorTermsMatrix,
    bw::LeftBoundaryWindow{name},
) where {name} = get_boundary(matrix_op.op, bw)
get_boundary(
    matrix_op::FiniteDifferenceOperatorTermsMatrix,
    bw::RightBoundaryWindow{name},
) where {name} = get_boundary(matrix_op.op, bw)

stencil_interior_width(
    matrix_op::FiniteDifferenceOperatorTermsMatrix,
    args...,
) = stencil_interior_width(matrix_op.op, args...)

boundary_width(
    matrix_op::FiniteDifferenceOperatorTermsMatrix,
    bc::BoundaryCondition,
    args...,
) = boundary_width(matrix_op.op, bc, args...)

function return_eltype(matrix_op::FiniteDifferenceOperatorTermsMatrix, args...)
    ld, ud = stencil_interior_width(matrix_op.op, args...)[end]
    return band_matrix_row_type(ld, ud, return_eltype(matrix_op.op, args...))
end

return_space(matrix_op::FiniteDifferenceOperatorTermsMatrix, spaces...) =
    return_space(matrix_op.op, spaces...)

# TODO: Figure out how to rewrite finitedifference.jl to simplify the methods
# of stencil_interior, stencil_left_boundary, and stencil_right_boundary for
# FiniteDifferenceOperatorTermsMatrix.


################################################################################
################################################################################
################################################################################


#=
original ∂Yₜ/∂Y =

    ∂cₜ/∂c ∂cₜ/∂f
    ∂fₜ/∂c ∂fₜ/∂f =

    ∂cₜ[1]/∂c[1]      ∂cₜ[1]/∂c[2]      ∂cₜ[1]/∂c[3]         ∂cₜ[1]/∂f[half]      ∂cₜ[1]/∂f[1+half]      ∂cₜ[1]/∂f[2+half]      ∂cₜ[1]/∂f[3+half]
    ∂cₜ[2]/∂c[1]      ∂cₜ[2]/∂c[2]      ∂cₜ[2]/∂c[3]         ∂cₜ[2]/∂f[half]      ∂cₜ[2]/∂f[1+half]      ∂cₜ[2]/∂f[2+half]      ∂cₜ[2]/∂f[3+half]
    ∂cₜ[3]/∂c[1]      ∂cₜ[3]/∂c[2]      ∂cₜ[3]/∂c[3]         ∂cₜ[3]/∂f[half]      ∂cₜ[3]/∂f[1+half]      ∂cₜ[3]/∂f[2+half]      ∂cₜ[3]/∂f[3+half]

    ∂fₜ[half]/∂c[1]   ∂fₜ[half]/∂c[2]   ∂fₜ[half]/∂c[3]      ∂fₜ[half]/∂f[half]   ∂fₜ[half]/∂f[1+half]   ∂fₜ[half]/∂f[2+half]   ∂fₜ[half]/∂f[3+half]
    ∂fₜ[1+half]/∂c[1] ∂fₜ[1+half]/∂c[2] ∂fₜ[1+half]/∂c[3]    ∂fₜ[1+half]/∂f[half] ∂fₜ[1+half]/∂f[1+half] ∂fₜ[1+half]/∂f[2+half] ∂fₜ[1+half]/∂f[3+half]
    ∂fₜ[2+half]/∂c[1] ∂fₜ[2+half]/∂c[2] ∂fₜ[2+half]/∂c[3]    ∂fₜ[2+half]/∂f[half] ∂fₜ[2+half]/∂f[1+half] ∂fₜ[2+half]/∂f[2+half] ∂fₜ[2+half]/∂f[3+half]
    ∂fₜ[3+half]/∂c[1] ∂fₜ[3+half]/∂c[2] ∂fₜ[3+half]/∂c[3]    ∂fₜ[3+half]/∂f[half] ∂fₜ[3+half]/∂f[1+half] ∂fₜ[3+half]/∂f[2+half] ∂fₜ[3+half]/∂f[3+half]

If f[3+half] is constant, we can drop all terms related to f[3+half].

rearranged ∂Yₜ/∂Y =

    ∂cell_1ₜ/∂cell_1 ∂cell_1ₜ/∂cell_2 ∂cell_1ₜ/∂cell_3
    ∂cell_2ₜ/∂cell_1 ∂cell_2ₜ/∂cell_2 ∂cell_2ₜ/∂cell_3
    ∂cell_3ₜ/∂cell_1 ∂cell_3ₜ/∂cell_2 ∂cell_3ₜ/∂cell_3 =

    ∂cₜ[1]/∂c[1]      ∂cₜ[1]/∂f[half]         ∂cₜ[1]/∂c[2]      ∂cₜ[1]/∂f[1+half]         ∂cₜ[1]/∂c[3]      ∂cₜ[1]/∂f[2+half]
    ∂fₜ[half]/∂c[1]   ∂fₜ[half]/∂f[half]      ∂fₜ[half]/∂c[2]   ∂fₜ[half]/∂f[1+half]      ∂fₜ[half]/∂c[3]   ∂fₜ[half]/∂f[2+half]

    ∂cₜ[2]/∂c[1]      ∂cₜ[2]/∂f[half]         ∂cₜ[2]/∂c[2]      ∂cₜ[2]/∂f[1+half]         ∂cₜ[2]/∂c[3]      ∂cₜ[2]/∂f[2+half]
    ∂fₜ[1+half]/∂c[1] ∂fₜ[1+half]/∂f[half]    ∂fₜ[1+half]/∂c[2] ∂fₜ[1+half]/∂f[1+half]    ∂fₜ[1+half]/∂c[3] ∂fₜ[1+half]/∂f[2+half]

    ∂cₜ[3]/∂c[1]      ∂cₜ[3]/∂f[half]         ∂cₜ[3]/∂c[2]      ∂cₜ[3]/∂f[1+half]         ∂cₜ[3]/∂c[3]      ∂cₜ[3]/∂f[2+half]
    ∂fₜ[2+half]/∂c[1] ∂fₜ[2+half]/∂f[half]    ∂fₜ[2+half]/∂c[2] ∂fₜ[2+half]/∂f[1+half]    ∂fₜ[2+half]/∂c[3] ∂fₜ[2+half]/∂f[2+half]
=#

"""
    NestedPropertyName{names}()

A singleton type that represents a chain of `getproperty` calls, which can be
used to access a property/sub-property of an object `x` by calling
`get_nested_property(x, ::NestedPropertyName)`. The entire object `x` can also
be accessed using the trivial `NestedPropertyName`, which has `names = ()`.
"""
struct NestedPropertyName{names} end

get_nested_property(x, ::NestedPropertyName{()}) = x
get_nested_property(x, ::NestedPropertyName{names}) where {names} =
    get_nested_property(
        getproperty(x, names[1]),
        NestedPropertyName{names[2:end]}(),
    )

function Base.show(io::IO, ::NestedPropertyName{names}) where {names}
    name_strings = map(name -> name isa Symbol ? "$name" : ":$name", names)
    nested_name_string = isempty(name_strings) ? "*" : join(name_strings, '.')
    print(io, "NestedPropertyName($nested_name_string)")
end

"""
    find_all_nested_properties(f, x)

Recursively finds all properties/sub-properties of `x` for which `f(property)`
is `true`. Returns a tuple that contains the `NestedPropertyName` of every such
`property`. If `f(property)` is `true`, this function will not continue to
search through the sub-properties of `property`. In order to ensure type
stability, the result of `f` should be inferrable from the type of its argument.

# Examples
```julia-repl
julia> find_all_nested_properties(x -> x isa Int, (; a = 1, b = (2.0, (; c = 3))))
(NestedPropertyName(a), NestedPropertyName(b.:2.c))

julia> find_all_nested_properties(_ -> true, (; a = 1, b = (2.0, (; c = 3))))
(NestedPropertyName(*),)
```
"""
find_all_nested_properties(f, x) =
    _find_all_nested(f, x, NestedPropertyName{()}())
_find_all_nested(f, x, nested_name) =
    if f(x)
        (nested_name,)
    elseif isempty(propertynames(x))
        ()
    else
        _find_all_nested(f, x, nested_name, map(Val, propertynames(x))...)
    end
_find_all_nested(
    f,
    x,
    ::NestedPropertyName{names},
    ::Val{name},
) where {names, name} = _find_all_nested(
    f,
    getproperty(x, name),
    NestedPropertyName{(names..., name)}(),
)
_find_all_nested(f, x, nested_name, first_val_name, other_val_names...) = (
    _find_all_nested(f, x, nested_name, first_val_name)...,
    _find_all_nested(f, x, nested_name, other_val_names...)...,
)

# This appears to be required for type stability as of Julia 1.8.
# TODO: Remove this when it is no longer needed.
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(_find_all_nested)
        m.recursion_relation = dont_limit
    end
end

"""
    @block_name(row_name_expr, col_name_expr)

Shorthand for indexing into a `ColumnwiseBlockMatrix`. Generates a tuple that
contains two `NestedPropertyName`s: one for the row of the block matrix and
another for the column. If either the row expression or the column expression is
`*`, the corresponding `NestedPropertyName` will be the trivial one.

# Examples
```julia-repl
julia> @block_name(c.ρ, f.w.components.data.:1)
(NestedPropertyName(c.ρ), NestedPropertyName(f.w.components.data.:1))
```
"""
macro block_name(row_name_expr, col_name_expr)
    return :(
        $(NestedPropertyName{flatten_nested_property_expr(row_name_expr)}()),
        $(NestedPropertyName{flatten_nested_property_expr(col_name_expr)}()),
    )
end
flatten_nested_property_expr(name) = name == :* ? () : (name,)
function flatten_nested_property_expr(expr::Expr)
    expr.head == :. || error("@block_name only supports expressions with .")
    return (flatten_nested_property_expr(expr.args[1])..., expr.args[2].value)
end

const ColumnwiseBandMatrixField = Field{<:AbstractData{<:BandMatrixRow}}

const BlockNameAndValueType = Pair{
    <:Tuple{NestedPropertyName, NestedPropertyName},
    <:Union{ColumnwiseBandMatrixField, LinearAlgebra.UniformScaling},
}

# This doesn't need to actually act like a matrix, since we have to permute its
# data before doing any computations. So, it can act like a dictionary instead.
struct ColumnwiseBlockMatrix{B <: NTuple{N, BlockNameAndValueType} where {N}}
    block_names_and_values::B
end
ColumnwiseBlockMatrix(block_names_and_values...) =
    ColumnwiseBlockMatrix(block_names_and_values)

Base.getindex(matrix::ColumnwiseBlockMatrix, block_name) =
    _get_block(block_name, matrix.block_names_and_values...)
_get_block(_) = nothing
_get_block(block_name, block_name_and_value, block_names_and_values...) =
    block_name_and_value.first == block_name ? block_name_and_value.second :
    _get_block(block_name, block_names_and_values...)

Base.iterate(matrix::ColumnwiseBlockMatrix, state = 1) =
    state > length(matrix.block_names_and_values) ? nothing :
    (matrix.block_names_and_values[state], state + 1)

matching_space(space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)
matching_space(space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    Spaces.FaceExtrudedFiniteDifferenceSpace(space)
matching_space(space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)
matching_space(space::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    Spaces.CenterExtrudedFiniteDifferenceSpace(space)

is_center_space(_) = false
is_center_space(::Spaces.CenterFiniteDifferenceSpace) = true
is_center_space(::Spaces.CenterExtrudedFiniteDifferenceSpace) = true
is_face_space(_) = false
is_face_space(::Spaces.FaceFiniteDifferenceSpace) = true
is_face_space(::Spaces.FaceExtrudedFiniteDifferenceSpace) = true

levels(space) = 1:Spaces.nlevels(space)
columns(space) = Spaces.eachslabindex(Spaces.horizontal_space(space))

is_number_field_on_centers(x) =
    x isa Field && eltype(x) <: Number && is_center_space(axes(x))
is_number_field_on_faces(x) =
    x isa Field && eltype(x) <: Number && is_face_space(axes(x))

permuted_outer_diagonals(matrix::ColumnwiseBlockMatrix) =
    _outer_diagonals(map(x -> x.second, matrix.block_names_and_values)...)
function _outer_diagonals(block::ColumnwiseBandMatrixField)
    original_outer_diagonals = outer_diagonals(eltype(block))
    if eltype(original_outer_diagonals) isa PlusHalf
        if is_center_space(axes(block))
            return original_outer_diagonals .+ half # (center, face) block
        else
            return original_outer_diagonals .- half # (face, center) block
        end
    else
        return original_outer_diagonals
    end
end
_outer_diagonals(::LinearAlgebra.UniformScaling) = (0, 0)
function _outer_diagonals(block, blocks...)
    ld1, ud1 = _outer_diagonals(block)
    ld2, ud2 = _outer_diagonals(blocks...)
    return min(ld1, ld2), max(ud1, ud2)
end

struct BlockMatrixSystemSolver{PX, PA, C}
    permuted_x::PX
    permuted_A::PA
    permuted_b::PX
    cache::C
end

function BlockMatrixSystemSolver(x, A::ColumnwiseBlockMatrix)
    center_field_names =
        find_all_nested_properties(is_number_field_on_centers, x)
    face_field_names = find_all_nested_properties(is_number_field_on_faces, x)
    all_field_names = (center_field_names..., face_field_names...)
    x_center_fields =
        map(name -> get_nested_property(x, name), center_field_names)
    x_face_fields = map(name -> get_nested_property(x, name), face_field_names)

    center_space = axes(x_center_fields[1])
    for field in x_center_fields
        axes(field) === center_space ||
            error("Center spaces in x are not all identical")
    end
    for field in x_face_fields
        axes(field) === matching_space(center_space) ||
            error("Face space in x does not match center space in x")
    end
    for ((row_name, col_name), block) in A
        row_name in all_field_names ||
            error("Row name $row_name does not correspond to a field in x")
        col_name in all_field_names ||
            error("Column name $col_name does not correspond to a field in x")
        if block isa ColumnwiseBandMatrixField
            axes(get_nested_property(x, row_name)) === axes(block) ||
                error("Row field space is not identical to block field space")
            axes(get_nested_property(x, col_name)) === (
                eltype(outer_diagonals(eltype(block))) isa PlusHalf ?
                matching_space(axes(block)) : axes(block)
            ) || error(
                "Column field space is not consistent with block field space \
                 and block field type",
            )
        end
    end

    FT = eltype(parent(x_center_fields[1]))
    DA = Device.device_array_type(x_center_fields[1])
    n_fields = length(all_field_names)
    n_cells = length(levels(center_space))
    n_cols = length(columns(center_space))
    PermutedVectorBlockType = SVector{n_fields, FT}
    PermutedMatrixBlockRowType = band_matrix_row_type(
        permuted_outer_diagonals(A)...,
        SMatrix{n_fields, n_fields, FT},
    )
    # TODO: If the solver is too slow, we should make PermutedMatrixBlockRowType
    # use a specialized wrapper for SMatrix that has fast methods for diagonal
    # and arrowhead matrices.

    columnwise_array(::Type{T}) where {T} = DA{T}(undef, n_cells, n_cols)
    permuted_x = columnwise_array(PermutedVectorBlockType)
    permuted_A = columnwise_array(PermutedMatrixBlockRowType)
    permuted_b = columnwise_array(PermutedVectorBlockType)
    cache = if PermutedMatrixBlockRowType <: DiagonalMatrixRow
        (;)
    elseif PermutedMatrixBlockRowType <: TridiagonalMatrixRow
        (;
            A₊₁′ = columnwise_array(eltype(PermutedMatrixBlockRowType)),
            b′ = columnwise_array(PermutedVectorBlockType),
        )
    elseif PermutedMatrixBlockRowType <: PentadiagonalMatrixRow
        (;) # TODO
    else
        error("BlockMatrixSystemSolver is only implemented for block diagonal, \
               block tridiagonal, and block pentadiagonal matrices (after \
               permutation of rows and columns)")
    end
    return BlockMatrixSystemSolver(permuted_x, permuted_A, permuted_b, cache)
end

function (linear_solve!::BlockMatrixSystemSolver)(x, A, b)
    (; permuted_x, permuted_A, permuted_b, cache) = linear_solve!
    center_field_names =
        find_all_nested_properties(is_number_field_on_centers, x)
    face_field_names = find_all_nested_properties(is_number_field_on_faces, x)
    all_field_names = (center_field_names..., face_field_names...)
    x_center_fields =
        map(name -> get_nested_property(x, name), center_field_names)
    x_face_fields = map(name -> get_nested_property(x, name), face_field_names)
    b_center_fields =
        map(name -> get_nested_property(b, name), center_field_names)
    b_face_fields = map(name -> get_nested_property(b, name), face_field_names)

    b_data_layouts =
        map(Fields.field_values, (b_center_fields..., b_face_fields...))
    for (col, col_indices) in enumerate(columns(b_center_fields[1]))
        for cell in levels(b_center_fields[1])
            for (index, data_layout) in enumerate(b_data_layouts)
                @inbounds permuted_b[cell, col][index] = DataLayouts.level(
                    DataLayouts.column(data_layout, col_indices...),
                    cell,
                )
            end
        end
    end

    # TODO: Finish this.
end

# A direct implementation of the (first) algorithm presented in
# https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm, but with the
# following variable name substitutions: a → A₋₁, b → A₀, c → A₊₁, c′ → A₊₁′,
# d → b, and d′ → b′.
# TODO: How should this be made compatible with GPUs?
function tridiagonal_solve_column!(x, b, A₋₁, A₀, A₊₁, A₊₁′, b′)
    @assert allequal(length.((x, b, A₋₁, A₀, A₊₁, A₊₁′, b′)))
    n = length(x)
    @inbounds begin
        A₊₁′[1] = A₊₁[1] / A₀[1]
        for i in 2:(n - 1)
            A₊₁′[i] = A₊₁[i] / (A₀[i] - A₋₁[i] * A₊₁′[i - 1])
        end
        b′[1] = b[1] / A₀[1]
        for i in 2:n
            b′[i] = (b[i] - A₋₁[i] * b′[i - 1]) / (A₀[i] - A₋₁[i] * A₊₁′[i - 1])
        end
        x[n] = b′[n]
        for i in (n - 1):-1:1
            x[i] = b′[i] - A₊₁′[i] * x[i + 1]
        end
    end
end

# TODO: https://github.com/GeoStat-Framework/pentapy/blob/main/pentapy/solver.pyx or
# https://www.mathworks.com/matlabcentral/fileexchange/4671-fast-pentadiagonal-system-solver
