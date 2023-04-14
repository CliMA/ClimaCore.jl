"""
    BandMatrixRow{ld}(entries...)

Stores the nonzero entries in a row of a band matrix, starting with the lowest
diagonal, which has index `ld`. Supported operations include accessing the entry
on the diagonal with index `d` by calling `row[d]`, as well as taking linear
combinations with other band matrix rows. Diagonal matrix rows also support
taking inverses by calling `inv(row)`.
"""
struct BandMatrixRow{ld, bw, T <: Number}
    entries::NTuple{bw, T}
end
# The parameter bw is the "bandwidth", i.e., the number of nonzero entries.
# We need something like T <: Number to ensure that the LinearSystemSolver makes
# sense---in order to "ldiv!" by a band matrix, the entries of the matrix must
# support taking inverses.

BandMatrixRow{ld}(entries::T...) where {ld, T} =
    BandMatrixRow{ld, length(entries), T}(entries)

function (::Type{BMR})(row::BandMatrixRow) where {BMR <: BandMatrixRow}
    ld, ud = outer_diagonals(typeof(row))
    new_ld, new_ud = outer_diagonals(BMR)
    if new_ld > ld || new_ud < ud
        error(
            "Cannot convert a band matrix with diagonals in the range $ld:$ud to
             a band matrix with diagonals in the range $new_ld:$new_ud",
        )
    end
    first_zeros = ntuple(_ -> zero(eltype(BMR)), Val(ld - new_ld))
    last_zeros = ntuple(_ -> zero(eltype(BMR)), Val(new_ud - ud))
    return BMR(first_zeros..., row.entries..., last_zeros...)
end

"""
    outer_diagonals(::Type{<:BandMatrixRow})

Gets the indices of the lower and upper diagonals, `ld` and `ud`, of the given
`BandMatrixRow` type.
"""
outer_diagonals(::Type{<:BandMatrixRow{ld, bw}}) where {ld, bw} =
    (ld, ld + bw - 1)

"""
    band_matrix_row_type(ld, ud, T)

Returns the type of a band matrix with diagonals in the range `ld:ud` and
entries of type `T`.
"""
band_matrix_row_type(ld, ud, ::Type{T}) where {T} =
    BandMatrixRow{ld, ud - ld + 1, T}
# TODO: Should this use @inline or Val to enforce constant propagation?

Base.eltype(::Type{<:BandMatrixRow{ld, bw, T}}) where {ld, bw, T} = T

Base.convert(::Type{BMR}, row::BandMatrixRow) where {BMR <: BandMatrixRow} =
    BMR(row)

function Base.promote_rule(
    ::Type{BMR1},
    ::Type{BMR2},
) where {BMR1 <: BandMatrixRow, BMR2 <: BandMatrixRow}
    ld1, ud1 = outer_diagonals(BMR1)
    ld2, ud2 = outer_diagonals(BMR2)
    ld, ud = min(ld1, ld2), max(ud1, ud2)
    T = promote_type(eltype(BMR1), eltype(BMR2))
    return band_matrix_row_type(ld, ud, T)
end

Base.@propagate_inbounds Base.getindex(row::BandMatrixRow, d) =
    row.entries[d - outer_diagonals(typeof(row))[1] + 1]

Base.map(f, rows::BMR...) where {BMR <: BandMatrixRow} =
    BMR(map(f, map(row -> row.entries, rows)...)...)
Base.map(f, rows::BandMatrixRow...) = map(f, promote(rows...)...)

# Define all necessary operations for computing linear combinations of matrices.
for op in (:+, :-)
    @eval begin
        Base.:($op)(row::BandMatrixRow) = map($op, row)
        Base.:($op)(row1::BandMatrixRow, row2::BandMatrixRow) =
            map($op, row1, row2)
        Base.:($op)(row::BandMatrixRow, diag::LinearAlgebra.UniformScaling) =
            $op(row, BandMatrixRow{0}(diag.λ))
        Base.:($op)(diag::LinearAlgebra.UniformScaling, row::BandMatrixRow) =
            $op(BandMatrixRow{0}(diag.λ), row)
    end
end
for op in (:*, :/, :÷, :%)
    @eval begin
        Base.:($op)(row::BandMatrixRow, scalar::Number) =
            map(Base.Fix2($op, scalar), row)
    end
end
for op in (:*, :\)
    @eval begin
        Base.:($op)(scalar::Number, row::BandMatrixRow) =
            map(Base.Fix1($op, scalar), row)
    end
end

# Don't implement the Hadamard product in order to avoid accidental ⋅/* typos.
Base.:*(::BandMatrixRow, ::BandMatrixRow) =
    error("Band matrices must be multiplied using ⋅, not *")

Base.inv(row::BandMatrixRow{0, 1}) = BandMatrixRow{0}(inv(row[0]))
function Base.inv(row::BandMatrixRow)
    ld, ud = outer_diagonals(typeof(row))
    error_reason_string = if !(ld isa Integer)
        "it is not a square matrix"
    elseif ld > 0 || ud < 0
        "it is singular"
    else
        "its inverse is a dense matrix, which cannot be efficiently \
         represented/manipulated using ClimaCore Fields and Operators"
    end
    error(
        "Cannot take the inverse of a band matrix with diagonals in the range \
         $ld:$ud because $error_reason_string",
    )
end


################################################################################
################################################################################
################################################################################


# Aliases used for dispatch and for allocating matrix fields.
const ColumnwiseBandMatrix{ld, bw} =
    Field{<:AbstractData{<:BandMatrixRow{ld, bw}}}
const ColumnwiseDiagonalMatrix = ColumnwiseBandMatrix{0, 1}
const ColumnwiseBidiagonalMatrix = ColumnwiseBandMatrix{-half, 2}
const ColumnwiseTridiagonalMatrix = ColumnwiseBandMatrix{-1, 3}
const ColumnwiseQuaddiagonalMatrix = ColumnwiseBandMatrix{-1 - half, 4}
const ColumnwisePentadiagonalMatrix = ColumnwiseBandMatrix{-2, 5}


################################################################################
################################################################################
################################################################################


"""
    MultiplyColumnwiseBandMatrix

An operator used to multiply a band matrix field by a scalar field or by another
band matrix field, i.e., matrix-vector or matrix-matrix multiplication. The `⋅`
symbol is an alias for `MultiplyColumnwiseBandMatrix()`.
"""
struct MultiplyColumnwiseBandMatrix <: FiniteDifferenceOperator end
const ⋅ = MultiplyColumnwiseBandMatrix()

#=
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
If M[idx, idx′] is only defined when idx + ld ≤ idx′ ≤ idx + ub, this becomes
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

Consider a BandMatrixRow field M and another BandMatrixRow field M′.
From the definition of matrix-matrix multiplication,
    (M ⋅ M′)[idx, idx′] = ∑_{idx′′} M[idx, idx′′] * M′[idx′′, idx′].
If M′[idx′′] is only defined when left_idx ≤ idx′′ ≤ right_idx, this becomes
    (M ⋅ M′)[idx, idx′] =
        ∑_{idx′′ ∈ left_idx:right_idx} M[idx, idx′′] * M′[idx′′, idx′].
If M[idx, idx′′] is only defined when idx + ld ≤ idx′′ ≤ idx + ud, this becomes
    (M ⋅ M′)[idx, idx′] =
        ∑_{idx′′ ∈ max(left_idx, idx + ld):min(right_idx, idx + ud)}
            M[idx, idx′′] * M′[idx′′, idx′].
If M′[idx′′, idx′] is only defined when idx′′ + ld′ ≤ idx′ ≤ idx′′ + ud′, or,
equivalently, when idx′ - ud′ ≤ idx′′ ≤ idx′ - ld′, this becomes
    (M ⋅ M′)[idx, idx′] =
        ∑_{
            idx′′ ∈
                max(left_idx, idx + ld, idx′ - ud′):
                min(right_idx, idx + ud, idx′ - ld′)
        } M[idx, idx′′] * M′[idx′′, idx′].
Replacing the variable idx′ with the variable prod_d = idx′ - idx gives us
    (M ⋅ M′)[idx, idx + prod_d] =
        ∑_{
            idx′′ ∈
                max(left_idx, idx + ld, idx + prod_d - ud′):
                min(right_idx, idx + ud, idx + prod_d - ld′)
        } M[idx, idx′′] * M′[idx′′, idx + prod_d].
Replacing the variable idx′′ with the variable d = idx′′ - idx gives us
    (M ⋅ M′)[idx, idx + prod_d] =
        ∑_{
            d ∈
                max(left_idx - idx, ld, prod_d - ud′):
                min(right_idx - idx, ud, prod_d - ld′)
        } M[idx, idx + d] * M′[idx + d, idx + prod_d].
This can be rewritten using the standard indexing notation as
    (M ⋅ M′)[idx][prod_d] =
        ∑_{
            d ∈
                max(left_idx - idx, ld, prod_d - ud′):
                min(right_idx - idx, ud, prod_d - ld′)
        } M[idx][d] * M′[idx + d][prod_d - d].
Finally, we can express this in terms of left/right boundaries and an interior:
    (M ⋅ M′)[idx][prod_d] =
        ∑_{
            d ∈
                if idx < left_idx - ld
                    max(left_idx - idx, prod_d - ud′):min(ud, prod_d - ld′)
                elseif idx > right_idx - ud
                    max(ld, prod_d - ud′):min(right_idx - idx, prod_d - ld′)
                else
                    max(ld, prod_d - ud′):min(ud, prod_d - ld′)
                end
        } M[idx][d] * M′[idx + d][prod_d - d].
We only need to define (M ⋅ M′)[idx][prod_d] when it has a nonzero value in the
interior, which will be the case when
    max(ld, prod_d - ud′) ≤ min(ud, prod_d - ld′).
This can be rewritten as a system of four inequalities:
    ld ≤ ud, ld ≤ prod_d - ld′, prod_d - ud′ ≤ ud, prod_d - ud′ ≤ prod_d - ld′.
By definition, ld ≤ ud and ld′ ≤ ud′, so the first and last inequality are
always true. Rearranging the remaining two inequalities gives us
    ld + ld′ ≤ prod_d ≤ ud + ud′.
=#

struct TopLeftMatrixCorner <: BoundaryCondition end
struct BottomRightMatrixCorner <: BoundaryCondition end

has_boundary(
    ::MultiplyColumnwiseBandMatrix,
    ::LeftBoundaryWindow{name},
) where {name} = true
has_boundary(
    ::MultiplyColumnwiseBandMatrix,
    ::RightBoundaryWindow{name},
) where {name} = true

get_boundary(
    ::MultiplyColumnwiseBandMatrix,
    ::LeftBoundaryWindow{name},
) where {name} = TopLeftMatrixCorner()
get_boundary(
    ::MultiplyColumnwiseBandMatrix,
    ::RightBoundaryWindow{name},
) where {name} = BottomRightMatrixCorner()

stencil_interior_width(::MultiplyColumnwiseBandMatrix, matrix, arg) =
    ((0, 0), outer_diagonals(eltype(matrix)))

function boundary_width(
    ::MultiplyColumnwiseBandMatrix,
    ::TopLeftMatrixCorner,
    matrix,
    arg,
)
    ld = outer_diagonals(eltype(matrix))[1]
    return max((left_idx(axes(arg)) - ld) - left_idx(axes(matrix)), 0)
end
function boundary_width(
    ::MultiplyColumnwiseBandMatrix,
    ::BottomRightMatrixCorner,
    matrix,
    arg,
)
    ud = outer_diagonals(eltype(matrix))[2]
    return max(right_idx(axes(matrix)) - (right_idx(axes(arg)) - ud), 0)
end

function product_matrix_outer_diagonals(matrix1, matrix2)
    ld1, ud1 = outer_diagonals(eltype(matrix1))
    ld2, ud2 = outer_diagonals(eltype(matrix2))
    return (ld1 + ld2, ud1 + ud2)
end

# TODO: This is not correct for the same reason as the other two-argument finite
# difference operators---it assumes that the result of multiplying two values
# will have the same type as the second value, instead of properly inferring the
# result type.
function return_eltype(::MultiplyColumnwiseBandMatrix, matrix, arg)
    if !(eltype(matrix) <: BandMatrixRow)
        error("⋅ should only be used after a ColumnwiseBandMatrix")
    end
    return if eltype(arg) <: BandMatrixRow # matrix-matrix multiplication
        prod_ld, prod_ud = product_matrix_outer_diagonals(matrix, arg)
        band_matrix_row_type(prod_ld, prod_ud, eltype(eltype(arg)))
    else # matrix-vector multiplication
        eltype(arg)
    end
end

return_space(::MultiplyColumnwiseBandMatrix, matrix_space, _) = matrix_space

# TODO: Propagate @inbounds through the anonymous functions.
Base.@propagate_inbounds function mul_cbm_at_idx(
    loc,
    idx,
    hidx,
    matrix,
    arg;
    ld = nothing,
    ud = nothing,
)
    if isnothing(ld)
        ld = outer_diagonals(eltype(matrix))[1]
    end
    if isnothing(ud)
        ud = outer_diagonals(eltype(matrix))[2]
    end
    return if eltype(arg) <: BandMatrixRow # matrix-matrix multiplication
        arg_ld, arg_ud = outer_diagonals(eltype(arg))
        prod_ld, prod_ud = product_matrix_outer_diagonals(matrix, arg)
        entries = map(prod_ld:prod_ud) do prod_d
            mapreduce(⊞, max(ld, prod_d - arg_ud):min(ud, prod_d - arg_ld)) do d
                getidx(matrix, loc, idx, hidx)[d] ⊠
                getidx(arg, loc, idx + d, hidx)[prod_d - d]
            end
        end
        BandMatrixRow{prod_ld}(entries...)
    else # matrix-vector multiplication
        mapreduce(⊞, ld:ud) do d
            getidx(matrix, loc, idx, hidx)[d] ⊠ getidx(arg, loc, idx + d, hidx)
        end
    end
end

Base.@propagate_inbounds stencil_interior(
    ::MultiplyColumnwiseBandMatrix,
    loc,
    idx,
    hidx,
    matrix,
    arg,
) = mul_cbm_at_idx(loc, idx, hidx, matrix, arg)

Base.@propagate_inbounds stencil_left_boundary(
    ::MultiplyColumnwiseBandMatrix,
    ::TopLeftMatrixCorner,
    loc,
    idx,
    hidx,
    matrix,
    arg,
) = mul_cbm_at_idx(loc, idx, hidx, matrix, arg; ld = left_idx(axes(arg)) - idx)

Base.@propagate_inbounds stencil_right_boundary(
    ::MultiplyColumnwiseBandMatrix,
    ::BottomRightMatrixCorner,
    loc,
    idx,
    hidx,
    matrix,
    arg,
) = mul_cbm_at_idx(loc, idx, hidx, matrix, arg; ud = right_idx(axes(arg)) - idx)


################################################################################
################################################################################
################################################################################


"""
    LinearSystemSolver(A_prototype)

An object that used to solve linear systems of the form `A * x = b` by calling
`solver(x, A, b)`. `A_prototype` must satisfy `A_prototype = similar(A)`.
"""
struct LinearSystemSolver{C <: NamedTuple}
    cache::C
end

scalar_field(A_prototype::ColumnwiseBandMatrix) =
    similar(A_prototype, eltype(eltype(A_prototype)))
make_linear_system_solver(; kwargs...) = LinearSystemSolver(values(kwargs))

function LinearSystemSolver(A_prototype::ColumnwiseBandMatrix)
    ld, ud = outer_diagonals(eltype(A_prototype))
    error("LinearSystemSolver is not yet implemented for band matrices with \
           diagonals in the range $ld:$ud")
end

LinearSystemSolver(::ColumnwiseDiagonalMatrix) = make_linear_system_solver()
LinearSystemSolver(A_prototype::ColumnwiseTridiagonalMatrix) =
    make_linear_system_solver(;
        c′ = scalar_field(A_prototype),
        d′ = scalar_field(A_prototype),
    )
LinearSystemSolver(A_prototype::ColumnwisePentadiagonalMatrix) =
    make_linear_system_solver(;
        α = scalar_field(A_prototype),
        β = scalar_field(A_prototype),
        γ = scalar_field(A_prototype),
        δ = scalar_field(A_prototype),
        c = scalar_field(A_prototype),
        z = scalar_field(A_prototype),
    )

(solver::LinearSystemSolver)(x, A::ColumnwiseDiagonalMatrix, b) =
    @. x = inv(A) ⋅ b
# TODO: Should this be @. x = b / A.entries.:1?

# A direct implementation of the (first) algorithm presented in
# https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm, but with the
# following variable name substitutions: a → A₋₁, b → A₀, c → A₊₁, c′ → A₊₁′,
# d → b, d′ → b′, i → v, and n → Nv.
# TODO: How should this be made compatible with GPUs?
# TODO: How should this check whether x, A, and b are on compatible spaces?
# TODO: Should this avoid allocating Nv levels if it only needs Nv - 1 levels?
function (solver::LinearSystemSolver)(x, A::ColumnwiseTridiagonalMatrix, b)
    Fields.bycolumn(axes(A)) do colidx
        A₋₁, A₀, A₊₁ = Fields.field_values(A.entries[colidx])
        b = Fields.field_values(b[colidx])
        A₊₁′, b′ = getindex.(Fields.field_values.(solver.cache), colidx)
        Nv = Spaces.nlevels(axes(A))
        @inbounds begin
            A₊₁′[1] = A₊₁[1] / A₀[1]
            for v in 2:(Nv - 1)
                A₊₁′[v] = A₊₁[v] / (A₀[v] - A₋₁[v] * A₊₁′[v - 1])
            end
            b′[1] = b[1] / A₀[1]
            for v in 2:Nv
                b′[v] =
                    (b[v] - A₋₁[v] * b′[v - 1]) / (A₀[v] - A₋₁[v] * A₊₁′[v - 1])
            end
            x[Nv] = b′[Nv]
            for v in (Nv - 1):-1:1
                x[v] = b′[v] - A₊₁′[v] * x[v + 1]
            end
        end
    end
    return x
end

function (solver::LinearSystemSolver)(x, A::ColumnwisePentadiagonalMatrix, b)
    (; α, β, γ, δ, c, z) = solver.cache
    # TODO: https://github.com/GeoStat-Framework/pentapy/blob/main/pentapy/solver.pyx or
    # https://www.mathworks.com/matlabcentral/fileexchange/4671-fast-pentadiagonal-system-solver
end


################################################################################
################################################################################
################################################################################


# This is the matrix with respect to the last argument of the operator.
struct FDOperatorTermsMatrix{O <: FiniteDifferenceOperator} <:
       FiniteDifferenceOperator
    op::O
end

has_boundary(
    matrix_op::FDOperatorTermsMatrix,
    bw::LeftBoundaryWindow{name},
) where {name} = has_boundary(matrix_op.op, bw)
has_boundary(
    matrix_op::FDOperatorTermsMatrix,
    bw::RightBoundaryWindow{name},
) where {name} = has_boundary(matrix_op.op, bw)

get_boundary(
    matrix_op::FDOperatorTermsMatrix,
    bw::LeftBoundaryWindow{name},
) where {name} = get_boundary(matrix_op.op, bw)
get_boundary(
    matrix_op::FDOperatorTermsMatrix,
    bw::RightBoundaryWindow{name},
) where {name} = get_boundary(matrix_op.op, bw)

stencil_interior_width(matrix_op::FDOperatorTermsMatrix, args...) =
    stencil_interior_width(matrix_op.op, args...)

boundary_width(
    matrix_op::FDOperatorTermsMatrix,
    bc::BoundaryCondition,
    args...,
) = boundary_width(matrix_op.op, bc, args...)

function return_eltype(matrix_op::FDOperatorTermsMatrix, args...)
    ld, ud = stencil_interior_width(matrix_op.op, args...)[end]
    return band_matrix_row_type(ld, ud, return_eltype(matrix_op.op, args...))
end

return_space(matrix_op::FDOperatorTermsMatrix, spaces...) =
    return_space(matrix_op.op, spaces...)

# TODO: Figure out how to rewrite finitedifference.jl to simplify the methods
# of stencil_interior, stencil_left_boundary, and stencil_right_boundary for
# FDOperatorTermsMatrix.


################################################################################
################################################################################
################################################################################


abstract type BlockMatrix end

abstract type InvertibleBlockMatrix <: BlockMatrix end

#=
D 0 0 0
D D 0 0
D D D 0
D D D D
=#
struct InvertibleBlockLowerTriangularMatrix{B} <: InvertibleBlockMatrix
    blocks::B # can be Nothing, UniformScaling, or ColumnwiseDiagonalMatrix
end

#=
A B
C D
=#
struct SchurComplementBlockMatrix{A <: InvertibleBlockMatrix, B, C, D}
    a::A
    b::B
    c::C
    d::D
end

# TODO: Add BlockArrowheadMatrix, or maybe just GenericBlockMatrix, since both
# require iterative solvers.

# TODO: Add methods for LinearSystemSolver.

#=
Requirements for BlockMatrix:
    - Get or set a block: matrix[2, 1] .= ∂ᶜρeₜ∂ᶜρ
    - Construct a sub-matrix: matrix[1:4, 3]
    - Compute a linear combination: @. C1 * matrix1 + C2 * matrix2 + C3 * I
    - Multiply by a FieldVector: @. matrix ⋅ vector
    - Multiply by another BlockMatrix: @. matrix1 ⋅ matrix2
    - Run a LinearSystemSolver with a FieldVector: solver(x, matrix, b)
Good things to have:
    - Get or set a block symbolically: matrix[@var(c.ρe), @var(c.ρ)] .= ∂ᶜρeₜ∂ᶜρ
      (this would also make it easier to implement matrix-vector multiplication)
=#
