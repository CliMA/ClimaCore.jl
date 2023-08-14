"""
    MatrixFields

This module adds support for defining and manipulating `Field`s that represent
matrices. Specifically, it specifies the `BandMatrixRow` type, which can be used
to store the entries of a band matrix. A `Field` of `BandMatrixRow`s on a
`FiniteDifferenceSpace` can be interpreted as a band matrix by vertically
concatenating the `BandMatrixRow`s. Similarly, a `Field` of `BandMatrixRow`s on
an `ExtrudedFiniteDifferenceSpace` can be interpreted as a collection of band
matrices, one for each column of the `Field`. Such `Field`s are called
`ColumnwiseBandMatrixField`s, and this module adds the following functionality
for them:
- Constructors, e.g., `matrix_field = @. BidiagonalMatrixRow(field1, field2)`
- Linear combinations, e.g., `@. 3 * matrix_field1 + matrix_field2 / 3`
- Matrix-vector multiplication, e.g., `@. matrix_field ⋅ field`
- Matrix-matrix multiplication, e.g., `@. matrix_field1 ⋅ matrix_field2`
- Compatibility with `LinearAlgebra.I`, e.g., `@. matrix_field = (4I,)` or
    `@. matrix_field - (4I,)`
- Integration with `RecursiveApply`, e.g., the entries of `matrix_field` can be
    `Tuple`s or `NamedTuple`s instead of single values, which allows
    `matrix_field` to represent multiple band matrices at the same time
- Conversions to native array types, e.g., `field2arrays(matrix_field)` can
    convert each column of `matrix_field` into a `BandedMatrix` from
    `BandedMatrices.jl`
- Custom printing, e.g., `matrix_field` gets displayed as a `BandedMatrix`,
    specifically, as the `BandedMatrix` that corresponds to its first column
"""
module MatrixFields

import CUDA: @allowscalar
import LinearAlgebra: UniformScaling, Adjoint, AdjointAbsVec
import StaticArrays: SMatrix, SVector
import BandedMatrices: BandedMatrix, band, _BandedMatrix

import ..Utilities: PlusHalf, half
import ..RecursiveApply:
    rmap, rmaptype, rpromote_type, rzero, rconvert, radd, rsub, rmul, rdiv
import ..DataLayouts: AbstractData
import ..Geometry
import ..Spaces
import ..Fields
import ..Operators

export ⋅
export DiagonalMatrixRow,
    BidiagonalMatrixRow,
    TridiagonalMatrixRow,
    QuaddiagonalMatrixRow,
    PentadiagonalMatrixRow

# Types that are teated as single values when using matrix fields.
const SingleValue =
    Union{Number, Geometry.AxisTensor, Geometry.AdjointAxisTensor}

include("band_matrix_row.jl")
include("rmul_with_projection.jl")
include("matrix_shape.jl")
include("matrix_multiplication.jl")
include("lazy_operators.jl")
include("operator_matrices.jl")
include("field2arrays.jl")

const ColumnwiseBandMatrixField{V, S} = Fields.Field{
    V,
    S,
} where {
    V <: AbstractData{<:BandMatrixRow},
    S <: Union{
        Spaces.FiniteDifferenceSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
}

function Base.show(io::IO, field::ColumnwiseBandMatrixField)
    print(io, eltype(field), "-valued Field")
    if eltype(eltype(field)) <: Number
        shape = typeof(matrix_shape(field)).name.name
        if field isa Fields.FiniteDifferenceField
            println(io, " that corresponds to the $shape matrix")
        else
            println(io, " whose first column corresponds to the $shape matrix")
        end
        column_field = Fields.column(field, 1, 1, 1)
        io = IOContext(io, :compact => true, :limit => true)
        @allowscalar Base.print_array(io, column_field2array_view(column_field))
    else
        # When a BandedMatrix with non-number entries is printed, it currently
        # either prints in an illegible format (e.g., if it has AxisTensor or
        # AdjointAxisTensor entries) or crashes during the evaluation of
        # isassigned (e.g., if it has Tuple or NamedTuple entries). So, for
        # matrix fields with non-number entries, we fall back to the default
        # function for printing fields.
        print(io, ":")
        Fields._show_compact_field(io, field, "  ", true)
    end
end

end
