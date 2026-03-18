"""
    MatrixFields

This module adds support for defining and manipulating `Field`s that represent
matrices. Specifically, it adds the `BandMatrixRow` type, which can be used
to store the entries of a band matrix. A `Field` of `BandMatrixRow`s on a
`FiniteDifferenceSpace` can be interpreted as a band matrix by vertically
concatenating the `BandMatrixRow`s. Similarly, a `Field` of `BandMatrixRow`s on
an `ExtrudedFiniteDifferenceSpace` can be interpreted as a collection of band
matrices, one for each column of the `Field`. Such `Field`s are called
`ColumnwiseBandMatrixField`s, and this module adds the following functionality
for them:
- Constructors, e.g., `matrix_field = @. BidiagonalMatrixRow(field1, field2)`
- Linear combinations, e.g., `@. 3 * matrix_field1 + matrix_field2 / 3`
- Matrix-vector multiplication, e.g., `@. matrix_field * field`
- Matrix-matrix multiplication, e.g., `@. matrix_field1 * matrix_field2`
- Compatibility with `LinearAlgebra.I`, e.g., `@. matrix_field = (4I,)` or
    `@. matrix_field - (4I,)`
- Compatibility with generic data types, e.g., the entries of `matrix_field` can
    be iterators instead of single values, which allows `matrix_field` to
    represent multiple band matrices at the same time
- Integration with `Operators`, e.g., the `matrix_field` that gets applied to
    the argument of any `FiniteDifferenceOperator` `op` can be obtained using
    the `FiniteDifferenceOperator` `operator_matrix(op)`
- Conversions to native array types, e.g., `field2arrays(matrix_field)` can
    convert each column of `matrix_field` into a `BandedMatrix` from
    `BandedMatrices.jl`
- Custom printing, e.g., `matrix_field` gets displayed as a `BandedMatrix`,
    specifically, as the `BandedMatrix` that corresponds to its first column

This module also adds support for defining and manipulating sparse block
matrices of `Field`s. Specifically, it adds the `FieldMatrix` type, which is a
dictionary that maps pairs of `FieldName`s to `ColumnwiseBandMatrixField`s or
multiples of `LinearAlgebra.I`. This comes with the following functionality:
- Addition and subtraction, e.g., `@. field_matrix1 + field_matrix2`
- Matrix-vector multiplication, e.g., `@. field_matrix * field_vector`
- Matrix-matrix multiplication, e.g., `@. field_matrix1 * field_matrix2`
- The ability to solve linear equations using `FieldMatrixSolver`, which is a
    generalization of `ldiv!` that is designed to optimize solver performance
"""
module MatrixFields

import LinearAlgebra: I, UniformScaling, Adjoint, AdjointAbsVec
import LinearAlgebra: inv, norm, ldiv!, mul!
import StaticArrays: SMatrix, SVector
import BandedMatrices: BandedMatrix, band, _BandedMatrix
import RecursiveArrayTools: recursive_bottom_eltype
import KrylovKit
import ClimaComms
import NVTX
import Adapt
using UnrolledUtilities

import ..Utilities: PlusHalf, half, AutoBroadcaster, is_auto_broadcastable
import ..Utilities: enable_auto_broadcasting, disable_auto_broadcasting
import ..DataLayouts
import ..DataLayouts: AbstractData, broadcast_arg
import ..DataLayouts: vindex
import ..Geometry
import ..Topologies
import ..Spaces
import ..Spaces: local_geometry_type
import ..Fields
import ..Operators
using ..Geometry: mul_with_projection, mul_return_type, axis_tensor_type

export DiagonalMatrixRow,
    BidiagonalMatrixRow,
    TridiagonalMatrixRow,
    QuaddiagonalMatrixRow,
    PentadiagonalMatrixRow
export FieldVectorKeys, FieldMatrixKeys, FieldVectorView, FieldMatrix
export FieldMatrixWithSolver, ⋅

const SingleValue =
    Union{Number, Geometry.AxisTensor, Geometry.AdjointAxisTensor}

include("band_matrix_row.jl")

const ColumnwiseBandMatrixField{V, S} = Fields.Field{
    V, S,
} where {
    V <: AbstractData{<:BandMatrixRow},
    S <: Union{Spaces.AbstractSpace, Operators.PlaceholderSpace}, # so that this can exist inside cuda kernels
}

include("matrix_shape.jl")
include("matrix_multiplication.jl")
include("lazy_operators.jl")
include("operator_matrices.jl")
include("field2arrays.jl")
include("field_name.jl")
include("field_name_set.jl")
include("field_name_dict.jl")
include("single_field_solver.jl")
include("multiple_field_solver.jl")
include("field_matrix_solver.jl")
include("field_matrix_iterative_solver.jl")
include("field_matrix_with_solver.jl")

Base.broadcasted(::Fields.AbstractFieldStyle, ::typeof(*), args...) =
    unrolled_reduce(reverse(args)) do y, x
        (is_field_broadcasted(x) && eltype(x) <: BandMatrixRow) &&
            is_field_broadcasted(y) ?
        Base.broadcasted(MultiplyColumnwiseBandMatrixField(), x, y) :
        Base.Broadcast.Broadcasted(
            Base.Broadcast.combine_styles(x, y),
            *,
            (broadcast_arg(x), broadcast_arg(y)),
        )
    end

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
        ClimaComms.allowscalar(ClimaComms.device(field)) do
            Base.print_array(io, column_field2array_view(column_field))
        end
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
