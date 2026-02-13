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
- Integration with `RecursiveApply`, e.g., the entries of `matrix_field` can be
    `Tuple`s or `NamedTuple`s instead of single values, which allows
    `matrix_field` to represent multiple band matrices at the same time
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
- Integration with `RecursiveApply`, e.g., the entries of `field_matrix` can be
    specified either as matrix `Field`s of `Tuple`s or `NamedTuple`s, or as
    separate matrix `Field`s of single values
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

import ..Utilities: PlusHalf, half
import ..RecursiveApply:
    rmap, rmaptype, rpromote_type, rzero, rconvert, radd, rsub, rmul, rdiv
import ..RecursiveApply: ⊠, ⊞, ⊟
import ..DataLayouts
import ..DataLayouts: AbstractData
import ..DataLayouts: vindex
import ..Geometry
import ..Topologies
import ..Spaces
import ..Spaces: local_geometry_type
import ..Fields
import ..Operators
using ..Geometry:
    rmul_with_projection,
    mul_with_projection,
    axis_tensor_type,
    rmul_return_type,
    project,
    dual,
    SingleValue,
    AdjointAxisVector,
    Axis2TensorOrAdj,
    AxisTensor

export DiagonalMatrixRow,
    BidiagonalMatrixRow,
    TridiagonalMatrixRow,
    QuaddiagonalMatrixRow,
    PentadiagonalMatrixRow
export FieldVectorKeys, FieldMatrixKeys, FieldVectorView, FieldMatrix
export FieldMatrixWithSolver, ⋅

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

const FieldOrStencilStyleType = Union{
    Fields.Field,
    Base.Broadcast.Broadcasted{<:Fields.AbstractFieldStyle},
    Operators.StencilBroadcasted,
    LazyOperatorBroadcasted,
}

function Base.Broadcast.broadcasted(
    ::typeof(*),
    field_or_broadcasted::FieldOrStencilStyleType,
    args...,
)
# @show "ppppppp"

    unrolled_reduce(args; init = field_or_broadcasted) do arg1, arg2
        arg1_isa_matrix =
            eltype(arg1) <: BandMatrixRow || (arg1 isa LazyOperatorBroadcasted)
        if arg1 isa LazyOperatorBroadcasted && length(arg1.args) > 0
            arg1_isa_matrix = eltype(arg1.args[1]) <: BandMatrixRow || arg1.args[1] isa LazyOperatorBroadcasted
        end
        use_matrix_mul_op = arg1_isa_matrix && arg2 isa FieldOrStencilStyleType
        op = use_matrix_mul_op ? MultiplyColumnwiseBandMatrixField() : ⊠
        Base.Broadcast.broadcasted(op, arg1, arg2)
    end
end
Base.Broadcast.broadcasted(
    ::typeof(*),
    single_value_or_broadcasted::SingleValueStyleType,
    field_or_broadcasted::FieldOrStencilStyleType,
    args...,
) = Base.Broadcast.broadcasted(
    ⊠,
    single_value_or_broadcasted,
    Base.Broadcast.broadcasted(*, field_or_broadcasted, args...),
)
# TODO: Generalize this to handle, e.g., @. scalar * scalar * matrix * matrix.

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


project_for_mul(x, y::BandMatrixRow, lg) = map(y_component -> project_for_mul(x, y_component, lg), y)
project_for_mul(x::BandMatrixRow, y::BandMatrixRow, lg) = map(y_component -> project_for_mul(x.entries[1], y_component, lg), y)
project_for_mul(x::BandMatrixRow, y::SingleValue, lg) = project_for_mul(x.entries[1], y, lg)
project_for_mul(x::BandMatrixRow, y, lg) = rmap(y′ -> project_for_mul(x.entries[1], y′, lg), y)
project_for_mul(x, y, lg) = rmap((x′, y′) -> project_for_mul(x′, y′, lg), x, y)
project_for_mul(x::SingleValue, y, lg) = rmap(y′ -> project_for_mul(x, y′, lg), y)
project_for_mul(x, y::SingleValue, lg) = rmap(x′ -> project_for_mul(x′, y, lg), y)
project_for_mul(x::SingleValue, y::SingleValue, lg) = maybe_project(x, y, lg)
maybe_project(_, y, _) = y
maybe_project(x::Union{AdjointAxisVector, Axis2TensorOrAdj}, y::AxisTensor, lg) =
    project(dual(axes(x)[2]), y, lg)

end
