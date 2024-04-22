import CUDA
import ClimaComms
import LinearAlgebra: UniformScaling
import ClimaCore.Operators
import ClimaCore.Fields: Field
import ClimaCore.Fields
import ClimaCore.MatrixFields: _single_field_solve!
import ClimaCore.MatrixFields: band_matrix_solve!, unzip_tuple_field_values
import ClimaCore.RecursiveApply: ⊠, ⊞, ⊟, rmap, rzero, rdiv

# called by TuplesOfNTuples.jl's `inner_dispatch`:
# which requires a particular argument order:
_single_field_solve!(
    cache::Fields.Field,
    x::Fields.Field,
    A::Union{Fields.Field, UniformScaling},
    b::Fields.Field,
    dev::ClimaComms.CUDADevice,
) = _single_field_solve!(dev, cache, x, A, b)

function _single_field_solve!(
    ::ClimaComms.CUDADevice,
    cache::Fields.ColumnField,
    x::Fields.ColumnField,
    A::Fields.ColumnField,
    b::Fields.ColumnField,
)
    band_matrix_solve!(
        eltype(A),
        unzip_tuple_field_values(Fields.field_values(cache)),
        Fields.field_values(x),
        unzip_tuple_field_values(Fields.field_values(A.entries)),
        Fields.field_values(b),
    )
end

function _single_field_solve!(
    ::ClimaComms.CUDADevice,
    cache::Fields.ColumnField,
    x::Fields.ColumnField,
    A::UniformScaling,
    b::Fields.ColumnField,
)
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)
    n = length(x_data)
    @inbounds for i in 1:n
        x_data[i] = inv(A.λ) ⊠ b_data[i]
    end
end

function _single_field_solve!(
    ::ClimaComms.CUDADevice,
    cache::Fields.PointDataField,
    x::Fields.PointDataField,
    A::UniformScaling,
    b::Fields.PointDataField,
)
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)
    n = length(x_data)
    @inbounds begin
        x_data[] = inv(A.λ) ⊠ b_data[]
    end
end
