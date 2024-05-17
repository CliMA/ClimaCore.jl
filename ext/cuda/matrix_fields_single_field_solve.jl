import CUDA
import ClimaComms
import LinearAlgebra: UniformScaling
import ClimaCore.Operators
import ClimaCore.Fields: Field
import ClimaCore.Fields
import ClimaCore.Spaces
import ClimaCore.Topologies
import ClimaCore.MatrixFields: single_field_solve!
import ClimaCore.MatrixFields: _single_field_solve!
import ClimaCore.MatrixFields: band_matrix_solve!, unzip_tuple_field_values
import ClimaCore.RecursiveApply: ⊠, ⊞, ⊟, rmap, rzero, rdiv

function single_field_solve!(device::ClimaComms.CUDADevice, cache, x, A, b)
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    nitems = Ni * Nj * Nh
    nthreads = min(256, nitems)
    nblocks = cld(nitems, nthreads)
    args = (device, cache, x, A, b)
    auto_launch!(
        single_field_solve_kernel!,
        args,
        x;
        threads_s = nthreads,
        blocks_s = nblocks,
    )
end

function single_field_solve_kernel!(device, cache, x, A, b)
    idx = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    if idx <= Ni * Nj * Nh
        i, j, h = Topologies._get_idx((Ni, Nj, Nh), idx)
        _single_field_solve!(
            device,
            Spaces.column(cache, i, j, h),
            Spaces.column(x, i, j, h),
            Spaces.column(A, i, j, h),
            Spaces.column(b, i, j, h),
        )
    end
    return nothing
end

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
