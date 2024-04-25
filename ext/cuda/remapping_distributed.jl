import ClimaCore: Topologies, Spaces, Fields
import ClimaComms
import CUDA
using CUDA: @cuda
import ClimaCore.Remapping: _set_interpolated_values_device!


function _set_interpolated_values_device!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_field_values,
    local_horiz_indices,
    interpolation_matrix,
    vert_interpolation_weights::AbstractArray,
    vert_bounding_indices::AbstractArray,
    ::ClimaComms.CUDADevice,
)
    # FIXME: Avoid allocation of tuple
    field_values = tuple(map(f -> Fields.field_values(f), fields)...)
    nblocks, _ = size(interpolation_matrix[1])
    nthreads = length(vert_interpolation_weights)
    args = (
        out,
        interpolation_matrix,
        local_horiz_indices,
        vert_interpolation_weights,
        vert_bounding_indices,
        field_values,
    )
    auto_launch!(
        set_interpolated_values_kernel!,
        args,
        out;
        threads_s = (nthreads),
        blocks_s = (nblocks),
    )
end

# GPU, 3D case
function set_interpolated_values_kernel!(
    out::AbstractArray,
    (I1, I2)::NTuple{2},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
)
    # TODO: Check the memory access pattern. This was not optimized and likely inefficient!
    num_horiz = length(local_horiz_indices)
    num_vert = length(vert_bounding_indices)
    num_fields = length(field_values)

    hindex = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    vindex = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    findex = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    totalThreadsX = gridDim().x * blockDim().x
    totalThreadsY = gridDim().y * blockDim().y
    totalThreadsZ = gridDim().z * blockDim().z

    _, Nq = size(I1)

    for i in hindex:totalThreadsX:num_horiz
        h = local_horiz_indices[i]
        for j in vindex:totalThreadsY:num_vert
            v_lo, v_hi = vert_bounding_indices[j]
            A, B = vert_interpolation_weights[j]
            for k in findex:totalThreadsZ:num_fields
                if i ≤ num_horiz && j ≤ num_vert && k ≤ num_fields
                    out[i, j, k] = 0
                    for t in 1:Nq, s in 1:Nq
                        out[i, j, k] +=
                            I1[i, t] *
                            I2[i, s] *
                            (
                                A * field_values[k][t, s, nothing, v_lo, h] +
                                B * field_values[k][t, s, nothing, v_hi, h]
                            )
                    end
                end
            end
        end
    end
    return nothing
end

# GPU, 2D case
function set_interpolated_values_kernel!(
    out::AbstractArray,
    (I,)::NTuple{1},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
)
    # TODO: Check the memory access pattern. This was not optimized and likely inefficient!
    num_horiz = length(local_horiz_indices)
    num_vert = length(vert_bounding_indices)
    num_fields = length(field_values)

    hindex = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    vindex = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    findex = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    totalThreadsX = gridDim().x * blockDim().x
    totalThreadsY = gridDim().y * blockDim().y
    totalThreadsZ = gridDim().z * blockDim().z

    _, Nq = size(I)

    for i in hindex:totalThreadsX:num_horiz
        h = local_horiz_indices[i]
        for j in vindex:totalThreadsY:num_vert
            v_lo, v_hi = vert_bounding_indices[j]
            A, B = vert_interpolation_weights[j]
            for k in findex:totalThreadsZ:num_fields
                if i ≤ num_horiz && j ≤ num_vert && k ≤ num_fields
                    out[i, j, k] = 0
                    for t in 1:Nq
                        out[i, j, k] +=
                            I[i, t] *
                            I[i, s] *
                            (
                                A *
                                field_values[k][t, nothing, nothing, v_lo, h] +
                                B *
                                field_values[k][t, nothing, nothing, v_hi, h]
                            )
                    end
                end
            end
        end
    end
    return nothing
end

function _set_interpolated_values_device!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    _scratch_field_values,
    local_horiz_indices,
    local_horiz_interpolation_weights,
    ::Nothing,
    ::Nothing,
    ::ClimaComms.CUDADevice,
)

    space = axes(first(fields))
    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quad)

    hdims = length(local_horiz_interpolation_weights)
    hdims in (1, 2) || error("Cannot handle $hdims horizontal dimensions")

    # FIXME: Avoid allocation of tuple
    field_values = tuple(map(f -> Fields.field_values(f), fields)...)
    nitems = length(out)
    nthreads, nblocks = _configure_threadblock(nitems)
    args = (
        out,
        local_horiz_interpolation_weights,
        local_horiz_indices,
        field_values,
    )
    auto_launch!(
        set_interpolated_values_kernel!,
        args,
        out;
        threads_s = (nthreads),
        blocks_s = (nblocks),
    )
end

function set_interpolated_values_kernel!(
    out,
    (I1, I2)::NTuple{2},
    local_horiz_indices,
    field_values,
)
    # TODO: Check the memory access pattern. This was not optimized and likely inefficient!
    num_horiz = length(local_horiz_indices)
    num_fields = length(field_values)

    hindex = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    findex = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    totalThreadsX = gridDim().x * blockDim().x
    totalThreadsZ = gridDim().z * blockDim().z

    _, Nq = size(I1)

    for i in hindex:totalThreadsX:num_horiz
        h = local_horiz_indices[i]
        for k in findex:totalThreadsZ:num_fields
            if i ≤ num_horiz && k ≤ num_fields
                out[i, k] = 0
                for t in 1:Nq, s in 1:Nq
                    out[i, k] +=
                        I1[i, t] *
                        I2[i, s] *
                        field_values[k][t, s, nothing, nothing, h]
                end
            end
        end
    end
    return nothing
end

function set_interpolated_values_kernel!(
    out::AbstractArray,
    (I,)::NTuple{1},
    local_horiz_indices,
    field_values,
)
    # TODO: Check the memory access pattern. This was not optimized and likely inefficient!
    num_horiz = length(local_horiz_indices)
    num_fields = length(field_values)

    hindex = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    findex = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    totalThreadsX = gridDim().x * blockDim().x
    totalThreadsZ = gridDim().z * blockDim().z

    _, Nq = size(I)

    for i in hindex:totalThreadsX:num_horiz
        h = local_horiz_indices[i]
        for k in findex:totalThreadsZ:num_fields
            if i ≤ num_horiz && k ≤ num_fields
                out[i, k] = 0
                for t in 1:Nq, s in 1:Nq
                    out[i, k] +=
                        I[i, i] *
                        field_values[k][t, nothing, nothing, nothing, h]
                end
            end
        end
    end
    return nothing
end
