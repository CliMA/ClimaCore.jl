import ClimaCore.Remapping: interpolate_slab!
import ClimaCore: Topologies, Spaces, Fields, Operators
import CUDA
using CUDA: @cuda
function interpolate_slab!(
    output_array,
    field::Fields.Field,
    slab_indices,
    weights,
    device::ClimaComms.CUDADevice,
)
    space = axes(field)
    FT = Spaces.undertype(space)

    output_cuarray = CuArray(zeros(FT, length(output_array)))
    cuweights = CuArray(weights)
    cuslab_indices = CuArray(slab_indices)

    nitems = length(output_array)
    nthreads, nblocks = _configure_threadblock(nitems)

    args = (output_cuarray, field, cuslab_indices, cuweights)
    auto_launch!(
        interpolate_slab_kernel!,
        args,
        output_cuarray;
        threads_s = (nthreads),
        blocks_s = (nblocks),
    )

    output_array .= Array(output_cuarray)
end

# GPU kernel for 3D configurations
function interpolate_slab_kernel!(
    output_array,
    field,
    slab_indices,
    weights::AbstractArray{Tuple{A, A}},
) where {A}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    space = axes(field)
    FT = Spaces.undertype(space)

    if index <= length(output_array)
        I1, I2 = weights[index]
        Nq1, Nq2 = length(I1), length(I2)

        output_array[index] = zero(FT)

        for j in 1:Nq2, i in 1:Nq1
            ij = CartesianIndex((i, j))
            output_array[index] +=
                I1[i] *
                I2[j] *
                Operators.get_node(space, field, ij, slab_indices[index])
        end
    end
    return nothing
end

# GPU kernel for 2D configurations
function interpolate_slab_kernel!(
    output_array,
    field,
    slab_indices,
    weights::AbstractArray{Tuple{A}},
) where {A}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    space = axes(field)
    FT = Spaces.undertype(space)

    if index <= length(output_array)
        I1, = weights[index]
        Nq = length(I1)

        output_array[index] = zero(FT)

        for i in 1:Nq
            ij = CartesianIndex((i))
            output_array[index] +=
                I1[i] *
                Operators.get_node(space, field, ij, slab_indices[index])
        end
    end
    return nothing
end


# GPU
function interpolate_slab_level!(
    output_array,
    field::Fields.Field,
    vidx_ref_coordinates,
    h::Integer,
    Is::Tuple,
    device::ClimaComms.CUDADevice,
)
    cuvidx_ref_coordinates = CuArray(vidx_ref_coordinates)

    output_cuarray = CuArray(
        zeros(Spaces.undertype(axes(field)), length(vidx_ref_coordinates)),
    )

    nitems = length(vidx_ref_coordinates)
    nthreads, nblocks = _configure_threadblock(nitems)
    args = (output_cuarray, field, cuvidx_ref_coordinates, h, Is)
    auto_launch!(
        interpolate_slab_level_kernel!,
        args,
        out;
        threads_s = (nthreads),
        blocks_s = (nblocks),
    )
    output_array .= Array(output_cuarray)
end

# GPU kernel for 3D configurations
function interpolate_slab_level_kernel!(
    output_array,
    field,
    vidx_ref_coordinates,
    h,
    (I1, I2)::Tuple{<:AbstractArray, <:AbstractArray},
)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    space = axes(field)
    FT = Spaces.undertype(space)
    Nq1, Nq2 = length(I1), length(I2)

    if index <= length(vidx_ref_coordinates)
        v_lo, v_hi, ξ3 = vidx_ref_coordinates[index]

        f_lo = zero(FT)
        f_hi = zero(FT)

        for j in 1:Nq2, i in 1:Nq1
            ij = CartesianIndex((i, j))
            f_lo +=
                I1[i] *
                I2[j] *
                Operators.get_node(space, field, ij, Fields.SlabIndex(v_lo, h))
            f_hi +=
                I1[i] *
                I2[j] *
                Operators.get_node(space, field, ij, Fields.SlabIndex(v_hi, h))
        end
        output_array[index] = ((1 - ξ3) * f_lo + (1 + ξ3) * f_hi) / 2
    end
    return nothing
end

# GPU kernel for 2D configurations
function interpolate_slab_level_kernel!(
    output_array,
    field,
    vidx_ref_coordinates,
    h,
    (I1,)::Tuple{<:AbstractArray},
)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    space = axes(field)
    FT = Spaces.undertype(space)
    Nq = length(I1)

    if index <= length(vidx_ref_coordinates)
        v_lo, v_hi, ξ3 = vidx_ref_coordinates[index]

        f_lo = zero(FT)
        f_hi = zero(FT)

        for i in 1:Nq
            ij = CartesianIndex((i,))
            f_lo +=
                I1[i] *
                Operators.get_node(space, field, ij, Fields.SlabIndex(v_lo, h))
            f_hi +=
                I1[i] *
                Operators.get_node(space, field, ij, Fields.SlabIndex(v_hi, h))
        end

        output_array[index] = ((1 - ξ3) * f_lo + (1 + ξ3) * f_hi) / 2
    end
    return nothing
end
