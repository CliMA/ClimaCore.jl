import ClimaCore.Remapping: interpolate_slab!
import ClimaCore: Topologies, Spaces, Fields, Operators, Quadratures
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
        args;
        threads_s = (nthreads),
        blocks_s = (nblocks),
    )
    call_post_op_callback() && post_op_callback(
        output_array,
        output_array,
        field,
        slab_indices,
        weights,
        device,
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

    index <= length(output_array) || return nothing
    space = axes(field)
    FT = Spaces.undertype(space)
    @inbounds begin
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

    index <= length(output_array) || return nothing
    @inbounds begin
        space = axes(field)
        FT = Spaces.undertype(space)
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
        args;
        threads_s = (nthreads),
        blocks_s = (nblocks),
    )
    output_array .= Array(output_cuarray)
end

# GPU kernel for 3D configurations
#
# SOURCE GRID: GLL nodes in the element (where the field is defined). Indexed by (ii, jj).
# TARGET: interpolation point (one per thread); we compute value = Σ I1[i]*I2[j]*field[ii,jj].
#
# Horizontal indexing (per element):
#
#   Spectral (Nq1 = Nq2 = Nq): target uses all source GLL nodes. Weights I1, I2 have length Nq.
#     Source index = loop index: (ii, jj) = (i, j). Example Nq=4 (· = source node, * = target):
#
#         source grid (GLL)          target * gets value from all 16 nodes
#         j 1   2   3   4
#       i +---+---+---+---+
#       1 | · | · | · | · |
#       2 | · | · | * | · |   →  ij = CartesianIndex((i, j)) for each ·
#       3 | · | · | · | · |
#       4 +---+---+---+---+
#
#   Bilinear (Nq1 = Nq2 = 2): only when the element has Nq=2 (two GLL points per dim).
#
#     Code: ii = is_bilinear ? (1 + (i-1)*(Nq-1)) : i  (and jj likewise).
#     Bilinear branch (Nq1=Nq2=2): i,j∈{1,2} → ii,jj∈{1,Nq} → four corners only.
#     Spectral branch (Nq1=Nq2=Nq): ii=i, jj=j → all nodes, or 4 via zeros in I1,I2.
#
#     Nq=2 only (2×2 = corners):
#
#         (1,1)·-------·(1,2)
#              |   *   |        * = target; (i,j)=(1,1),(2,1),(1,2),(2,2) → (ii,jj) = corners
#         (2,1)·-------·(2,2)
#
#     Nq>2: target between interior nodes → stencil = 4 nodes of containing sub-cell (spectral branch).
#
#         source grid (GLL, Nq=4)    2×2 sub-cell containing target *
#         j 1   2   3   4
#       i +---+---+---+---+
#       1 | · | · | · | · |
#       2 | · | · | · | · |
#       3 | · | · | * | · |   * = target; stencil = (2,2),(3,2),(2,3),(3,3) — the 4 · around *
#       4 +---+---+---+---+
#
#         (2,2)·-------·(2,3)
#              |   *   |        same idea as Nq=2, but sub-cell is interior
#         (3,2)·-------·(3,3)
#
function interpolate_slab_level_kernel!(
    output_array,
    field,
    vidx_ref_coordinates,
    h,
    (I1, I2)::Tuple{<:AbstractArray, <:AbstractArray},
)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    index <= length(vidx_ref_coordinates) || return nothing
    @inbounds begin
        space = axes(field)
        FT = Spaces.undertype(space)
        Nq1, Nq2 = length(I1), length(I2)
        quad = Spaces.quadrature_style(space)
        Nq = Quadratures.degrees_of_freedom(quad)
        is_bilinear = (Nq1 == 2 && Nq2 == 2)
        v_lo, v_hi, ξ3 = vidx_ref_coordinates[index]

        f_lo = zero(FT)
        f_hi = zero(FT)

        for j in 1:Nq2, i in 1:Nq1
            # Bilinear: map stencil index 1,2 → node index 1,Nq; else use index as-is
            ii = is_bilinear ? (1 + (i - 1) * (Nq - 1)) : i
            jj = is_bilinear ? (1 + (j - 1) * (Nq - 1)) : j
            ij = CartesianIndex((ii, jj))
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

    index <= length(vidx_ref_coordinates) || return nothing
    @inbounds begin
        space = axes(field)
        FT = Spaces.undertype(space)
        Nq1 = length(I1)
        quad = Spaces.quadrature_style(space)
        Nq = Quadratures.degrees_of_freedom(quad)
        # When Nq1==2 we have a 2-point (linear) bilinear stencil: map stencil index 1,2 → node 1,Nq
        is_bilinear = (Nq1 == 2)
        v_lo, v_hi, ξ3 = vidx_ref_coordinates[index]

        f_lo = zero(FT)
        f_hi = zero(FT)

        for i in 1:Nq1
            # Bilinear: map stencil index 1,2 → node index 1,Nq; else use index as-is
            ii = is_bilinear ? (1 + (i - 1) * (Nq - 1)) : i
            ij = CartesianIndex((ii,))
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
