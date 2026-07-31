import ClimaCore: Fields, Spaces, Quadratures, Topologies
import Base.Broadcast: Broadcasted
import ClimaComms
using CUDA: @cuda, i32
import ClimaCore.Utilities: half
import ClimaCore.Operators
import ClimaCore.Operators: AbstractStencilStyle, strip_space
import ClimaCore.Operators: setidx!, getidx
import ClimaCore.Operators: StencilBroadcasted
import ClimaCore.Operators: LeftBoundaryWindow, RightBoundaryWindow, Interior

struct CUDAColumnStencilStyle <: AbstractStencilStyle end

AbstractStencilStyle(bc, ::ClimaComms.CUDADevice) = CUDAColumnStencilStyle

Base.Broadcast.BroadcastStyle(
    x::Operators.ColumnStencilStyle,
    y::CUDAColumnStencilStyle,
) = y
include("operators_fd_eager.jl")

function Base.copyto!(
    out::Field,
    bc::Union{
        StencilBroadcasted{CUDAColumnStencilStyle},
        Broadcasted{CUDAColumnStencilStyle},
    };
    mask = Spaces.get_mask(axes(out)),
)
    space = axes(out)
    bounds = Operators.window_bounds(space, bc)
    out_fv = Fields.field_values(out)

    fspace = Spaces.face_space(space)
    n_face_levels = Spaces.nlevels(fspace)
    high_resolution = !(n_face_levels ≤ 256)
    # https://github.com/JuliaGPU/CUDA.jl/issues/2672
    max_shmem = CUDA.attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    )

    (_, Ni, Nj, Nh) = size(out_fv)
    # This uses block and grid indices instead of computing cartesian indices from a
    # linear index. The launch configuration is optimized for common use case of 64 face
    # levels and Ni = Nj = 4. Both periodic and non-periodic vertical topologies are
    # supported (`has_padding_thread` accounts for the extra face level of non-periodic
    # spaces); high-resolution columns fall through to `copyto_stencil_kernel!` below.
    # `eager_copyto_stencil_kernel!` requires a  block size of (n_face_levels, Ni, 1)
    # this block config is better for VIJFH. It is only used when the total number of
    # threads in a block is between 32 and 256 to avoid underutilization of the GPU and
    # errors due to too many registers used when the block size is too large.
    # TODO: auto reduce max reg usage when needed because of high res columns
    # Size the dynamic shared memory to fit the largest single expression result in
    # the broadcasted tree (see `max_eager_shmem_per_thread`). If even that does not
    # fit in the device's per-block shared memory, there is no way to eagerly evaluate
    # the expression, so error out instead of silently falling back.
    eager_shmem_per_thread = max_eager_shmem_per_thread(bc)
    if !high_resolution
        #    32 <= n_face_levels * Ni <= 256
        # mask.N holds the active column count in a one-element device array;
        # reading it on the host needs @allowscalar.
        n_columns =
            mask isa NoMask ? Ni * Nj * Nh :
            CUDA.@allowscalar(mask.N[1])
        # 108 is the number of SMs in an A100. TODO: get this value from CUDA.jl to better optimize for different GPUs
        threads_dim_y = n_columns > 256 * 108 ? div(256, n_face_levels) : 1
        block_dim_x = div(n_columns, threads_dim_y, RoundUp)
        eager_shmem = n_face_levels * threads_dim_y * eager_shmem_per_thread
        eager_shmem ≤ max_shmem || error(
            "The intermediate results of this broadcasted expression are too \
             large to fit in GPU shared memory: evaluating it eagerly needs \
             $(eager_shmem_per_thread) bytes per thread ($(eager_shmem) bytes \
             per block of $(n_face_levels * threads_dim_y) threads), but the \
             device only provides $(max_shmem) bytes of shared memory per block. \
             Split the expression into smaller sub-expressions so that each \
             intermediate matrix/vector result is smaller.",
        )
        # `axes(out)` is passed so the kernel can recover the output layout's
        # horizontal extents from its type parameters. The `CartesianIndices` the
        # kernel builds from them therefore divides by compile-time constants,
        # keeping the per-thread `divrem` cheap.
        args = (
            strip_space(out, space),
            strip_space(bc, space),
            mask,
            axes(out),
        )

        auto_launch!(
            eager_copyto_stencil_kernel!,
            args;
            threads_s = (n_face_levels, threads_dim_y, 1),
            blocks_s = (block_dim_x, 1, 1),
            always_inline = true,
            shmem = eager_shmem,
        )
        return out
    end
    cart_inds = if mask isa NoMask
        cartesian_indices(out_fv)
    else
        cartesian_indices_mask(out_fv, mask)
    end

    args = cudaconvert((
        strip_space(out, space),
        strip_space(bc, space),
        axes(out),
        bounds,
        mask,
        cart_inds,
    ))

    threads = threads_via_occupancy(copyto_stencil_kernel!, args)
    n_max_threads = min(threads, length(out_fv))
    p = if mask isa NoMask
        linear_partition(prod(size(out_fv)), n_max_threads)
    else
        masked_partition(mask, n_max_threads, out_fv)
    end
    auto_launch!(
        copyto_stencil_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
    call_post_op_callback() && post_op_callback(out, out, bc)
    return out
end

function copyto_stencil_kernel!(
    out,
    bc::Union{
        StencilBroadcasted{CUDAColumnStencilStyle},
        Broadcasted{CUDAColumnStencilStyle},
    },
    space,
    bds,
    mask,
    cart_inds,
)
    @inbounds begin
        out_fv = Fields.field_values(out)
        tidx = linear_thread_idx()
        if linear_is_valid_index(tidx, out_fv) && tidx ≤ length(unval(cart_inds))
            I = if mask isa NoMask
                unval(cart_inds)[tidx]
            else
                masked_universal_index(mask, cart_inds)
            end
            (li, lw, rw, ri) = bds
            (v, i, j, h) = I.I
            hidx = (i, j, h)
            idx = v - 1 + li
            val = Operators.getidx(space, bc, idx, hidx)
            setidx!(space, out, idx, hidx, val)
        end
    end
    return nothing
end
