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
    # `eager_copyto_stencil_kernel!` requires one x-thread per face level and computes
    # one column per block (its shared-memory buffers are static, so their size -- one
    # slot per face level -- must be known at compile time), so a block is
    # (n_face_levels, 1, 1) and the grid indexes the columns. Each block decomposes its
    # block index into the column's `(i, j, h)`; this layout suits VIJFH, where the
    # vertical axis is contiguous. Both periodic and non-periodic vertical topologies
    # are supported (`has_padding_thread` accounts for the extra face level of
    # non-periodic spaces), as are masked spaces. High-resolution columns (more face
    # levels than fit in a block) fall through to `copyto_stencil_kernel!` below.
    # TODO: auto reduce max reg usage when needed because of high res columns
    # The eager kernel's per-level indexing (`calc_level_val` for `Field`s, and
    # `has_padding_thread`) is written for the finite difference space families
    # in `Operators.AllFiniteDifferenceSpace` (extruded, single-column, and
    # multi-column); any other family with a vertical dimension must take the
    # lazy kernel below, since `calc_level_val`'s space gate would misread its
    # fields as level fields and evaluate them entirely at level 1. Keep this
    # gate and that space gate in sync.
    eager_supported = space isa Operators.AllFiniteDifferenceSpace
    if !high_resolution && eager_supported
        # Predict the kernel's static shared memory: the caching nodes' allocations
        # are merged into one region sized to the largest single entry (see
        # `max_eager_shmem_per_thread`). `nothing` means an expression's cached entry
        # type could not be sized (non-concrete inference), so the eager kernel cannot
        # be compiled and the lazy kernel below (which needs no shared memory) is used.
        eager_shmem_per_thread = max_eager_shmem_per_thread(bc)
        # mask.N holds the active column count in a one-element device array;
        # reading it on the host needs @allowscalar.
        n_columns =
            mask isa NoMask ? Ni * Nj * Nh :
            CUDA.@allowscalar(mask.N[1])
        block_dim_x = n_columns
        eager_shmem =
            isnothing(eager_shmem_per_thread) ? nothing :
            n_face_levels * eager_shmem_per_thread
        # use fallback lazy evaluation if the eager kernel's static shared memory
        # could exceed the device's per-block limit
        if !isnothing(eager_shmem) && eager_shmem ≤ max_shmem
            # `axes(out)` is passed as the space the kernel evaluates `bc` on, since
            # `out` and `bc` are space-stripped. The kernel recovers the output
            # layout's horizontal extents from the type parameters of
            # `field_values(out)` (see `vijh_params`), so the `CartesianIndices` it
            # builds from them divides by compile-time constants, keeping the
            # per-thread `divrem` cheap.
            args = (
                strip_space(out, space),
                strip_space(bc, space),
                mask,
                axes(out),
            )

            auto_launch!(
                eager_copyto_stencil_kernel!,
                args;
                threads_s = (n_face_levels, 1, 1),
                blocks_s = (block_dim_x, 1, 1),
                always_inline = true,
            )
            return out
        end
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
