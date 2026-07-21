import ClimaCore: Spaces, Quadratures, Topologies
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
    },
    mask = Spaces.get_mask(axes(out)),
)
    space = axes(out)
    bounds = Operators.window_bounds(space, bc)
    out_fv = Fields.field_values(out)
    us = DataLayouts.UniversalSize(out_fv)

    fspace = Spaces.face_space(space)
    n_face_levels = Spaces.nlevels(fspace)

    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(out_fv)
    # This uses block and grid indices instead of computing cartesian indices from a
    # linear index. The launch configuration is optimized for common use case of 64 face
    # levels and Ni = Nj = 4. Periodic toppologies and masks are not currently supported
    # `eager_copyto_stencil_kernel!` requires a  block size of (n_face_levels, Ni, 1)
    # this block config is better for VIJFH. It is only used when the total number of
    # threads in a block is between 32 and 256 to avoid underutilization of the GPU and
    # errors due to too many registers used when the block size is too large.
    if !Topologies.isperiodic(space) && mask isa NoMask &&
       32 <= n_face_levels * Ni <= 256
        op_matrix_bc = replace_fd_ops(bc)
        args = (
            strip_space(out, space),
            strip_space(op_matrix_bc, space),
            axes(out),
        )
        auto_launch!(
            eager_copyto_stencil_kernel!,
            args;
            threads_s = (n_face_levels, Ni, 1),
            blocks_s = (1, Nj, Nh),
            always_inline = true,
            shmem = n_face_levels * Ni * 9 * 4, # see `check_if_fits_in_shmem` for how this is calculated
        )
        call_post_op_callback() && post_op_callback(out, out, bc)
        return out
    end
    cart_inds = if mask isa NoMask
        cartesian_indices(us)
    else
        cartesian_indices_mask(us, mask)
    end

    args = cudaconvert((
        strip_space(out, space),
        strip_space(bc, space),
        axes(out),
        bounds,
        us,
        mask,
        cart_inds,
    ))

    threads = threads_via_occupancy(copyto_stencil_kernel!, args)
    n_max_threads = min(threads, get_N(us))
    p = if mask isa NoMask
        linear_partition(prod(size(out_fv)), n_max_threads)
    else
        masked_partition(mask, n_max_threads, us)
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
import ClimaCore.DataLayouts: get_N, get_Nv, get_Nij, get_Nij, get_Nh


function copyto_stencil_kernel!(
    out,
    bc::Union{
        StencilBroadcasted{CUDAColumnStencilStyle},
        Broadcasted{CUDAColumnStencilStyle},
    },
    space,
    bds,
    us,
    mask,
    cart_inds,
)
    @inbounds begin
        out_fv = Fields.field_values(out)
        tidx = linear_thread_idx()
        if linear_is_valid_index(tidx, us) && tidx ≤ length(unval(cart_inds))
            I = if mask isa NoMask
                unval(cart_inds)[tidx]
            else
                masked_universal_index(mask, cart_inds)
            end
            (li, lw, rw, ri) = bds
            (i, j, _, v, h) = I.I
            hidx = (i, j, h)
            idx = v - 1 + li
            val = Operators.getidx(space, bc, idx, hidx)
            setidx!(space, out, idx, hidx, val)
        end
    end
    return nothing
end
