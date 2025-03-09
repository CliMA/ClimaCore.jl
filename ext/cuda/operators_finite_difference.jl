import ClimaCore: Spaces, Quadratures, Topologies
import Base.Broadcast: Broadcasted
import ClimaComms
using CUDA: @cuda
import ClimaCore.Utilities: half
import ClimaCore.Operators
import ClimaCore.Operators: AbstractStencilStyle, strip_space
import ClimaCore.Operators: setidx!, getidx
import ClimaCore.Operators: StencilBroadcasted
import ClimaCore.Operators: LeftBoundaryWindow, RightBoundaryWindow, Interior

struct CUDAColumnStencilStyle <: AbstractStencilStyle end
struct CUDAWithShmemColumnStencilStyle <: AbstractStencilStyle end
AbstractStencilStyle(bc, ::ClimaComms.CUDADevice) =
    Operators.any_fd_shmem_supported(bc) ? CUDAWithShmemColumnStencilStyle :
    CUDAColumnStencilStyle

include("operators_fd_shmem_is_supported.jl")

function Base.copyto!(
    out::Field,
    bc::Union{
        StencilBroadcasted{CUDAColumnStencilStyle},
        StencilBroadcasted{CUDAWithShmemColumnStencilStyle},
        Broadcasted{CUDAColumnStencilStyle},
        Broadcasted{CUDAWithShmemColumnStencilStyle},
    },
    mask = Spaces.get_mask(axes(out)),
)
    space = axes(out)
    bounds = Operators.window_bounds(space, bc)
    out_fv = Fields.field_values(out)
    us = DataLayouts.UniversalSize(out_fv)

    fspace = Spaces.face_space(space)
    n_face_levels = Spaces.nlevels(fspace)
    high_resolution = !(n_face_levels ≤ 256)
    # https://github.com/JuliaGPU/CUDA.jl/issues/2672
    # max_shmem = 166912 # CUDA.limit(CUDA.LIMIT_SHMEM_SIZE) #
    max_shmem = CUDA.attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    )
    total_shmem = fd_shmem_needed_per_column(bc)
    enough_shmem = total_shmem ≤ max_shmem

    # TODO: Use CUDA.limit(CUDA.LIMIT_SHMEM_SIZE) to determine how much shmem should be used
    # TODO: add shmem support for masked operations
    if Operators.any_fd_shmem_supported(bc) &&
       !high_resolution &&
       mask isa NoMask &&
       enough_shmem
        p = fd_stencil_partition(us, n_face_levels)
        args = (
            strip_space(out, space),
            strip_space(bc, space),
            axes(out),
            bounds,
            us,
            mask,
            Val(p.Nvthreads),
        )
        auto_launch!(
            copyto_stencil_kernel_shmem!,
            args;
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    else
        bc′ = disable_shmem_style(bc)
        @assert !any_fd_shmem_style(bc′)
        args = (
            strip_space(out, space),
            strip_space(bc′, space),
            axes(out),
            bounds,
            us,
            mask,
        )

        threads = threads_via_occupancy(copyto_stencil_kernel!, args)
        n_max_threads = min(threads, get_N(us))
        p = if mask isa NoMask
            partition(out_fv, n_max_threads)
        else
            masked_partition(us, n_max_threads, mask)
        end

        auto_launch!(
            copyto_stencil_kernel!,
            args;
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    end
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
)
    @inbounds begin
        out_fv = Fields.field_values(out)
        I = if mask isa NoMask
            universal_index(out_fv)
        else
            masked_universal_index(mask)
        end
        if is_valid_index(out_fv, I, us)
            (li, lw, rw, ri) = bds
            (i, j, _, v, h) = I.I
            hidx = (i, j, h)
            idx = v - 1 + li
            if idx < lw
                lwindow = LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
                val = Operators.getidx(space, bc, lwindow, idx, hidx)
            elseif idx > rw
                rwindow =
                    RightBoundaryWindow{Spaces.right_boundary_name(space)}()
                val = Operators.getidx(space, bc, rwindow, idx, hidx)
            else
                iwindow = Interior()
                val = Operators.getidx(space, bc, iwindow, idx, hidx)
            end
            setidx!(space, out, idx, hidx, val)
        end
    end
    return nothing
end

function copyto_stencil_kernel_shmem!(
    out,
    bc′::Union{
        StencilBroadcasted{CUDAWithShmemColumnStencilStyle},
        Broadcasted{CUDAWithShmemColumnStencilStyle},
    },
    space,
    bds,
    us,
    mask,
    ::Val{Nvt},
) where {Nvt}
    @inbounds begin
        out_fv = Fields.field_values(out)
        us = DataLayouts.UniversalSize(out_fv)
        I = fd_stencil_universal_index(space, us)
        if fd_stencil_is_valid_index(I, us) # check that hidx is in bounds
            (li, lw, rw, ri) = bds
            (i, j, _, v, h) = I.I
            hidx = (i, j, h)
            idx = v - 1 + li
            bc = Operators.reconstruct_placeholder_broadcasted(space, bc′)
            bc_shmem = fd_allocate_shmem(Val(Nvt), bc) # allocates shmem

            fd_resolve_shmem!(bc_shmem, idx, hidx, bds) # recursively fills shmem
            CUDA.sync_threads()

            nv = Spaces.nlevels(space)
            isactive = if space isa Operators.AllFaceFiniteDifferenceSpace # check that idx is in bounds
                idx + half <= nv
            else
                idx <= nv
            end
            if isactive
                # Call getidx overloaded in operators_fd_shmem_common.jl
                if li <= idx <= (lw - 1)
                    lwindow =
                        LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
                    val = Operators.getidx(space, bc_shmem, lwindow, idx, hidx)
                elseif (rw + 1) <= idx <= ri
                    rwindow =
                        RightBoundaryWindow{Spaces.right_boundary_name(space)}()
                    val = Operators.getidx(space, bc_shmem, rwindow, idx, hidx)
                else
                    # @assert lw <= idx <= rw
                    iwindow = Interior()
                    val = Operators.getidx(space, bc_shmem, iwindow, idx, hidx)
                end
                setidx!(space, out, idx, hidx, val)
            end
        end
    end
    return nothing
end
