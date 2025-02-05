import ClimaCore: Spaces, Quadratures, Topologies
import Base.Broadcast: Broadcasted
import ClimaComms
using CUDA: @cuda
import ClimaCore.Utilities: half
import ClimaCore.Operators: AbstractStencilStyle, strip_space
import ClimaCore.Operators: setidx!, getidx
import ClimaCore.Operators: StencilBroadcasted
import ClimaCore.Operators: LeftBoundaryWindow, RightBoundaryWindow, Interior

struct CUDAColumnStencilStyle <: AbstractStencilStyle end
AbstractStencilStyle(::ClimaComms.CUDADevice) = CUDAColumnStencilStyle

Base.@propagate_inbounds function getidx(
    space,
    sbc::StencilBroadcasted{CUDAColumnStencilStyle},
    ij,
    slabidx,
)
    operator_evaluate(sbc.op, sbc.work, sbc.axes, ij, slabidx)
end

function Base.copyto!(
    out::Field,
    bc::Union{
        StencilBroadcasted{CUDAColumnStencilStyle},
        Broadcasted{CUDAColumnStencilStyle},
    },
)
    space = axes(out)
    bounds = Operators.window_bounds(space, bc)
    out_fv = Fields.field_values(out)
    us = DataLayouts.UniversalSize(out_fv)

    p = fd_stencil_partition(us)
    args = (
        strip_space(out, space),
        strip_space(bc, space),
        axes(out),
        bounds,
        us,
        Val(p.Nvthreads),
    )

    if bc isa StencilBroadcasted && bc.op isa DivergenceF2C{@NamedTuple{}}
        auto_launch!(
            copyto_stencil_kernel_shmem!,
            args;
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    else
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

function copyto_stencil_kernel!(out, bc, space, bds, us, ::Val{Nvt}) where {Nvt}
    @inbounds begin
        out_fv = Fields.field_values(out)
        I = universal_index(out_fv)
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
    bc′,
    space,
    bds,
    us,
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
            isactive = if Operators.is_face_space(space) # check that idx is in bounds
                idx + half <= nv
            else
                idx <= nv
            end
            if isactive
                # Call getidx overloaded in operators_fd_shmem_common.jl
                if li <= idx <= (lw - 1)
                    lwindow =
                        LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
                    val = shmem_getidx(space, bc_shmem, lwindow, idx, hidx)
                elseif (rw + 1) <= idx <= ri
                    rwindow =
                        RightBoundaryWindow{Spaces.right_boundary_name(space)}()
                    val = shmem_getidx(space, bc_shmem, rwindow, idx, hidx)
                else
                    # @assert lw <= idx <= rw
                    iwindow = Interior()
                    val = shmem_getidx(space, bc_shmem, iwindow, idx, hidx)
                end
                setidx!(space, out, idx, hidx, val)
            end
        end
    end
    return nothing
end
