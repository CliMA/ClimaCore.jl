import ClimaCore: Spaces, Quadratures, Topologies
import Base.Broadcast: Broadcasted
import ClimaComms
using CUDA: @cuda
import ClimaCore.Operators: AbstractStencilStyle, strip_space
import ClimaCore.Operators: setidx!, getidx
import ClimaCore.Operators: StencilBroadcasted
import ClimaCore.Operators: LeftBoundaryWindow, RightBoundaryWindow, Interior

struct CUDAColumnStencilStyle <: AbstractStencilStyle end
AbstractStencilStyle(::ClimaComms.CUDADevice) = CUDAColumnStencilStyle

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
    args =
        (strip_space(out, space), strip_space(bc, space), axes(out), bounds, us)

    threads = threads_via_occupancy(copyto_stencil_kernel!, args)
    n_max_threads = min(threads, get_N(us))
    p = partition(out_fv, n_max_threads)

    auto_launch!(
        copyto_stencil_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
    return out
end
import ClimaCore.DataLayouts: get_N, get_Nv, get_Nij, get_Nij, get_Nh

function copyto_stencil_kernel!(out, bc, space, bds, us)
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
