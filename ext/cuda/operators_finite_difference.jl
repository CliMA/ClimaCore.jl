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
    if space isa Spaces.ExtrudedFiniteDifferenceSpace
        QS = Spaces.quadrature_style(space)
        Nq = Quadratures.degrees_of_freedom(QS)
        Nh = Topologies.nlocalelems(Spaces.topology(space))
    else
        Nq = 1
        Nh = 1
    end
    (li, lw, rw, ri) = bounds = Operators.window_bounds(space, bc)
    Nv = ri - li + 1
    max_threads = 256
    us = DataLayouts.UniversalSize(Fields.field_values(out))
    nitems = Nv * Nq * Nq * Nh # # of independent items
    (nthreads, nblocks) = _configure_threadblock(max_threads, nitems)
    args =
        (strip_space(out, space), strip_space(bc, space), axes(out), bounds, us)
    auto_launch!(
        copyto_stencil_kernel!,
        args;
        threads_s = (nthreads,),
        blocks_s = (nblocks,),
    )
    return out
end
import ClimaCore.DataLayouts: get_N, get_Nv, get_Nij, get_Nij, get_Nh

function copyto_stencil_kernel!(out, bc, space, bds, us)
    @inbounds begin
        gid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if gid ≤ get_N(us)
            (li, lw, rw, ri) = bds
            Nv = get_Nv(us)
            Nq = get_Nij(us)
            Nh = get_Nh(us)
            (v, i, j, h) = cart_ind((Nv, Nq, Nq, Nh), gid).I
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
