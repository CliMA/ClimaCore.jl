import ClimaCore: Spaces, Quadratures, Topologies
import Base.Broadcast: Broadcasted
import ClimaComms
using CUDA: @cuda
import ClimaCore.Operators: AbstractStencilStyle, strip_space
import ClimaCore.Operators: StencilBroadcasted

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
    (li, lw, rw, ri) = bounds = window_bounds(space, bc)
    Nv = ri - li + 1
    max_threads = 256
    nitems = Nv * Nq * Nq * Nh # # of independent items
    (nthreads, nblocks) = Topologies._configure_threadblock(max_threads, nitems)
    @cuda always_inline = true threads = (nthreads,) blocks = (nblocks,) copyto_stencil_kernel!(
        strip_space(out, space),
        strip_space(bc, space),
        axes(out),
        bounds,
        Nq,
        Nh,
        Nv,
    )
    return out
end

function copyto_stencil_kernel!(out, bc, space, bds, Nq, Nh, Nv)
    gid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if gid ≤ Nv * Nq * Nq * Nh
        (li, lw, rw, ri) = bds
        (v, i, j, h) = Topologies._get_idx((Nv, Nq, Nq, Nh), gid)
        hidx = (i, j, h)
        idx = v - 1 + li
        window =
            idx < lw ? LeftBoundaryWindow{Spaces.left_boundary_name(space)}() :
            (
                idx > rw ?
                RightBoundaryWindow{Spaces.right_boundary_name(space)}() :
                Interior()
            )
        setidx!(space, out, idx, hidx, getidx(space, bc, window, idx, hidx))
    end
    return nothing
end
