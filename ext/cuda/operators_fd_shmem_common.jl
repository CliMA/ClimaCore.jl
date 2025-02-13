import ClimaCore: DataLayouts, Spaces, Geometry, RecursiveApply, DataLayouts
import CUDA
import ClimaCore.Operators: DivergenceF2C
import ClimaCore.Operators: return_eltype, get_local_geometry
import ClimaCore.Operators: getidx

# We don't support all operators yet, so let's use a custom getidx
Base.@propagate_inbounds function shmem_getidx(
    # Base.@propagate_inbounds function getidx(
    space,
    bc::StencilBroadcasted{CUDAColumnStencilStyle},
    loc,
    idx,
    hidx,
)
    fd_operator_evaluate(bc.op, bc.work, loc, space, idx, hidx, bc.args...)
end

"""
    fd_allocate_shmem(Val(Nvt), b)

Create a new broadcasted object with necessary share memory allocated,
using `Nvt` nodal points per block.
"""
@inline function fd_allocate_shmem(::Val{Nvt}, obj) where {Nvt}
    obj
end
@inline function fd_allocate_shmem(
    ::Val{Nvt},
    bc::Broadcasted{Style},
) where {Nvt, Style}
    Broadcasted{Style}(bc.f, _fd_allocate_shmem(Val(Nvt), bc.args...), bc.axes)
end
@inline function fd_allocate_shmem(
    ::Val{Nvt},
    sbc::StencilBroadcasted{Style},
) where {Nvt, Style}
    args = _fd_allocate_shmem(Val(Nvt), sbc.args...)
    work = fd_operator_shmem(sbc.axes, Val(Nvt), sbc.op, args...)
    StencilBroadcasted{Style}(sbc.op, args, sbc.axes, work)
end

@inline _fd_allocate_shmem(::Val{Nvt}) where {Nvt} = ()
@inline _fd_allocate_shmem(::Val{Nvt}, arg, xargs...) where {Nvt} = (
    fd_allocate_shmem(Val(Nvt), arg),
    _fd_allocate_shmem(Val(Nvt), xargs...)...,
)

"""
    fd_resolve_shmem!(
        sbc::StencilBroadcasted,
        idx,
        hidx,
        bds
    )

Recursively stores the arguments to all operators into shared memory, at the
given indices (if they are valid).

As this calls `sync_threads()`, it should be called collectively on all threads
at the same time.
"""
Base.@propagate_inbounds function fd_resolve_shmem!(
    sbc::StencilBroadcasted,
    idx,
    hidx,
    bds,
)
    (li, lw, rw, ri) = bds
    space = axes(sbc)
    # Do we need an _extra_ + 1 since shmem is loaded at idx - half? need to generalize to boundary window?
    isactive = li <= idx <= ri + 1

    _fd_resolve_shmem!(idx, hidx, bds, sbc.args...)

    if isactive
        if li <= idx <= (lw - 1)
            lwindow =
                Operators.LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
            fd_operator_fill_shmem!(
                sbc.op,
                sbc.work,
                lwindow,
                space,
                idx,
                hidx,
                sbc.args...,
            )
        elseif (rw + 1 + 1) <= idx <= ri
            rwindow = RightBoundaryWindow{Spaces.right_boundary_name(space)}()
            fd_operator_fill_shmem!(
                sbc.op,
                sbc.work,
                rwindow,
                space,
                idx,
                hidx,
                sbc.args...,
            )
        else
            iwindow = Interior()
            fd_operator_fill_shmem!(
                sbc.op,
                sbc.work,
                iwindow,
                space,
                idx,
                hidx,
                sbc.args...,
            )
        end
    end
    return nothing
end

@inline _fd_resolve_shmem!(idx, hidx, bds) = nothing
@inline function _fd_resolve_shmem!(idx, hidx, bds, arg, xargs...)
    fd_resolve_shmem!(arg, idx, hidx, bds)
    _fd_resolve_shmem!(idx, hidx, bds, xargs...)
end


Base.@propagate_inbounds function fd_resolve_shmem!(
    bc::Broadcasted,
    idx,
    hidx,
    bds,
)
    _fd_resolve_shmem!(idx, hidx, bds, bc.args...)
    return nothing
end
Base.@propagate_inbounds function fd_resolve_shmem!(obj, idx, hidx, bds)
    nothing
end
