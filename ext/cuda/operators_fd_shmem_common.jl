import ClimaCore: DataLayouts, Spaces, Geometry, RecursiveApply, DataLayouts
import CUDA
import ClimaCore.Operators: return_eltype, get_local_geometry
import ClimaCore.Operators: getidx
import ClimaCore.Utilities: PlusHalf
import ClimaCore.Utilities

Base.@propagate_inbounds function getidx(
    parent_space,
    bc::StencilBroadcasted{CUDAColumnStencilStyle},
    loc::Interior,
    idx,
    hidx,
)
    space = axes(bc)
    if Operators.fd_shmem_is_supported(bc)
        return fd_operator_evaluate(
            bc.op,
            bc.work,
            loc,
            space,
            idx,
            hidx,
            bc.args...,
        )
    end
    Operators.stencil_interior(bc.op, loc, space, idx, hidx, bc.args...)
end


Base.@propagate_inbounds function getidx(
    parent_space,
    bc::StencilBroadcasted{CUDAColumnStencilStyle},
    loc::Operators.LeftBoundaryWindow,
    idx,
    hidx,
)
    space = axes(bc)
    if Operators.fd_shmem_is_supported(bc)
        return fd_operator_evaluate(
            bc.op,
            bc.work,
            loc,
            space,
            idx,
            hidx,
            bc.args...,
        )
    end
    op = bc.op
    if Operators.call_left_boundary(idx, space, bc, loc)
        Operators.stencil_left_boundary(
            op,
            Operators.get_boundary(op, loc),
            loc,
            space,
            idx,
            hidx,
            bc.args...,
        )
    else
        # fallback to interior stencil
        Operators.stencil_interior(op, loc, space, idx, hidx, bc.args...)
    end
end

Base.@propagate_inbounds function getidx(
    parent_space,
    bc::StencilBroadcasted{CUDAColumnStencilStyle},
    loc::Operators.RightBoundaryWindow,
    idx,
    hidx,
)
    space = axes(bc)
    if Operators.fd_shmem_is_supported(bc)
        return fd_operator_evaluate(
            bc.op,
            bc.work,
            loc,
            space,
            idx,
            hidx,
            bc.args...,
        )
    end
    op = bc.op
    if Operators.call_right_boundary(idx, space, bc, loc)
        Operators.stencil_right_boundary(
            op,
            Operators.get_boundary(op, loc),
            loc,
            space,
            idx,
            hidx,
            bc.args...,
        )
    else
        # fallback to interior stencil
        Operators.stencil_interior(op, loc, space, idx, hidx, bc.args...)
    end
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

######### MatrixFields
# MatrixField operators are not yet supported, and we must stop recursing because
# we can have something of the form
# MatrixFields.LazyOneArgFDOperatorMatrix{DivergenceF2C{@NamedTuple{}}}(DivergenceF2C{@NamedTuple{}}(NamedTuple()))
# which `fd_shmem_is_supported` will return `true` for.

@inline fd_allocate_shmem(_, bc::MatrixFields.LazyOperatorBroadcasted) = bc
@inline fd_allocate_shmem(_, bc::MatrixFields.FDOperatorMatrix) = bc
@inline fd_allocate_shmem(_, bc::MatrixFields.LazyOneArgFDOperatorMatrix) = bc
#########

@inline function fd_allocate_shmem(
    ::Val{Nvt},
    sbc::StencilBroadcasted{Style},
) where {Nvt, Style}
    args = _fd_allocate_shmem(Val(Nvt), sbc.args...)
    work = if Operators.fd_shmem_is_supported(sbc)
        fd_operator_shmem(sbc.axes, Val(Nvt), sbc.op, args...)
    else
        nothing
    end
    StencilBroadcasted{Style}(sbc.op, args, sbc.axes, work)
end

@inline _fd_allocate_shmem(::Val{Nvt}) where {Nvt} = ()
@inline _fd_allocate_shmem(::Val{Nvt}, arg, xargs...) where {Nvt} = (
    fd_allocate_shmem(Val(Nvt), arg),
    _fd_allocate_shmem(Val(Nvt), xargs...)...,
)

get_arg_space(bc::StencilBroadcasted{CUDAColumnStencilStyle}, args::Tuple{}) =
    axes(bc)
get_arg_space(bc::StencilBroadcasted{CUDAColumnStencilStyle}, args::Tuple) =
    axes(args[1])

get_cent_idx(idx::Integer) = idx
get_face_idx(idx::PlusHalf) = idx
get_cent_idx(idx::PlusHalf) = idx + half
get_face_idx(idx::Integer) = idx - half

"""
    fd_resolve_shmem!(
        sbc::StencilBroadcasted,
        idx,
        hidx,
        bds
    )

Recursively stores the arguments to all operators into shared memory, at the
given indices (if they are valid).
"""
Base.@propagate_inbounds function fd_resolve_shmem!(
    sbc::StencilBroadcasted{CUDAColumnStencilStyle},
    idx, # top-level index
    hidx,
    bds,
)
    (li, lw, rw, ri) = bds
    space = axes(sbc)

    ᶜspace = Spaces.center_space(space)
    ᶠspace = Spaces.face_space(space)
    arg_space = get_arg_space(sbc, sbc.args)
    ᶜidx = get_cent_idx(idx)
    ᶠidx = get_face_idx(idx)

    _fd_resolve_shmem!(idx, hidx, bds, sbc.args...) # propagate idx, not bc_idx recursively through broadcast expressions

    # After recursion, check if shmem is supported for this operator
    Operators.fd_shmem_is_supported(sbc) || return nothing

    (; op) = sbc
    lloc = Operators.LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    rloc = Operators.RightBoundaryWindow{Spaces.right_boundary_name(space)}()
    iloc = Operators.Interior()

    IP = Topologies.isperiodic(Spaces.vertical_topology(space))

    # There are `Nf` threads, where `Nf` is the number of face levels. So,
    # each thread is responsible for filling shared memory at its cell center
    # (if the broadcasted argument lives on cell centers)
    # or cell face (if the broadcasted argument lives on cell faces) index.
    # We use `get_face_idx` and `get_cent_idx` to grab the nearest in-bounds
    # index, and `get_arg_space` to get the space of the first broadcasted argument
    # (the space of all broadcasted arguments must all match, so using the first is valid).

    bc_bds = Operators.window_bounds(space, sbc)
    (bc_li, bc_lw, bc_rw, bc_ri) = bc_bds
    if arg_space isa Operators.AllFaceFiniteDifferenceSpace # populate shmem on faces
        if IP || get_face_idx(bc_lw) ≤ ᶠidx ≤ get_face_idx(bc_rw) + 1 # interior
            fd_operator_fill_shmem_interior!(
                sbc.op,
                sbc.work,
                iloc,
                space,
                ᶠidx,
                hidx,
                sbc.args...,
            )
        elseif ᶠidx < get_face_idx(bc_lw) && Operators.has_boundary(op, lloc) # left
            fd_operator_fill_shmem_left_boundary!(
                sbc.op,
                Operators.get_boundary(op, lloc),
                sbc.work,
                lloc,
                space,
                ᶠidx,
                hidx,
                sbc.args...,
            )
        elseif ᶠidx > get_face_idx(bc_rw) && Operators.has_boundary(op, rloc) # right
            fd_operator_fill_shmem_right_boundary!(
                sbc.op,
                Operators.get_boundary(op, rloc),
                sbc.work,
                rloc,
                space,
                ᶠidx,
                hidx,
                sbc.args...,
            )
        elseif ᶠidx < get_face_idx(bc_lw) && !Operators.has_boundary(op, lloc) # left
            fd_operator_fill_shmem_interior!(
                sbc.op,
                sbc.work,
                lloc,
                space,
                ᶠidx,
                hidx,
                sbc.args...,
            )
        elseif ᶠidx > get_face_idx(bc_rw) && !Operators.has_boundary(op, rloc) # right
            fd_operator_fill_shmem_interior!(
                sbc.op,
                sbc.work,
                rloc,
                space,
                ᶠidx,
                hidx,
                sbc.args...,
            )
        else # this else should never run
        end
    else  # populate shmem on centers
        if IP || get_cent_idx(bc_lw) ≤ ᶜidx ≤ get_cent_idx(bc_rw) + 1 # interior
            fd_operator_fill_shmem_interior!(
                sbc.op,
                sbc.work,
                iloc,
                space,
                ᶜidx,
                hidx,
                sbc.args...,
            )
        elseif ᶜidx < get_cent_idx(bc_lw) && Operators.has_boundary(op, lloc) # left
            fd_operator_fill_shmem_left_boundary!(
                sbc.op,
                Operators.get_boundary(op, lloc),
                sbc.work,
                lloc,
                space,
                ᶜidx,
                hidx,
                sbc.args...,
            )
        elseif ᶜidx > get_cent_idx(bc_rw) && Operators.has_boundary(op, rloc) # right
            fd_operator_fill_shmem_right_boundary!(
                sbc.op,
                Operators.get_boundary(op, rloc),
                sbc.work,
                rloc,
                space,
                ᶜidx,
                hidx,
                sbc.args...,
            )
        elseif ᶜidx < get_cent_idx(bc_lw) && !Operators.has_boundary(op, lloc) # left
            fd_operator_fill_shmem_interior!(
                sbc.op,
                Operators.get_boundary(op, lloc),
                sbc.work,
                lloc,
                space,
                ᶜidx,
                hidx,
                sbc.args...,
            )
        elseif ᶜidx > get_cent_idx(bc_rw) && !Operators.has_boundary(op, rloc) # right
            fd_operator_fill_shmem_interior!(
                sbc.op,
                Operators.get_boundary(op, rloc),
                sbc.work,
                rloc,
                space,
                ᶜidx,
                hidx,
                sbc.args...,
            )
        else # this else should never run
        end
    end
    return nothing
end

Base.@propagate_inbounds _fd_resolve_shmem!(idx, hidx, bds) = nothing
Base.@propagate_inbounds function _fd_resolve_shmem!(
    idx,
    hidx,
    bds,
    arg,
    xargs...,
)
    fd_resolve_shmem!(arg, idx, hidx, bds)
    _fd_resolve_shmem!(idx, hidx, bds, xargs...)
end

Base.@propagate_inbounds fd_resolve_shmem!(
    bc::Broadcasted{CUDAColumnStencilStyle},
    idx,
    hidx,
    bds,
) = _fd_resolve_shmem!(idx, hidx, bds, bc.args...)
@inline fd_resolve_shmem!(obj, idx, hidx, bds) = nothing

if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(fd_resolve_shmem!)
        m.recursion_relation = dont_limit
    end
    for m in methods(_fd_resolve_shmem!)
        m.recursion_relation = dont_limit
    end
    for m in methods(_fd_allocate_shmem)
        m.recursion_relation = dont_limit
    end
    for m in methods(fd_allocate_shmem)
        m.recursion_relation = dont_limit
    end
end
