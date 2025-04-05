import ClimaCore: DataLayouts, Spaces, Geometry, RecursiveApply, DataLayouts
import CUDA
import ClimaCore.Operators: return_eltype, get_local_geometry
import ClimaCore.Operators: getidx
import ClimaCore.Utilities: PlusHalf
import ClimaCore.Utilities

#####
##### Boundary helpers
#####

@inline function has_left_boundary(space, op)
    lloc = Operators.LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    return Operators.has_boundary(op, lloc)
end
@inline function has_right_boundary(space, op)
    rloc = Operators.RightBoundaryWindow{Spaces.right_boundary_name(space)}()
    return Operators.has_boundary(op, rloc)
end

@inline on_boundary(space, op, loc, idx) =
    Operators.has_boundary(op, loc) && on_boundary(idx, space)

@inline on_left_boundary(idx, space, op) =
    has_left_boundary(space, op) && on_left_boundary(idx, space)
@inline on_right_boundary(idx, space, op) =
    has_right_boundary(space, op) && on_right_boundary(idx, space)

@inline on_boundary(idx::PlusHalf, space) =
    idx == Operators.left_face_boundary_idx(space) ||
    idx == Operators.right_face_boundary_idx(space)
@inline on_boundary(idx::Integer, space) =
    idx == Operators.left_center_boundary_idx(space) ||
    idx == Operators.right_center_boundary_idx(space)

@inline on_left_boundary(idx::PlusHalf, space) =
    idx == Operators.left_face_boundary_idx(space)
@inline on_left_boundary(idx::Integer, space) =
    idx == Operators.left_center_boundary_idx(space)

@inline on_right_boundary(idx::PlusHalf, space) =
    idx == Operators.right_face_boundary_idx(space)
@inline on_right_boundary(idx::Integer, space) =
    idx == Operators.right_center_boundary_idx(space)

@inline on_any_boundary(idx, space, op) =
    (has_left_boundary(space, op) && on_left_boundary(idx, space)) ||
    has_right_boundary(space, op) && on_right_boundary(idx, space)

#####
##### range window helpers (faces)
#####

@inline function in_interior_range(idx::PlusHalf, bc_bds)
    (bc_li, bc_lw, bc_rw, bc_ri) = bc_bds
    get_face_idx(bc_lw) ≤ idx ≤ get_face_idx(bc_rw) + 1
end
@inline function in_left_boundary_window_range(idx::PlusHalf, bc_bds)
    (bc_li, bc_lw, bc_rw, bc_ri) = bc_bds
    return get_face_idx(bc_li) - 1 ≤ idx ≤ get_face_idx(bc_lw + half)
end
@inline function in_right_boundary_window_range(idx::PlusHalf, bc_bds)
    (bc_li, bc_lw, bc_rw, bc_ri) = bc_bds
    return get_face_idx(bc_rw) ≤ idx ≤ get_face_idx(bc_ri) + 1
end

#####
##### range window helpers (centers)
#####

@inline function in_interior_range(idx::Integer, bc_bds)
    (bc_li, bc_lw, bc_rw, bc_ri) = bc_bds
    get_cent_idx(bc_lw) ≤ idx ≤ get_cent_idx(bc_rw)
end
@inline function in_left_boundary_window_range(idx::Integer, bc_bds)
    (bc_li, bc_lw, bc_rw, bc_ri) = bc_bds
    return get_cent_idx(bc_li) ≤ idx < get_cent_idx(bc_lw)
end
@inline function in_right_boundary_window_range(idx::Integer, bc_bds)
    (bc_li, bc_lw, bc_rw, bc_ri) = bc_bds
    return get_cent_idx(bc_rw) < idx ≤ get_cent_idx(bc_ri)
end

#####
##### window helpers (faces)
#####

@inline function in_left_boundary_window(
    idx::PlusHalf,
    space,
    bc_bds,
    op,
    args...,
)
    lloc = Operators.LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    Operators.should_call_left_boundary(idx, space, lloc, op, args...) ||
        in_left_boundary_window_range(idx, bc_bds)
end

@inline function in_right_boundary_window(
    idx::PlusHalf,
    space,
    bc_bds,
    op,
    args...,
)
    rloc = Operators.RightBoundaryWindow{Spaces.right_boundary_name(space)}()
    Operators.should_call_right_boundary(idx, space, rloc, op, args...) ||
        in_right_boundary_window_range(idx, bc_bds)
end

@inline function in_interior(idx::PlusHalf, space, bc_bds, op, args...)
    # TODO: simplify this function / logic / arithmetic
    (bc_li, bc_lw, bc_rw, bc_ri) = bc_bds
    IP = Topologies.isperiodic(Spaces.vertical_topology(space))
    crb = in_right_boundary_window_range(idx, bc_bds)
    clb = in_left_boundary_window_range(idx, bc_bds)
    return IP || in_interior_range(idx, bc_bds) && !(crb || clb)
end

#####
##### window helpers (centers)
#####

@inline function in_interior(idx::Integer, space, bc_bds, op, args...)
    # TODO: simplify this function / logic / arithmetic
    # TODO: use the (commented) range methods instead, as it would be much simpler.
    (bc_li, bc_lw, bc_rw, bc_ri) = bc_bds
    IP = Topologies.isperiodic(Spaces.vertical_topology(space))
    # crb = in_right_boundary_window_range(idx, bc_bds)
    crb = in_right_boundary_window(idx, space, bc_bds, op, args...)
    # clb = in_left_boundary_window_range(idx, bc_bds)
    clb = in_left_boundary_window(idx, space, bc_bds, op, args...)
    return IP || in_interior_range(idx, bc_bds) && !(crb || clb)
end

@inline function in_domain(idx::Integer, space)
    ᶜspace = Spaces.center_space(space)
    return 1 ≤ idx ≤ Spaces.nlevels(ᶜspace)
end

@inline function in_left_boundary_window(
    idx::Integer,
    space,
    bc_bds,
    op,
    args...,
)
    lloc = Operators.LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    Operators.should_call_left_boundary(idx, space, lloc, op, args...) ||
        in_left_boundary_window_range(idx, bc_bds)
end

@inline function in_right_boundary_window(
    idx::Integer,
    space,
    bc_bds,
    op,
    args...,
)
    rloc = Operators.RightBoundaryWindow{Spaces.right_boundary_name(space)}()
    ᶜspace = Spaces.center_space(space)
    idx > Spaces.nlevels(ᶜspace) && return false # short-circuit if
    Operators.should_call_right_boundary(idx, space, rloc, op, args...) ||
        in_right_boundary_window_range(idx, bc_bds)
end

#####
#####
#####

Base.@propagate_inbounds function getidx(
    parent_space,
    bc::StencilBroadcasted{CUDAWithShmemColumnStencilStyle},
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
    bc::StencilBroadcasted{CUDAWithShmemColumnStencilStyle},
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
    if Operators.should_call_left_boundary(idx, space, loc, bc.op, bc.args...)
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
    bc::StencilBroadcasted{CUDAWithShmemColumnStencilStyle},
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
    if Operators.should_call_right_boundary(idx, space, loc, bc.op, bc.args...)
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

"""
    fd_shmem_needed_per_column(::Base.Broadcast.Broadcasted)
    fd_shmem_needed_per_column(::StencilBroadcasted)

Return the total number of shared memory (in bytes) for the given
broadcast expression.
"""
@inline fd_shmem_needed_per_column(bc) = fd_shmem_needed_per_column(0, bc)
@inline fd_shmem_needed_per_column(shmem_bytes, obj) = shmem_bytes
@inline fd_shmem_needed_per_column(
    shmem_bytes,
    bc::Broadcasted{Style},
) where {Style} =
    shmem_bytes + _fd_shmem_needed_per_column(shmem_bytes, bc.args)

@inline function fd_shmem_needed_per_column(
    shmem_bytes,
    sbc::StencilBroadcasted{Style},
) where {Style}
    shmem_bytes₀ = _fd_shmem_needed_per_column(shmem_bytes, sbc.args)
    return if Operators.fd_shmem_is_supported(sbc)
        sizeof(return_eltype(sbc.op, sbc.args...)) + shmem_bytes₀
    else
        shmem_bytes₀
    end
end

@inline _fd_shmem_needed_per_column(shmem_bytes::Integer, ::Tuple{}) =
    shmem_bytes
@inline _fd_shmem_needed_per_column(shmem_bytes::Integer, args::Tuple{Any}) =
    shmem_bytes + fd_shmem_needed_per_column(shmem_bytes::Integer, args[1])
@inline _fd_shmem_needed_per_column(shmem_bytes::Integer, args::Tuple) =
    shmem_bytes +
    fd_shmem_needed_per_column(shmem_bytes::Integer, args[1]) +
    _fd_shmem_needed_per_column(shmem_bytes::Integer, Base.tail(args))


get_arg_space(bc::StencilBroadcasted, args::Tuple{}) = axes(bc)
get_arg_space(bc::StencilBroadcasted, args::Tuple) = axes(args[1])

get_cent_idx(idx::Integer) = idx # center when traversing centers (trivial)
get_face_idx(idx::PlusHalf) = idx # face when traversing faces (trivial)

get_cent_idx(idx::PlusHalf) = idx + half # center when traversing faces. Convention: use center right of face
get_face_idx(idx::Integer) = idx - half # face when traversing centers. Convention: use face left of center

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
    sbc::StencilBroadcasted{CUDAWithShmemColumnStencilStyle},
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
    ᵃidx = arg_space isa Operators.AllFaceFiniteDifferenceSpace ? ᶠidx : ᶜidx

    if in_interior(ᵃidx, arg_space, bc_bds, sbc.op, sbc.args...)
        fd_operator_fill_shmem!(
            sbc.op,
            sbc.work,
            iloc,
            bc_bds,
            arg_space,
            space,
            ᵃidx,
            hidx,
            sbc.args...,
        )
    elseif in_left_boundary_window(ᵃidx, arg_space, bc_bds, sbc.op, sbc.args...)
        fd_operator_fill_shmem!(
            sbc.op,
            sbc.work,
            lloc,
            bc_bds,
            arg_space,
            space,
            ᵃidx,
            hidx,
            sbc.args...,
        )
    elseif in_right_boundary_window(
        ᵃidx,
        arg_space,
        bc_bds,
        sbc.op,
        sbc.args...,
    )
        fd_operator_fill_shmem!(
            sbc.op,
            sbc.work,
            rloc,
            bc_bds,
            arg_space,
            space,
            ᵃidx,
            hidx,
            sbc.args...,
        )
    end
    CUDA.sync_threads()
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
    bc::Broadcasted{CUDAWithShmemColumnStencilStyle},
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
