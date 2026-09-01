"""
    AbstractNumericalFlux

Abstract type for numerical flux functions used in DG methods.
"""
abstract type AbstractNumericalFlux end

@inline face_node_index_1d(face, Nq) = face == 1 ? 1 : Nq

# Surface Jacobian weight and outward unit normal for a 1D spectral-element
# endpoint. For extruded spaces, `local_geometry` should be the product
# geometry at that horizontal node and vertical level.
function compute_surface_geometry_1d(local_geometry, face)
    (; J, ∂ξ∂x) = local_geometry
    nvec = face == 1 ? (-J * ∂ξ∂x[1, :]) : (J * ∂ξ∂x[1, :])
    sWJ = LinearAlgebra.norm(nvec)
    n = nvec / sWJ
    # Project onto the horizontal orthonormal axis used by plane (x–z) states.
    n = Geometry.project(Geometry.UWAxis(), n)
    return Geometry.SurfaceGeometry(sWJ, Geometry.UVector(n.u))
end

# Surface Jacobian weight and outward unit horizontal normal for a face node
# (i, j) of a 2D spectral element within an extruded space. `local_geometry`
# is the product geometry at that horizontal node and vertical level, so `J`
# carries the vertical measure and `sWJ` is consistent with the 3D `WJ` of the
# mass-weighted volume residual. The normal is returned in the local
# orthonormal horizontal frame (`UVVector`): at a shared face node this frame
# is identical from both sides — including across cubed-sphere panel
# boundaries, where covariant components are discontinuous.
function compute_surface_geometry_extruded_2d(
    local_geometry,
    quad_weights,
    face,
    i,
    j,
)
    (; J, ∂ξ∂x) = local_geometry
    nvec = if face == 4
        -J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 2
        J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 1
        -J * ∂ξ∂x[2, :] * quad_weights[i]
    elseif face == 3
        J * ∂ξ∂x[2, :] * quad_weights[i]
    else
        error("invalid face index $face")
    end
    sWJ = LinearAlgebra.norm(nvec)
    n = Geometry.project(Geometry.UVAxis(), nvec / sWJ)
    return Geometry.SurfaceGeometry(sWJ, n)
end

"""
    DGGhostExchange

Handle for one round of the DG ghost-face halo exchange, shared across face
operators — see [`start_dg_ghost_exchange`](@ref). The first operator that
consumes the handle completes the exchange (`ClimaComms.finish`); later
consumers read the same recv strips.
"""
struct DGGhostExchange{B}
    bufs::B
    finished::Base.RefValue{Bool}
end
DGGhostExchange(bufs) = DGGhostExchange(bufs, Ref(false))
const NO_DG_GHOST_EXCHANGE = DGGhostExchange(nothing, Ref(true))

# Device-dispatch seam (DSS-style): CPU methods live here; the
# `ClimaComms.CUDADevice` methods are provided by the ClimaCoreCUDAExt
# extension (ext/cuda/operators_dg.jl).
#
# Throughout the CPU face-operator call chain, `args` is declared as
# `Vararg{Any, N}` with `N` a type parameter: the arguments are mostly just
# forwarded, so Julia's Vararg heuristic would otherwise compile the chain
# unspecialized on them, heap-allocating the argument tuple at every call
# boundary and the index closures at every face node.
"""
    add_numerical_flux_interior!(fn, dydt, args...)
    add_numerical_flux_interior!(ghost_exchange, fn, dydt, args...)

Add the numerical flux at the interior faces of the spectral space mesh.

The numerical flux is determined by evaluating

    fn(normal, argvals⁻, argvals⁺)

where:

  - `normal` is the unit normal vector, pointing from the "minus" side to the "plus" side
  - `argvals⁻` is the tuple of values of `args` on the "minus" side of the face
  - `argvals⁺` is the tuple of values of `args` on the "plus" side of the face
    and should return the net flux from the "minus" side to the "plus" side.

For consistency, it should satisfy the property that

    fn(normal, argvals⁻, argvals⁺) == -fn(-normal, argvals⁺, argvals⁻)

The method with a leading `ghost_exchange` consumes a shared halo exchange
from [`start_dg_ghost_exchange`](@ref) on distributed spaces.

See also:

  - [`CentralNumericalFlux`](@ref)
  - [`RusanovNumericalFlux`](@ref)
"""
add_numerical_flux_interior!(fn::F, dydt, args::Vararg{Any, N}) where {F, N} =
    _add_numerical_flux_interior!(
        ClimaComms.device(axes(dydt)),
        nothing,
        fn,
        dydt,
        args...,
    )
add_numerical_flux_interior!(
    ghost_exchange::DGGhostExchange,
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N} = _add_numerical_flux_interior!(
    ClimaComms.device(axes(dydt)),
    ghost_exchange,
    fn,
    dydt,
    args...,
)

_add_numerical_flux_interior!(
    device,
    ghost_exchange,
    fn::F,
    dydt,
    args...,
) where {F} = error(
    "add_numerical_flux_interior! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

# One-step argument access for the CPU face loops: `Field` and data-layout
# arguments are read at node `(i, j)` of level `v` in element `h` (`(i,)` for
# 1D horizontal elements); other arguments (e.g. equation parameters) pass
# through. Each face node then needs a single `unrolled_map` from `args` to
# values — a chain of Field→data→slab→value maps compiles one generated-map
# instantiation per stage, with the latency that entails. Data layouts are
# indexed directly (like the GPU face kernels) rather than through per-node
# slab views, which the compiler does not reliably elide. The indices are
# structurally in bounds: they come from `face_node_index` over `1:Nq` and
# the topology's face lists.
@inline _face_node_value(arg::Fields.Field, v, i, j, h) =
    _face_node_value(Fields.field_values(arg), v, i, j, h)
@inline _face_node_value(arg::DataLayouts.DataLayout, v, i, j, h) =
    @inbounds arg[CartesianIndex(v, i, j, h)]
@inline _face_node_value(arg, v, i, j, h) = arg
@inline _face_node_value_1d(arg::Fields.Field, v, i, h) =
    _face_node_value_1d(Fields.field_values(arg), v, i, h)
@inline _face_node_value_1d(arg::DataLayouts.DataLayout, v, i, h) =
    @inbounds arg[CartesianIndex(v, i, 1, h)]
@inline _face_node_value_1d(arg, v, i, h) = arg

# Face residual increments for one interior face node. Shared by numerical
# flux (antisymmetric) and symmetric lifting; geometry loops pass `mode`.
@inline function _face_side_increments(
    ::Val{:numflux},
    fn::F,
    sWJ,
    normal,
    argvals⁻,
    argvals⁺,
) where {F}
    val = add_auto_broadcasters(fn(normal, argvals⁻, argvals⁺))
    δ = sWJ * val
    return (-δ, δ)
end

@inline function _face_side_increments(
    ::Val{:lifting},
    fn::F,
    sWJ,
    normal,
    argvals⁻,
    argvals⁺,
) where {F}
    lift⁻ = add_auto_broadcasters(fn(normal, argvals⁻, argvals⁺))
    lift⁺ = add_auto_broadcasters(fn(-normal, argvals⁺, argvals⁻))
    return (sWJ * lift⁻, sWJ * lift⁺)
end
function _add_numerical_flux_interior!(
    ::ClimaComms.AbstractCPUDevice,
    ghost_exchange,
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N}
    _add_interior_face_flux!(Val(:numflux), ghost_exchange, fn, dydt, args...)
end

# Topology dispatch for CPU interior-face updates (numflux or lifting). The
# ghost-face halo exchange is started (or the shared `ghost_exchange` handle
# checked) before the interior loops so communication overlaps the local face
# work, and finished afterwards by the ghost-face loop in
# `_finish_dg_ghost_faces!`.
function _add_interior_face_flux!(
    mode::Val,
    ghost_exchange,
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N}
    space = axes(dydt)
    grid = Spaces.grid(space)
    if grid isa Grids.ExtrudedFiniteDifferenceGrid &&
       grid.horizontal_grid isa Grids.SpectralElementGrid1D
        # 1D horizontal topologies are single-process; a passed handle is
        # necessarily a no-op one and needs no consumption.
        return _add_interior_face_flux_extruded_1d!(mode, fn, dydt, args...)
    end
    ghost = _dg_shared_or_start(
        ClimaComms.device(space),
        space,
        args,
        ghost_exchange,
    )
    if grid isa Grids.ExtrudedFiniteDifferenceGrid
        _add_interior_face_flux_extruded_2d!(mode, fn, dydt, args...)
    else
        _add_interior_face_flux_2d!(mode, fn, dydt, args...)
    end
    return _finish_dg_ghost_faces!(mode, fn, dydt, args, ghost)
end

# Pure 2D spectral element space (precomputed interior surface geometry).
function _add_interior_face_flux_2d!(
    mode::Val,
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N}
    space = axes(dydt)
    grid = Spaces.grid(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    topology = Spaces.topology(space)
    interior_surface_geometry = grid.interior_surface_geometry
    dydt_data = Fields.field_values(dydt)

    for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
        enumerate(Topologies.interior_faces(topology))
        for q in 1:Nq
            sgeom⁻ =
                @inbounds interior_surface_geometry[CartesianIndex(
                    1,
                    q,
                    1,
                    iface,
                )]

            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

            argvals⁻ = unrolled_map(
                arg -> _face_node_value(arg, 1, i⁻, j⁻, elem⁻),
                args,
            )
            argvals⁺ = unrolled_map(
                arg -> _face_node_value(arg, 1, i⁺, j⁺, elem⁺),
                args,
            )

            δ⁻, δ⁺ = _face_side_increments(
                mode,
                fn,
                sgeom⁻.sWJ,
                sgeom⁻.normal,
                argvals⁻,
                argvals⁺,
            )
            I⁻ = CartesianIndex(1, i⁻, j⁻, elem⁻)
            I⁺ = CartesianIndex(1, i⁺, j⁺, elem⁺)
            @inbounds dydt_data[I⁻] = dydt_data[I⁻] + δ⁻
            @inbounds dydt_data[I⁺] = dydt_data[I⁺] + δ⁺
        end
    end
    return dydt
end

# Extruded plane: SpectralElementSpace1D × finite-difference vertical.
# Surface geometry from product local geometry (sWJ carries vertical measure).
function _add_interior_face_flux_extruded_1d!(
    mode::Val,
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N}
    space = axes(dydt)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nv = Spaces.nlevels(space)
    topology = Spaces.topology(space)
    local_geometry = Spaces.local_geometry_data(space)

    dydt_data = Fields.field_values(dydt)

    for v in 1:Nv
        for (elem⁻, face⁻, elem⁺, face⁺, _reversed) in
            Topologies.interior_faces(topology)

            i⁻ = face_node_index_1d(face⁻, Nq)
            i⁺ = face_node_index_1d(face⁺, Nq)

            lg⁻ = @inbounds local_geometry[CartesianIndex(v, i⁻, 1, elem⁻)]
            sgeom⁻ = compute_surface_geometry_1d(lg⁻, face⁻)

            argvals⁻ = unrolled_map(
                arg -> _face_node_value_1d(arg, v, i⁻, elem⁻),
                args,
            )
            argvals⁺ = unrolled_map(
                arg -> _face_node_value_1d(arg, v, i⁺, elem⁺),
                args,
            )

            δ⁻, δ⁺ = _face_side_increments(
                mode,
                fn,
                sgeom⁻.sWJ,
                sgeom⁻.normal,
                argvals⁻,
                argvals⁺,
            )
            I⁻ = CartesianIndex(v, i⁻, 1, elem⁻)
            I⁺ = CartesianIndex(v, i⁺, 1, elem⁺)
            @inbounds dydt_data[I⁻] = dydt_data[I⁻] + δ⁻
            @inbounds dydt_data[I⁺] = dydt_data[I⁺] + δ⁺
        end
    end
    return dydt
end

# Extruded 3D: SpectralElementSpace2D horizontal × finite-difference vertical
# (e.g. cubed-sphere shell). Normals in local orthonormal UV frame.
function _add_interior_face_flux_extruded_2d!(
    mode::Val,
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N}
    space = axes(dydt)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    Nv = Spaces.nlevels(space)
    topology = Spaces.topology(space)
    local_geometry = Spaces.local_geometry_data(space)
    FT = Spaces.undertype(space)
    (_, quad_weights) = Quadratures.quadrature_points(FT, quadrature_style)

    dydt_data = Fields.field_values(dydt)

    for v in 1:Nv
        for (elem⁻, face⁻, elem⁺, face⁺, reversed) in
            Topologies.interior_faces(topology)
            for q in 1:Nq
                i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
                i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

                lg⁻ =
                    @inbounds local_geometry[CartesianIndex(v, i⁻, j⁻, elem⁻)]
                sgeom⁻ = compute_surface_geometry_extruded_2d(
                    lg⁻,
                    quad_weights,
                    face⁻,
                    i⁻,
                    j⁻,
                )

                argvals⁻ = unrolled_map(
                    arg -> _face_node_value(arg, v, i⁻, j⁻, elem⁻),
                    args,
                )
                argvals⁺ = unrolled_map(
                    arg -> _face_node_value(arg, v, i⁺, j⁺, elem⁺),
                    args,
                )

                δ⁻, δ⁺ = _face_side_increments(
                    mode,
                    fn,
                    sgeom⁻.sWJ,
                    sgeom⁻.normal,
                    argvals⁻,
                    argvals⁺,
                )
                I⁻ = CartesianIndex(v, i⁻, j⁻, elem⁻)
                I⁺ = CartesianIndex(v, i⁺, j⁺, elem⁺)
                @inbounds dydt_data[I⁻] = dydt_data[I⁻] + δ⁻
                @inbounds dydt_data[I⁺] = dydt_data[I⁺] + δ⁺
            end
        end
    end
    return dydt
end
# ---------------------------------------------------------------------------
# Distributed (MPI) ghost faces
# ---------------------------------------------------------------------------

# Ghost faces — those whose "plus" element is owned by another rank — are
# skipped by `Topologies.interior_faces`, so the interior loops above miss
# them. Here each rank completes its own ("minus") side, reading the neighbour
# values from the `Topologies.GhostFaceExchange` recv strips (see there for the
# slot pairing and node order).
#
# Writing only the minus side is exact: the mirror ghost face on the
# neighbouring rank writes its own minus side, and the two match the
# single-rank result because the normal is single-valued up to sign
# (`n̂⁺ = -n̂⁻`), `sWJ` is single-valued on a conforming face, and the flux is
# antisymmetric (`fn(n̂, a, b) == -fn(-n̂, b, a)`). So the minus-side increment
# of `_face_side_increments` (numflux or lifting, as in the interior loops) is
# applied and its plus side is dropped.
#
# Handles pure 2D (`Nv == 1`) and extruded 2D-horizontal spaces, with the
# geometry built as in the interior extruded-2D loop.


function _start_dg_ghost_exchange(space, args)
    topology = Spaces.topology(space)
    # `nothing` (not an empty tuple) marks "no exchange started": a zero-
    # argument call on a distributed context returns `()` from the `ntuple`
    # below and must still run the ghost-face loop in
    # `_finish_dg_ghost_faces!`.
    ClimaComms.context(topology) isa ClimaComms.SingletonCommsContext &&
        return nothing

    # Exchange the ghost elements of every data argument through memoized
    # ghost buffers (non-data arguments, e.g. equation parameters, are
    # identical on both sides).
    args_local = unrolled_map(
        arg -> arg isa Fields.Field ? Fields.field_values(arg) : arg,
        args,
    )
    ghost_bufs = ntuple(Val(length(args_local))) do i
        a = args_local[i]
        a isa DataLayouts.DataLayout || return nothing
        _dg_face_exchange(space, a, i)
    end
    _claim_dg_face_exchanges!(ghost_bufs)
    foreach(ntuple(identity, Val(length(args_local)))) do i
        ex = ghost_bufs[i]
        isnothing(ex) || Topologies.fill_face_send_buffer!(args_local[i], ex)
    end
    foreach(
        ex -> isnothing(ex) || ClimaComms.start(ex.graph_context),
        ghost_bufs,
    )
    return ghost_bufs
end

function _finish_dg_ghost_faces!(
    mode::Val,
    fn::F,
    dydt,
    args,
    ghost::DGGhostExchange,
) where {F}
    ghost_bufs = _consume_dg_ghost_exchange!(ghost)
    isnothing(ghost_bufs) && return dydt

    space = axes(dydt)
    topology = Spaces.topology(space)
    gfaces = Topologies.ghost_faces(topology)
    isempty(gfaces) && return dydt

    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    Nv = Spaces.nlevels(space)
    FT = Spaces.undertype(space)
    (_, quad_weights) = Quadratures.quadrature_points(FT, quadrature_style)
    local_geometry = Spaces.local_geometry_data(space)
    dydt_data = Fields.field_values(dydt)
    # The slot schedule is identical across the exchanged arguments; with no
    # exchanged argument (an argument's `ghost_bufs` entry is `nothing`),
    # slots are never read.
    exs = unrolled_filter(!isnothing, ghost_bufs)
    face_slot = isempty(exs) ? nothing : first(exs).face_slot

    # The face-node indices are structurally in bounds: `i⁻`, `j⁻`, and `q′`
    # come from `face_node_index` over 1:Nq, and elements and strip slots come
    # from the topology's ghost-face list and schedule.
    for v in 1:Nv
        for (f, (elem⁻, face⁻, _ridx⁺, _face⁺, reversed)) in enumerate(gfaces)
            slot = isnothing(face_slot) ? 0 : Int(face_slot[f])
            for q in 1:Nq
                i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
                q′ = reversed ? Nq - q + 1 : q

                lg⁻ =
                    @inbounds local_geometry[CartesianIndex(v, i⁻, j⁻, elem⁻)]
                sgeom⁻ = compute_surface_geometry_extruded_2d(
                    lg⁻,
                    quad_weights,
                    face⁻,
                    i⁻,
                    j⁻,
                )

                argvals⁻ = unrolled_map(
                    arg -> _face_node_value(arg, v, i⁻, j⁻, elem⁻),
                    args,
                )
                # A non-exchanged argument (no strip buffer) is identical on
                # both sides.
                argvals⁺ = unrolled_map(
                    (arg, ex) ->
                        isnothing(ex) ? arg :
                        (@inbounds ex.recv_data[CartesianIndex(
                            v,
                            q′,
                            1,
                            slot,
                        )]),
                    args,
                    ghost_bufs,
                )

                # Only δ⁻ is used; the plus side lives on the neighbour rank.
                δ⁻, _ = _face_side_increments(
                    mode,
                    fn,
                    sgeom⁻.sWJ,
                    sgeom⁻.normal,
                    argvals⁻,
                    argvals⁺,
                )
                I⁻ = CartesianIndex(v, i⁻, j⁻, elem⁻)
                @inbounds dydt_data[I⁻] = dydt_data[I⁻] + δ⁻
            end
        end
    end
    return dydt
end

_add_dg_ghost_faces!(mode::Val, fn::F, dydt, args) where {F} =
    _finish_dg_ghost_faces!(
        mode,
        fn,
        dydt,
        args,
        DGGhostExchange(_start_dg_ghost_exchange(axes(dydt), args)),
    )

@inline _extract_field_space(a::Fields.Field, rest...) = axes(a)
@inline _extract_field_space(a, rest...) = _extract_field_space(rest...)
@inline _extract_field_space() =
    error("start_dg_ghost_exchange requires at least one Field argument")

"""
    start_dg_ghost_exchange(args...)
    start_dg_ghost_exchange(space, args...)

Start the ghost-face halo exchange of the DG face operators once, to be
shared by several operator calls in the same tendency evaluation:

    ex = Operators.start_dg_ghost_exchange(y)
    # ... element-local volume terms (the exchange overlaps them) ...
    Operators.add_numerical_flux_interior!(ex, numflux, dydt, y)
    Operators.add_lifting_flux_interior!(ex, lift, dydt2, y)

Without the leading handle each operator performs its own exchange, so an RHS that
applies several face operators to the same state sends every halo message
once per operator; a shared exchange sends each once. The space is taken from
the first `Field` in `args`, or passed as a leading argument.

Contract: every operator receiving the handle must be called with the same
`args` (the same fields, in the same order). Mismatched argument types or
counts throw; two same-typed fields swapped in place cannot be detected. The
handle covers one exchange round — start a fresh one whenever an argument's
values change (e.g. each stage), and pass a started handle to at least one
operator call before starting the next round: the underlying buffers are
shared per argument type and position on a space (also across *distinct*
fields of the same type), and starting a round while another is in flight
throws. Returns a no-op handle on single-process contexts, so calling code
needs no distributed-vs-single branch.
"""
@inline start_dg_ghost_exchange(args...) =
    start_dg_ghost_exchange(_extract_field_space(args...), args...)

@inline function start_dg_ghost_exchange(space::Spaces.AbstractSpace, args...)
    return _start_dg_ghost_exchange_handle(
        ClimaComms.device(space),
        space,
        args,
    )
end

@inline function _start_dg_ghost_exchange_handle(
    ::ClimaComms.AbstractCPUDevice,
    space,
    args,
)
    bufs = _start_dg_ghost_exchange(space, args)
    isnothing(bufs) && return NO_DG_GHOST_EXCHANGE
    return DGGhostExchange(bufs)
end

# The exchange to use for one face-operator call: the operator's own (started
# here) unless a shared handle was passed, which is checked against the
# operator's arguments. The `::Nothing` methods are device-specific (the CUDA
# method, in ext/cuda/operators_dg.jl, packs with kernels).
@inline _dg_shared_or_start(device::ClimaComms.AbstractCPUDevice, space, args, ::Nothing) =
    _start_dg_ghost_exchange_handle(device, space, args)
@inline function _dg_shared_or_start(device, space, args, ghost_exchange::DGGhostExchange)
    _check_dg_ghost_exchange(space, args, ghost_exchange)
    return ghost_exchange
end

# Complete the exchange on first consumption (releasing the in-flight latch
# of each buffer); later consumers get the recv strips as-is. Returns
# `nothing` for a no-op (single-process) handle.
@inline function _consume_dg_ghost_exchange!(ghost::DGGhostExchange)
    bufs = ghost.bufs
    isnothing(bufs) && return nothing
    if !ghost.finished[]
        foreach(bufs) do ex
            isnothing(ex) && return
            ClimaComms.finish(ex.graph_context)
            ex.in_flight[] = false
        end
        ghost.finished[] = true
    end
    return bufs
end

# A shared exchange must have been started with the arguments the consuming
# operator receives. The exchanges are memoized per (grid, data type,
# position), so recomputing them here is a few dictionary hits; equal-typed
# fields swapped in place map to the same exchange objects and cannot be
# detected — that part of the contract stays with the caller.
function _check_dg_ghost_exchange(space, args, ghost::DGGhostExchange)
    bufs = ghost.bufs
    isnothing(bufs) && return nothing
    args_local = unrolled_map(
        arg -> arg isa Fields.Field ? Fields.field_values(arg) : arg,
        args,
    )
    length(args_local) == length(bufs) || error(
        "shared DG ghost exchange was started with $(length(bufs)) \
         arguments, but the consuming operator received $(length(args_local))",
    )
    foreach(ntuple(identity, Val(length(bufs)))) do i
        a = args_local[i]
        expected =
            a isa DataLayouts.DataLayout ? _dg_face_exchange(space, a, i) :
            nothing
        expected === bufs[i] || error(
            "shared DG ghost exchange does not match operator argument $i: \
             start_dg_ghost_exchange must receive the same arguments (same \
             fields, same order) as every operator that consumes it",
        )
    end
    return nothing
end

# Memoized `Topologies.GhostFaceExchange` for the `i`-th exchanged argument
# of a DG ghost-face call, keyed like `dg_connectivity` (so
# `Utilities.Cache.clean_cache!` releases it). The exchange is created once
# per key, matching the create-once pattern of the limiter and DSS buffers:
# each `ClimaComms.graph_context` draws a fresh MPI tag from a counter that
# wraps at 32767, so an exchange per call would reallocate the strips every
# tendency evaluation and eventually alias the tag of a live graph context.
# The argument position `i` is part of the key because same-typed arguments in
# one call each need their own send buffer. `nothing` (no ghost faces) is a
# valid cached value, hence the `:missing` sentinel. The key is deliberately
# not specific to the field instance — keying on identity would mint a graph
# context (and burn an MPI tag) per field ever exchanged — so distinct fields
# of the same type at the same position share one buffer; the in-flight latch
# below makes an overlapping second use an error instead of data corruption.
function _dg_face_exchange(space, data, i)
    key = (Topologies.GhostFaceExchange, Spaces.grid(space), typeof(data), i)
    ex = get(Cache.OBJECT_CACHE, key, :missing)
    if ex === :missing
        ex = Topologies.create_ghost_face_exchange(data, Spaces.topology(space))
        Cache.OBJECT_CACHE[key] = ex
    end
    return ex
end

# Claim the exchanges' buffers for one round (released by
# `_consume_dg_ghost_exchange!`): a second start before the first round is
# consumed — e.g. `start_dg_ghost_exchange(a)` then
# `start_dg_ghost_exchange(b)` for same-typed fields `a` and `b` — would
# overwrite the in-flight send strips and double-start the graph context.
# All latches are checked before any is set, so a rejected start leaves
# every buffer (including those of its other arguments) untouched.
function _claim_dg_face_exchanges!(ghost_bufs)
    foreach(ntuple(identity, Val(length(ghost_bufs)))) do i
        ex = ghost_bufs[i]
        isnothing(ex) && return
        ex.in_flight[] && error(
            "the DG ghost-face exchange for operator argument $i is still \
             in flight; pass the previously started handle to a face \
             operator (consuming it) before starting a new exchange. \
             Exchange buffers are shared by all fields of the same type at \
             the same argument position on a space.",
        )
    end
    foreach(ex -> isnothing(ex) || (ex.in_flight[] = true), ghost_bufs)
    return nothing
end
"""
    PeriodicBC <: HorizontalBoundaryCondition

Periodic boundary condition for the horizontal DG numerical-flux operators,
handled by the topology (no ghost state needed).
"""
struct PeriodicBC <: HorizontalBoundaryCondition end

"""
    ReflectingWallBC <: HorizontalBoundaryCondition

Reflecting-wall (no-normal-flow) boundary condition for the horizontal DG
numerical-flux operators: its [`ghost_state`](@ref) reflects the normal
momentum component and preserves the other prognostic fields.
"""
struct ReflectingWallBC <: HorizontalBoundaryCondition end

"""
    ghost_state(bc::HorizontalBoundaryCondition, normal, argvals⁻)

Construct the exterior-side argument tuple for the given BC.

Returns a tuple with the same length as `argvals⁻`, replacing only the
prognostic state `argvals⁻[1]` with the ghost state; remaining arguments
(e.g. equation parameters, coordinates) are forwarded unchanged.
"""
function ghost_state(::HorizontalBoundaryCondition, normal, argvals⁻)
    error("ghost_state not implemented for this boundary condition")
end

function ghost_state(::ReflectingWallBC, normal, argvals⁻)
    y⁻ = argvals⁻[1]
    ρu⁺ = y⁻.ρu - 2 * LinearAlgebra.dot(y⁻.ρu, normal) * normal
    # y⁻ may arrive wrapped in an AutoBroadcaster at element boundaries;
    # unwrap before merge so we always work with a plain NamedTuple.
    y⁺ = merge(unwrap(y⁻), (ρu = ρu⁺,))
    return (y⁺, argvals⁻[2:end]...)
end

"""
    add_numerical_flux_boundary!(fn, dydt, args...)

Add the numerical flux at the domain-boundary faces of the spectral space
mesh:

    dydt -= sWJ * fn(normal, argvals⁻)

per boundary face node, where `normal` is the outward unit normal and
`argvals⁻` is the tuple of values of `args` at that node. `dydt` must be in
mass-weighted residual form (`WJ * ∂Y/∂t`), matching
[`add_numerical_flux_interior!`](@ref). Implemented for pure 2D spectral
element spaces and extruded spaces with 2D horizontal spectral elements
(``sWJ`` then carries the vertical measure). No-op on domains without boundary
faces (e.g. the sphere).
"""
@inline add_numerical_flux_boundary!(
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N} = _add_numerical_flux_boundary!(
    ClimaComms.device(axes(dydt)),
    fn,
    dydt,
    args...,
)

_add_numerical_flux_boundary!(device, fn::F, dydt, args...) where {F} = error(
    "add_numerical_flux_boundary! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

function _add_numerical_flux_boundary!(
    ::ClimaComms.AbstractCPUDevice,
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N}
    space = axes(dydt)
    grid = Spaces.grid(space)
    if grid isa Grids.ExtrudedFiniteDifferenceGrid &&
       grid.horizontal_grid isa Grids.SpectralElementGrid1D
        error(
            "add_numerical_flux_boundary! is not implemented for extruded \
             spaces with 1D horizontal spectral elements",
        )
    end
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    Nv = Spaces.nlevels(space)
    FT = Spaces.undertype(space)
    (_, quad_weights) = Quadratures.quadrature_points(FT, quadrature_style)
    topology = Spaces.topology(space)
    # The surface geometry is built from the product local geometry (like the
    # interior extruded-2D loop), so `sWJ` carries the vertical measure on
    # extruded spaces; for pure-2D spaces this matches the grid's precomputed
    # `boundary_surface_geometries`.
    local_geometry = Spaces.local_geometry_data(space)
    dydt_data = Fields.field_values(dydt)

    for boundarytag in Topologies.boundary_tags(topology)
        for (elem⁻, face⁻) in Topologies.boundary_faces(topology, boundarytag)
            for v in 1:Nv
                for q in 1:Nq
                    i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
                    lg⁻ = @inbounds local_geometry[CartesianIndex(
                        v,
                        i⁻,
                        j⁻,
                        elem⁻,
                    )]
                    sgeom⁻ = compute_surface_geometry_extruded_2d(
                        lg⁻,
                        quad_weights,
                        face⁻,
                        i⁻,
                        j⁻,
                    )
                    argvals⁻ = unrolled_map(
                        arg -> _face_node_value(arg, v, i⁻, j⁻, elem⁻),
                        args,
                    )
                    # Wrap so a multi-field (NamedTuple/vector) flux value
                    # scales and subtracts elementwise, matching the interior
                    # loops and the GPU boundary kernel's `_fd_scale`.
                    numflux⁻ = add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻))
                    I⁻ = CartesianIndex(v, i⁻, j⁻, elem⁻)
                    @inbounds dydt_data[I⁻] =
                        dydt_data[I⁻] - (sgeom⁻.sWJ * numflux⁻)
                end
            end
        end
    end
    return dydt
end

"""
    add_numerical_flux_boundary!(numflux::AbstractNumericalFlux, bc::HorizontalBoundaryCondition, dydt, args...)

Add numerical flux at boundaries using a typed boundary condition.
Constructs the ghost state via `ghost_state(bc, normal, argvals⁻)` and applies the numerical flux.
"""
function add_numerical_flux_boundary!(
    numflux::AbstractNumericalFlux,
    bc::HorizontalBoundaryCondition,
    dydt,
    args::Vararg{Any, N},
) where {N}
    add_numerical_flux_boundary!(dydt, args...) do normal, argvals⁻
        argvals⁺ = ghost_state(bc, normal, argvals⁻)
        numflux(normal, argvals⁻, argvals⁺)
    end
end

# A boundary condition of the wrong family (a vertical finite-difference BC, or
# a direct `AbstractBoundaryCondition` subtype) would otherwise bind to the
# `fn`-taking method above as `dydt` and fail later with an unrelated message,
# so it goes through the same rejection the finite-difference operators use and
# both families report an out-of-family boundary condition the same way.
add_numerical_flux_boundary!(
    numflux::AbstractNumericalFlux,
    bc::AbstractBoundaryCondition,
    dydt,
    args...,
) = invalid_boundary_condition_error(typeof(numflux), typeof(bc))
# ---------------------------------------------------------------------------
# Symmetric face lifting for non-conservative (gradient / curl) terms
# ---------------------------------------------------------------------------

"""
    add_lifting_flux_interior!(fn, dydt, args...)
    add_lifting_flux_interior!(ghost_exchange, fn, dydt, args...)

Add *symmetric* face lifting terms at interior faces — the DG correction for
non-conservative (gradient / curl) terms, where both sides of a face receive
their own correction rather than equal-and-opposite fluxes:

    dydt⁻ += sWJ * fn(n̂⁻, argvals⁻, argvals⁺)
    dydt⁺ += sWJ * fn(n̂⁺, argvals⁺, argvals⁻)

with `n̂⁻ = -n̂⁺` the outward unit normals. For example, the strong-form DG
gradient of a scalar `q` is completed by `fn(n̂, (q⁻,), (q⁺,)) = ((q⁺ − q⁻)/2) * n̂`
(the lifting of `(q* − q⁻) n̂` with a central interface value `q*`).

`dydt` must be in mass-weighted residual form (`WJ * ∂Y/∂t`), matching
[`add_numerical_flux_interior!`](@ref). Implemented for pure 2D spectral
element spaces and for extruded spaces with 1D (plane) or 2D (e.g.
cubed-sphere) horizontal spectral elements. The method with a leading
`ghost_exchange` consumes a shared halo exchange from
[`start_dg_ghost_exchange`](@ref) on distributed spaces.
"""
add_lifting_flux_interior!(fn::F, dydt, args::Vararg{Any, N}) where {F, N} =
    _add_lifting_flux_interior!(
        ClimaComms.device(axes(dydt)),
        nothing,
        fn,
        dydt,
        args...,
    )
add_lifting_flux_interior!(
    ghost_exchange::DGGhostExchange,
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N} = _add_lifting_flux_interior!(
    ClimaComms.device(axes(dydt)),
    ghost_exchange,
    fn,
    dydt,
    args...,
)

_add_lifting_flux_interior!(
    device,
    ghost_exchange,
    fn::F,
    dydt,
    args...,
) where {F} = error(
    "add_lifting_flux_interior! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

function _add_lifting_flux_interior!(
    ::ClimaComms.AbstractCPUDevice,
    ghost_exchange,
    fn::F,
    dydt,
    args::Vararg{Any, N},
) where {F, N}
    _add_interior_face_flux!(Val(:lifting), ghost_exchange, fn, dydt, args...)
end

"""
    lifting_correction(fn, ::Type{T}, args...)

WJ-normalized DG face-lifting correction field of element type `T`: applies
[`add_lifting_flux_interior!`](@ref) with face function `fn` to a zero
residual on the space of `args[1]` and divides by `WJ`. The result is the
correction to the corresponding element-local strong-form operator.
"""
function lifting_correction(fn::F, ::Type{T}, args...) where {F, T}
    space = axes(args[1])
    lgeom = Fields.local_geometry_field(space)
    r = similar(args[1], T)
    fill!(parent(r), 0)
    add_lifting_flux_interior!(fn, r, args...)
    return r ./ lgeom.WJ
end
# ---------------------------------------------------------------------------
# Flux-differencing (split-form / FDDG) volume divergence
# ---------------------------------------------------------------------------

@inline _fd_add(a::NamedTuple, b::NamedTuple) = map(_fd_add, a, b)
@inline _fd_add(a, b) = a + b

@inline _fd_scale(c, x::NamedTuple) = map(v -> _fd_scale(c, v), x)
@inline _fd_scale(c, x) = c * x

# Metric-scaled contravariant basis vector J ∂ξʳᵒʷ/∂x, projected onto the
# local orthonormal horizontal frame (single-valued at shared nodes, including
# across cubed-sphere panel edges).
@inline _fd_metric_vector(local_geometry, row) = Geometry.project(
    Geometry.UVAxis(),
    local_geometry.J * local_geometry.∂ξ∂x[row, :],
)

"""
    add_flux_differencing_divergence!(fn2pt, dydt, y)

Add the horizontal flux-differencing (split-form / FDDG) volume divergence to
the mass-weighted residual `dydt` (Souza et al. 2023, JAMES, Eqs. 25–30):
the collocation derivative acts on symmetric two-point fluxes along each
reference direction.

`fn2pt(nvec_a, nvec_b, y_a, y_b)` returns the two-point flux contracted with
the (non-unit) nodal metric vectors in the local orthonormal horizontal frame.
It must be jointly linear in `(nvec_a, nvec_b)`, symmetric under
`(nvec_a, y_a) ↔ (nvec_b, y_b)`, and consistent
(`fn2pt(n, n, y, y) == F(y)⋅n`). Kinetic-energy / entropy properties are
fixed by this choice (e.g. Kennedy–Gruber → KEP).

Stored in weak-equivalent form (strong flux-differencing plus one-sided
boundary lifts), so it replaces `dydt = hwdiv(F) * (-WJ)` and composes with
[`add_numerical_flux_interior!`](@ref) to give the FDDG SAT
``F^* - F(y^-)⋅n̂``. SBP telescoping gives local conservation; global
conservation follows from antisymmetric interface fluxes.

Supports pure 2D spectral elements and extruded spaces with 2D horizontal
elements.
"""
add_flux_differencing_divergence!(fn2pt::F, dydt, y) where {F} =
    _add_flux_differencing_divergence!(
        ClimaComms.device(axes(dydt)),
        fn2pt,
        dydt,
        y,
    )

_add_flux_differencing_divergence!(device, fn2pt::F, dydt, y) where {F} = error(
    "add_flux_differencing_divergence! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

function _add_flux_differencing_divergence!(
    ::ClimaComms.AbstractCPUDevice,
    fn2pt::F,
    dydt,
    y,
) where {F}
    space = axes(dydt)
    grid = Spaces.grid(space)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    FT = Spaces.undertype(space)
    (_, w) = Quadratures.quadrature_points(FT, quadrature_style)
    D = Quadratures.differentiation_matrix(FT, quadrature_style)
    topology = Spaces.topology(space)
    Nh = Topologies.nlocalelems(topology)
    local_geometry = Spaces.local_geometry_data(space)
    dydt_data = Fields.field_values(dydt)
    y_data = Fields.field_values(y)

    if grid isa Grids.ExtrudedFiniteDifferenceGrid
        @assert grid.horizontal_grid isa Grids.SpectralElementGrid2D
        Nv = Spaces.nlevels(space)
        for h in 1:Nh, v in 1:Nv
            _fd_divergence_slab!(
                fn2pt,
                slab(dydt_data, v, h),
                slab(y_data, v, h),
                slab(local_geometry, v, h),
                D,
                w,
                Val(Nq),
            )
        end
    else
        @assert grid isa Grids.SpectralElementGrid2D
        for h in 1:Nh
            _fd_divergence_slab!(
                fn2pt,
                slab(dydt_data, h),
                slab(y_data, h),
                slab(local_geometry, h),
                D,
                w,
                Val(Nq),
            )
        end
    end
    return dydt
end

# Per-node flux-differencing body, shared by the CPU slab loop and
# the CUDA kernel. Returns the
# mass-weighted contribution (strong-form FD sum with coefficient
# −2 wᵢ wⱼ D, plus the one-sided consistent-flux boundary lifts of the
# weak-equivalent form; the outward sWJ·n̂ is ±(J a¹) wⱼ / ±(J a²) wᵢ,
# matching compute_surface_geometry).
@inline function _fd_volume_node_total(
    fn2pt::F,
    y_at::Y,
    lg_at::L,
    D,
    w,
    ::Val{Nq},
    i,
    j,
) where {F, Y, L, Nq}
    lg = lg_at(i, j)
    Ja1 = _fd_metric_vector(lg, 1)
    Ja2 = _fd_metric_vector(lg, 2)
    y_ij = y_at(i, j)

    c1 = -2 * w[i] * w[j] * D[i, 1]
    total = fn2pt(
        c1 * Ja1,
        c1 * _fd_metric_vector(lg_at(1, j), 1),
        y_ij,
        y_at(1, j),
    )
    c2 = -2 * w[i] * w[j] * D[j, 1]
    total = _fd_add(
        total,
        fn2pt(
            c2 * Ja2,
            c2 * _fd_metric_vector(lg_at(i, 1), 2),
            y_ij,
            y_at(i, 1),
        ),
    )
    for k in 2:Nq
        c1 = -2 * w[i] * w[j] * D[i, k]
        t1 = fn2pt(
            c1 * Ja1,
            c1 * _fd_metric_vector(lg_at(k, j), 1),
            y_ij,
            y_at(k, j),
        )
        c2 = -2 * w[i] * w[j] * D[j, k]
        t2 = fn2pt(
            c2 * Ja2,
            c2 * _fd_metric_vector(lg_at(i, k), 2),
            y_ij,
            y_at(i, k),
        )
        total = _fd_add(total, _fd_add(t1, t2))
    end

    i == 1 &&
        (total = _fd_add(total, fn2pt(-w[j] * Ja1, -w[j] * Ja1, y_ij, y_ij)))
    i == Nq &&
        (total = _fd_add(total, fn2pt(w[j] * Ja1, w[j] * Ja1, y_ij, y_ij)))
    j == 1 &&
        (total = _fd_add(total, fn2pt(-w[i] * Ja2, -w[i] * Ja2, y_ij, y_ij)))
    j == Nq &&
        (total = _fd_add(total, fn2pt(w[i] * Ja2, w[i] * Ja2, y_ij, y_ij)))
    return total
end

function _fd_divergence_slab!(
    fn2pt::F,
    dydt_slab,
    y_slab,
    lg_slab,
    D,
    w,
    ::Val{Nq},
) where {F, Nq}
    # `let` rebinds the slabs so the node accessors are type-stable and do not
    # box; `Val{Nq}` keeps the quadrature size in the type domain so the
    # shared `_fd_volume_node_total` loop can unroll.
    let y_slab = y_slab, lg_slab = lg_slab
        y_at = (a, b) -> y_slab[1, a, b, 1]
        lg_at = (a, b) -> lg_slab[1, a, b, 1]
        for j in 1:Nq, i in 1:Nq
            total = _fd_volume_node_total(
                fn2pt,
                y_at,
                lg_at,
                D,
                w,
                Val(Nq),
                i,
                j,
            )
            dydt_slab[1, i, j, 1] =
                dydt_slab[1, i, j, 1] + add_auto_broadcasters(total)
        end
    end
    return dydt_slab
end
# ---------------------------------------------------------------------------
# DG connectivity buffer (device-resident; used by the GPU face kernels)
# ---------------------------------------------------------------------------

"""
    DGConnectivity

Cached, device-resident connectivity and face geometry for the DG
interior-face operators (the DSS-buffer analog for DG):

  - `faces`: `5 × nfaces` `Int32` matrix of interior faces
    `(elem⁻, face⁻, elem⁺, face⁺, reversed)`;
  - `sgeom`: precomputed [`Geometry.SurfaceGeometry`](@ref) per
    `(level, q, face)` (level = 1 for pure 2D spaces), evaluated from the
    minus side exactly as the CPU loops do;
  - a deterministic gather map from element boundary nodes to their face
    contributions, in ragged-array form (`node_*`, `node_offset`,
    `contrib_*`): each boundary node `(elem, i, j)` lists the
    `(face, side, q)` face-node slots that accumulate into it (2 entries at
    element corners, 1 elsewhere), sorted at construction so the GPU gather
    is bitwise deterministic.

Built once per space by [`dg_connectivity`](@ref) and stored with the array
type of the space's device (`ClimaComms.array_type`).
[`dg_ghost_connectivity`](@ref) builds the same structure over the ghost
(inter-rank) faces, with the meanings documented there.
"""
struct DGConnectivity{FA, SG, IV}
    nfaces::Int
    nbnodes::Int
    faces::FA
    sgeom::SG
    node_elem::IV
    node_i::IV
    node_j::IV
    node_offset::IV
    contrib_face::IV
    contrib_side::IV
    contrib_q::IV
end

"""
    dg_connectivity(space)

Memoized [`DGConnectivity`](@ref) for `space`, keyed on the underlying grid
and the space type (so center/face extruded spaces get separate buffers).
Stored in `Utilities.Cache.OBJECT_CACHE` alongside the grid objects, so
`Utilities.Cache.clean_cache!` releases it (the buffer holds device arrays).
"""
function dg_connectivity(space)
    key = (DGConnectivity, Spaces.grid(space), typeof(space))
    return get!(() -> build_dg_connectivity(space), Cache.OBJECT_CACHE, key)
end

# Memoized device staging array of shape `(Nv, Nq, nsides, nfaces)` for the GPU
# face kernels, keyed (and released by `clean_cache!`) alongside the space's
# connectivity. `tag` separates the interior (`nsides` 1 or 2) and ghost
# (`nsides == 1`) buffers, which the same operator call uses in turn.
function _dg_staging_buffer(
    space,
    ::Type{T},
    nsides,
    nfaces;
    tag = :DGStagingBuffer,
) where {T}
    key = (tag, Spaces.grid(space), typeof(space), T, nsides, nfaces)
    buf = get(Cache.OBJECT_CACHE, key, nothing)
    if isnothing(buf)
        Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
        Nv = Spaces.nlevels(space)
        DA = ClimaComms.array_type(Spaces.topology(space))
        buf = DA{T}(undef, Nv, Nq, nsides, nfaces)
        Cache.OBJECT_CACHE[key] = buf
    end
    return buf
end

_dg_ghost_staging_buffer(space, ::Type{T}, nfaces) where {T} =
    _dg_staging_buffer(space, T, 1, nfaces; tag = :DGGhostStagingBuffer)

_dg_boundary_staging_buffer(space, ::Type{T}, nfaces) where {T} =
    _dg_staging_buffer(space, T, 1, nfaces; tag = :DGBoundaryStagingBuffer)

function build_dg_connectivity(space)
    topology = Spaces.topology(space)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    FT = Spaces.undertype(space)
    Nv = Spaces.nlevels(space)
    (_, w) = Quadratures.quadrature_points(FT, quadrature_style)
    DA = ClimaComms.array_type(topology)

    ifaces = collect(Topologies.interior_faces(topology))
    nfaces = length(ifaces)
    faces = Matrix{Int32}(undef, 5, nfaces)

    lg_host = Adapt.adapt(Array, Spaces.local_geometry_data(space))
    SG = Geometry.SurfaceGeometry{FT, Geometry.UVVector{FT}}
    sgeom = Array{SG}(undef, Nv, Nq, nfaces)

    # (elem, i, j) → list of (face, side, q); side 1 = minus, 2 = plus
    contrib = Dict{NTuple{3, Int}, Vector{NTuple{3, Int32}}}()
    for (f, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in enumerate(ifaces)
        faces[:, f] .=
            (elem⁻, face⁻, elem⁺, face⁺, reversed ? Int32(1) : Int32(0))
        for q in 1:Nq
            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)
            push!(
                get!(() -> NTuple{3, Int32}[], contrib, (elem⁻, i⁻, j⁻)),
                (Int32(f), Int32(1), Int32(q)),
            )
            push!(
                get!(() -> NTuple{3, Int32}[], contrib, (elem⁺, i⁺, j⁺)),
                (Int32(f), Int32(2), Int32(q)),
            )
            for v in 1:Nv
                lg = slab(lg_host, v, elem⁻)[1, i⁻, j⁻, 1]
                sgeom[v, q, f] = compute_surface_geometry_extruded_2d(
                    lg,
                    w,
                    face⁻,
                    i⁻,
                    j⁻,
                )
            end
        end
    end

    return DGConnectivity(nfaces, faces, sgeom, contrib, DA)
end

# Assemble a `DGConnectivity` from the host-side face matrix, surface
# geometry, and `(elem, i, j) → [(face, side, q), ...]` contribution map: the
# contributions of each boundary node are sorted so the GPU gather is bitwise
# deterministic, then everything is adapted to the device array type `DA`.
function DGConnectivity(nfaces, faces, sgeom, contrib, DA)
    bnodes = sort!(collect(keys(contrib)))
    nbnodes = length(bnodes)
    node_elem = Vector{Int32}(undef, nbnodes)
    node_i = Vector{Int32}(undef, nbnodes)
    node_j = Vector{Int32}(undef, nbnodes)
    node_offset = Vector{Int32}(undef, nbnodes + 1)
    contrib_face = Int32[]
    contrib_side = Int32[]
    contrib_q = Int32[]
    node_offset[1] = 1
    for (n, key) in enumerate(bnodes)
        (elem, i, j) = key
        node_elem[n] = elem
        node_i[n] = i
        node_j[n] = j
        entries = sort!(contrib[key])
        for (f, side, q) in entries
            push!(contrib_face, f)
            push!(contrib_side, side)
            push!(contrib_q, q)
        end
        node_offset[n + 1] = node_offset[n] + length(entries)
    end

    return DGConnectivity(
        nfaces,
        nbnodes,
        DA(faces),
        DA(sgeom),
        DA(node_elem),
        DA(node_i),
        DA(node_j),
        DA(node_offset),
        DA(contrib_face),
        DA(contrib_side),
        DA(contrib_q),
    )
end

"""
    dg_ghost_connectivity(space)

Memoized [`DGConnectivity`](@ref) over the topology's ghost (inter-rank)
faces, or `nothing` when there are none. The entries differ from
[`dg_connectivity`](@ref) in two ways: the third `faces` row holds the strip
slot into the recv buffer of a `Topologies.GhostFaceExchange` (the plus-side
value at loop node `q` is strip node `reversed ? Nq - q + 1 : q`) rather than
a local element, and the gather map contains only minus-side contributions
(`side == 1`) — the mirror ghost face on the neighbouring rank accumulates
the other side (see `_add_dg_ghost_faces!` for the operator contract that
makes the two sides consistent).
"""
function dg_ghost_connectivity(space)
    key = (DGConnectivity, :ghost, Spaces.grid(space), typeof(space))
    conn = get(Cache.OBJECT_CACHE, key, :missing)
    if conn === :missing
        conn = build_dg_ghost_connectivity(space)
        Cache.OBJECT_CACHE[key] = conn
    end
    return conn
end

function build_dg_ghost_connectivity(space)
    topology = Spaces.topology(space)
    gfaces = Topologies.ghost_faces(topology)
    isempty(gfaces) && return nothing
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    FT = Spaces.undertype(space)
    Nv = Spaces.nlevels(space)
    (_, w) = Quadratures.quadrature_points(FT, quadrature_style)
    DA = ClimaComms.array_type(topology)

    nfaces = length(gfaces)
    faces = Matrix{Int32}(undef, 5, nfaces)
    lg_host = Adapt.adapt(Array, Spaces.local_geometry_data(space))
    SG = Geometry.SurfaceGeometry{FT, Geometry.UVVector{FT}}
    sgeom = Array{SG}(undef, Nv, Nq, nfaces)

    (; face_slot) = Topologies.ghost_face_schedule(topology)
    contrib = Dict{NTuple{3, Int}, Vector{NTuple{3, Int32}}}()
    for (f, (elem⁻, face⁻, _ridx⁺, face⁺, reversed)) in enumerate(gfaces)
        faces[:, f] .=
            (elem⁻, face⁻, face_slot[f], face⁺, reversed ? Int32(1) : Int32(0))
        for q in 1:Nq
            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            push!(
                get!(() -> NTuple{3, Int32}[], contrib, (elem⁻, i⁻, j⁻)),
                (Int32(f), Int32(1), Int32(q)),
            )
            for v in 1:Nv
                lg = slab(lg_host, v, elem⁻)[1, i⁻, j⁻, 1]
                sgeom[v, q, f] = compute_surface_geometry_extruded_2d(
                    lg,
                    w,
                    face⁻,
                    i⁻,
                    j⁻,
                )
            end
        end
    end

    return DGConnectivity(nfaces, faces, sgeom, contrib, DA)
end

"""
    dg_boundary_connectivity(space)

Memoized [`DGConnectivity`](@ref) over the topology's boundary faces (all
boundary tags, concatenated in `Topologies.boundary_tags` order), or `nothing`
when there are none (e.g. on the sphere). Boundary faces have no plus side:
`faces` holds `(elem⁻, face⁻)` rows only, and the gather map contains only
minus-side contributions — a node at a domain corner receives one contribution
from each of its two boundary faces.
"""
function dg_boundary_connectivity(space)
    key = (DGConnectivity, :boundary, Spaces.grid(space), typeof(space))
    conn = get(Cache.OBJECT_CACHE, key, :missing)
    if conn === :missing
        conn = build_dg_boundary_connectivity(space)
        Cache.OBJECT_CACHE[key] = conn
    end
    return conn
end

function build_dg_boundary_connectivity(space)
    topology = Spaces.topology(space)
    bfaces = Tuple{Int, Int}[]
    for boundarytag in Topologies.boundary_tags(topology)
        append!(bfaces, Topologies.boundary_faces(topology, boundarytag))
    end
    isempty(bfaces) && return nothing
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    FT = Spaces.undertype(space)
    Nv = Spaces.nlevels(space)
    (_, w) = Quadratures.quadrature_points(FT, quadrature_style)
    DA = ClimaComms.array_type(topology)

    nfaces = length(bfaces)
    faces = Matrix{Int32}(undef, 2, nfaces)
    lg_host = Adapt.adapt(Array, Spaces.local_geometry_data(space))
    SG = Geometry.SurfaceGeometry{FT, Geometry.UVVector{FT}}
    sgeom = Array{SG}(undef, Nv, Nq, nfaces)

    contrib = Dict{NTuple{3, Int}, Vector{NTuple{3, Int32}}}()
    for (f, (elem⁻, face⁻)) in enumerate(bfaces)
        faces[:, f] .= (elem⁻, face⁻)
        for q in 1:Nq
            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            push!(
                get!(() -> NTuple{3, Int32}[], contrib, (elem⁻, i⁻, j⁻)),
                (Int32(f), Int32(1), Int32(q)),
            )
            for v in 1:Nv
                lg = slab(lg_host, v, elem⁻)[1, i⁻, j⁻, 1]
                sgeom[v, q, f] = compute_surface_geometry_extruded_2d(
                    lg,
                    w,
                    face⁻,
                    i⁻,
                    j⁻,
                )
            end
        end
    end

    return DGConnectivity(nfaces, faces, sgeom, contrib, DA)
end
