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
# `Vararg{Any, N}` with `N` a type parameter: the arguments are mostly
# forwarded unchanged, so Julia's Vararg heuristic would otherwise compile the
# chain unspecialized on them, heap-allocating the argument tuple at every call
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

# ===========================================================================
# Physics / flux library (moisture-0M DG branch) — additive, grafted onto the
# ts/mpi DG infrastructure above and the dg_fluxes.jl flux types. Uses the
# stable fn(normal, argvals..) and fn2pt(nvec_a, nvec_b, y_a, y_b) calling
# conventions preserved by that infra.
# ===========================================================================

"""
    RoeNumericalFlux(fluxfn, roe_avg_fn)

Evaluates the Roe numerical flux using `fluxfn` and Roe-averaging function `roe_avg_fn`.

The Roe flux computes a central flux plus an entropy-stable dissipation term based on
the characteristic decomposition of the jump in conserved variables.
"""
struct RoeNumericalFlux{F, A} <: AbstractNumericalFlux
    fluxfn::F
    roe_avg_fn::A
end

function (fn::RoeNumericalFlux)(normal, argvals⁻, argvals⁺)
    y⁻ = argvals⁻[1]
    y⁺ = argvals⁺[1]
    params⁻ = argvals⁻[2]
    params⁺ = argvals⁺[2]

    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    Favg = (F⁻ + F⁺) / 2

    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * normal

    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * normal

    λ = sqrt(params⁻.g)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)

    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    ρ̄ = sqrt(ρ⁻ * ρ⁺)
    ū = fn.roe_avg_fn(ρ⁻, ρ⁺, u⁻, u⁺)
    θ̄ = fn.roe_avg_fn(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c̄ = fn.roe_avg_fn(ρ⁻, ρ⁺, c⁻, c⁺)

    ūₙ = ū' * normal

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * normal

    c̄⁻² = 1 / c̄^2
    w1 = abs(ūₙ - c̄) * (Δp - ρ̄ * c̄ * Δuₙ) * 0.5 * c̄⁻²
    w2 = abs(ūₙ + c̄) * (Δp + ρ̄ * c̄ * Δuₙ) * 0.5 * c̄⁻²
    w3 = abs(ūₙ) * (Δρ - Δp * c̄⁻²)
    w4 = abs(ūₙ) * ρ̄
    w5 = abs(ūₙ) * (Δρθ - θ̄ * Δp * c̄⁻²)

    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρu =
        (w1 * (ū - c̄ * normal) + w2 * (ū + c̄ * normal) + w3 * ū + w4 * (Δu - Δuₙ * normal)) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ̄ + w5) * 0.5

    return (
        ρ = ((F⁻.ρ + F⁺.ρ) / 2)' * normal - fluxᵀn_ρ,
        ρu = ((F⁻.ρu + F⁺.ρu) / 2)' * normal - fluxᵀn_ρu,
        ρθ = ((F⁻.ρθ + F⁺.ρθ) / 2)' * normal - fluxᵀn_ρθ,
    )
end

# Kinetic energy of a velocity AxisVector (2D UV / UW plane or 1D U).
@inline _specific_ke(u::Geometry.UVVector) = (u.u^2 + u.v^2) / 2
@inline _specific_ke(u::Geometry.UWVector) = (u.u^2 + u.w^2) / 2
@inline _specific_ke(u::Geometry.UVector) = (u.u^2) / 2

# Tangential unit for Roe shear wave: rotate n̂ 90° CCW in the horizontal
# plane. 1D `UVector` normals (extruded x–z hybrid) have no in-plane shear.
@inline _roe_tangent(n::Geometry.UVVector) = typeof(n)(-n.v, n.u)
@inline _roe_tangent(::Geometry.UVector) = nothing
@inline _roe_tangent(::Geometry.UWVector) = nothing

"""
    ideal_gas_pressure(state, params)

Default pressure for `EntropyConservingFlux`: `p = (γ-1)(ρe - ρKE)`.
"""
function ideal_gas_pressure(state, params)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    return (params.γ - 1) * (ρe - ρ * _specific_ke(u))
end

"""
    EntropyConservingFlux(fluxfn, entropy_var_fn, roe_avg_fn[; pressure_fn, momentum_pressure_fn, roe_pressure_fn, sound_speed_fn])

Kennedy-Gruber kinetic energy preserving (KEP) interface flux with Roe entropy-stable
dissipation for compressible Euler equations, following Souza et al. (2023, JAMES, Eqs 40-42).

The central part uses arithmetic averages of primitive variables {ρ}, {u}, {p}, {e}, giving
the KEP property. The Roe dissipation uses a full characteristic decomposition of the Roe-averaged
Jacobian (4 waves in 2D: two acoustics, one entropy/contact, one shear).

- `fluxfn(state, params...)`: physical flux tensor F(U)
- `entropy_var_fn(state, params...)`: entropy variables v = ∂η/∂U (stored, not used in dissipation)
- `roe_avg_fn(ρ⁻, ρ⁺, var⁻, var⁺)`: Roe-averaging function, e.g. density-weighted average
- `pressure_fn(state, params...)`: thermodynamic pressure for enthalpy / energy (defaults to [`ideal_gas_pressure`](@ref))
- `momentum_pressure_fn(state, params...)`: pressure in the K-G momentum flux (defaults to `pressure_fn`)
- `roe_pressure_fn(state, params...)`: pressure in Roe wave amplitudes α₁, α₂, α₄ (defaults to `momentum_pressure_fn`, so stratified p′ formulations stay consistent with the volume flux)
- `sound_speed_fn(state, params...)`: optional Roe sound speed; if `nothing`, uses `√((γ-1)(H̃-KẼ))`
"""
struct EntropyConservingFlux{F, V, A, P, MP, RP, S} <: AbstractNumericalFlux
    fluxfn::F
    entropy_var_fn::V
    roe_avg_fn::A
    pressure_fn::P
    momentum_pressure_fn::MP
    roe_pressure_fn::RP
    sound_speed_fn::S

    function EntropyConservingFlux(
        fluxfn,
        entropy_var_fn,
        roe_avg_fn;
        pressure_fn = ideal_gas_pressure,
        momentum_pressure_fn = nothing,
        roe_pressure_fn = nothing,
        sound_speed_fn = nothing,
    )
        F, V, A, P = typeof.((fluxfn, entropy_var_fn, roe_avg_fn, pressure_fn))
        MP = momentum_pressure_fn === nothing ? pressure_fn : momentum_pressure_fn
        # Roe Δp must use the same pressure as the K-G / volume momentum flux (p′ for stratified).
        RP = roe_pressure_fn === nothing ? MP : roe_pressure_fn
        S = sound_speed_fn
        return new{F, V, A, P, typeof(MP), typeof(RP), typeof(S)}(
            fluxfn,
            entropy_var_fn,
            roe_avg_fn,
            pressure_fn,
            MP,
            RP,
            S,
        )
    end
end

# Positional `pressure_fn` for backward compatibility (e.g. Compressible Euler).
function EntropyConservingFlux(fluxfn, entropy_var_fn, roe_avg_fn, pressure_fn)
    return EntropyConservingFlux(
        fluxfn,
        entropy_var_fn,
        roe_avg_fn;
        pressure_fn,
    )
end

function (fn::EntropyConservingFlux)(normal, argvals⁻, argvals⁺)
    y⁻ = argvals⁻[1]
    y⁺ = argvals⁺[1]
    params = argvals⁻[2]

    ρ⁻, ρu⁻, ρe⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρe
    ρ⁺, ρu⁺, ρe⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρe

    u⁻ = ρu⁻ / ρ⁻
    u⁺ = ρu⁺ / ρ⁺
    γ = params.γ

    KE⁻ = _specific_ke(u⁻)
    KE⁺ = _specific_ke(u⁺)
    p⁻ = fn.pressure_fn(argvals⁻...)
    p⁺ = fn.pressure_fn(argvals⁺...)
    pm⁻ = fn.momentum_pressure_fn(argvals⁻...)
    pm⁺ = fn.momentum_pressure_fn(argvals⁺...)
    p_roe⁻ = fn.roe_pressure_fn(argvals⁻...)
    p_roe⁺ = fn.roe_pressure_fn(argvals⁺...)

    # Kennedy-Gruber KEP interface flux: arithmetic averages (Souza et al. 2023,
    # JAMES, Eqs 40–42 / App. A). Uses {ρ}{u}{p}{e} for the central flux.
    ρ̄ = (ρ⁻ + ρ⁺) / 2
    ū = (u⁻ + u⁺) / 2
    p̄ = (p⁻ + p⁺) / 2
    p̄m = (pm⁻ + pm⁺) / 2
    ē = (ρe⁻ / ρ⁻ + ρe⁺ / ρ⁺) / 2  # arithmetic mean of specific total energy

    Fc_ρ = (ρ̄ * ū)' * normal
    Fc_ρu = (ρ̄ * (ū ⊗ ū) + p̄m * I)' * normal
    Fc_ρe = (ū * (ρ̄ * ē + p̄))' * normal  # {u}({ρ}{e} + {p})

    # Roe-averaged state for compressible Euler (Roe 1981)
    # Guard non-positive densities at the face (does not floor prognostics).
    pos = ρ⁻ > 0 && ρ⁺ > 0
    ρ̃ = pos ? sqrt(ρ⁻ * ρ⁺) : abs(ρ̄)
    ũ = pos ? fn.roe_avg_fn(ρ⁻, ρ⁺, u⁻, u⁺) : ū
    H⁻ = (ρe⁻ + p⁻) / ρ⁻  # specific total enthalpy
    H⁺ = (ρe⁺ + p⁺) / ρ⁺
    H̃ = pos ? fn.roe_avg_fn(ρ⁻, ρ⁺, H⁻, H⁺) : (H⁻ + H⁺) / 2
    KE_tilde = _specific_ke(ũ)
    c̃ = if fn.sound_speed_fn === nothing
        # Fall back to thermodynamic Roe c only when H̃ > KẼ.
        ΔH = H̃ - KE_tilde
        ΔH > 0 ? sqrt((γ - 1) * ΔH) : FT_zero(ΔH)
    else
        c⁻ = fn.sound_speed_fn(argvals⁻...)
        c⁺ = fn.sound_speed_fn(argvals⁺...)
        pos ? fn.roe_avg_fn(ρ⁻, ρ⁺, c⁻, c⁺) : max(c⁻, c⁺)
    end

    # Normal (and tangential, in 2D) directions.
    # Extruded 1D faces use `UVector` normals → no shear wave (Souza 1D Euler).
    ũₙ = ũ' * normal
    tang = _roe_tangent(normal)

    Δρ = ρ⁺ - ρ⁻
    Δu = u⁺ - u⁻
    Δuₙ = Δu' * normal
    # p′ jump for Roe amplitudes when momentum_pressure_fn = p′ (stratified)
    Δp = p_roe⁺ - p_roe⁻

    c̃⁻² = 1 / c̃^2
    α₁ = (Δp - ρ̃ * c̃ * Δuₙ) * 0.5 * c̃⁻²   # left-running acoustic
    α₂ = Δρ - Δp * c̃⁻²                        # entropy / contact
    α₄ = (Δp + ρ̃ * c̃ * Δuₙ) * 0.5 * c̃⁻²   # right-running acoustic

    λ₁ = abs(ũₙ - c̃)
    λ₂ = abs(ũₙ)
    λ₄ = abs(ũₙ + c̃)

    diss_ρ = λ₁ * α₁ + λ₂ * α₂ + λ₄ * α₄
    diss_ρu =
        (λ₁ * α₁) * (ũ - c̃ * normal) +
        (λ₂ * α₂) * ũ +
        (λ₄ * α₄) * (ũ + c̃ * normal)
    diss_ρe =
        λ₁ * α₁ * (H̃ - c̃ * ũₙ) +
        λ₂ * α₂ * KE_tilde +
        λ₄ * α₄ * (H̃ + c̃ * ũₙ)

    if tang !== nothing
        Δuₜ = Δu' * tang
        ũₜ = ũ' * tang
        α₃ = ρ̃ * Δuₜ                               # shear / vorticity
        diss_ρu = diss_ρu + (λ₂ * α₃) * tang
        diss_ρe = diss_ρe + λ₂ * α₃ * ũₜ
    end

    base = (
        ρ = Fc_ρ - diss_ρ / 2,
        ρu = Fc_ρu - diss_ρu / 2,
        ρe = Fc_ρe - diss_ρe / 2,
    )
    return merge(base, _passive_tracer_fluxes(y⁻, y⁺, ū, normal, λ₂))
end

@inline FT_zero(x) = zero(typeof(x))

# Handle passive tracer fields (ρθ) not part of the Euler entropy structure.
function _passive_tracer_fluxes(y⁻, y⁺, ū, normal, λ₂)
    nt⁻, nt⁺ = unwrap(y⁻), unwrap(y⁺)
    if !hasfield(typeof(nt⁻), :ρθ)
        return NamedTuple()
    end
    # Central advection + upwind dissipation for the passive tracer ρθ.
    Fc_ρθ = ((nt⁻.ρθ + nt⁺.ρθ) / 2 * ū)' * normal
    diss_ρθ = λ₂ * (nt⁺.ρθ - nt⁻.ρθ)
    return (ρθ = Fc_ρθ - diss_ρθ / 2,)
end

# ---------------------------------------------------------------------------
# Two-point (volume) and interface fluxes
# ---------------------------------------------------------------------------

"""
    kennedy_gruber_scalars_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux for the flux-form (ρ, ρe) subsystem (Souza et
al. 2023, JAMES, Eqs. 39 & 41): ``F_ρ = \\{ρ\\}\\{ũ\\}``,
``F_{ρe} = \\{ũ\\}(\\{ρ\\}\\{e\\} + \\{p\\})``, with `e` the specific total
energy and ``\\{ũ\\} = \\{u ⋅ nvec\\}`` the average of the **contravariant
nodal fluxes** (each node's velocity contracted with its own metric vector —
see [`add_flux_differencing_divergence!`](@ref) for why). Symmetric,
consistent, jointly linear in `(nvec_a, nvec_b)`.

State fields required: `ρ`, `ρe`, `e`, `p`, and `uv` (velocity in the local
orthonormal horizontal frame).
"""
function kennedy_gruber_scalars_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    return (ρ = ρ̄ * ūn, ρe = (ρ̄ * ē + p̄) * ūn)
end

"""
    kennedy_gruber_rusanov_scalars(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe) subsystem: [`kennedy_gruber_scalars_flux`](@ref)
as the central part plus a Rusanov penalty scaled by the state field `λ`
(the paper's interface choice, Souza et al. 2023).
"""
function kennedy_gruber_rusanov_scalars(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_scalars_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
    )
end

"""
    kennedy_gruber_tracer_flux(nvec_a, nvec_b, y_a, y_b)

Two-point flux for a passive tracer ``ρq`` advected by the SAME Kennedy-Gruber
mass flux as continuity: ``F_{ρq} = \\{ρ\\}\\{ũ\\}\\{q\\}`` (the mass flux
``\\{ρ\\}\\{ũ\\}`` times the arithmetic-mean specific tracer ``\\{q\\}``). This
is free-stream-preserving for the tracer — with ``q`` uniform, ``F_{ρq} = q F_ρ``
so the tracer equation reduces to ``q``×continuity and a constant ``q`` stays
constant. State fields required: `ρ`, `uv`, `q` (specific tracer, e.g. total
specific humidity `q_tot`).
"""
function kennedy_gruber_tracer_flux(nvec_a, nvec_b, y_a, y_b)
    Fρ = ((y_a.ρ + y_b.ρ) / 2) * ((y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2)
    q̄ = (y_a.q + y_b.q) / 2
    return (ρq = Fρ * q̄,)
end

"""
    kennedy_gruber_rusanov_tracer(normal, argvals⁻, argvals⁺)

Interface flux for a passive tracer: [`kennedy_gruber_tracer_flux`](@ref) central
part plus a Rusanov penalty on the conserved tracer jump ``⟦ρq⟧`` scaled by the
state field `λ`. State fields: `ρ`, `uv`, `q`, `λ` (and `ρq = ρ·q`).
"""
function kennedy_gruber_rusanov_tracer(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_tracer_flux(normal, normal, y⁻, y⁺)
    return (ρq = F.ρq - λ / 2 * (y⁺.ρ * y⁺.q - y⁻.ρ * y⁻.q),)
end

"""
    kennedy_gruber_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux for the full (ρ, ρe, ρu⃗) system with momentum
carried in GLOBAL CARTESIAN components (Souza et al. 2023): the basis is
constant, so component-wise flux differencing retains the KEP property with
no curvature source terms. Contravariant nodal fluxes are averaged (each
node's own metric vector).

State fields required: `ρ`, `e`, `p`, `uv` (velocity, local orthonormal
horizontal frame), `u1`, `u2`, `u3` (Cartesian velocity components), and
`E1`, `E2`, `E3` (the tangential projections of the Cartesian unit vectors
ê₁, ê₂, ê₃, each as a `UVVector` — position-dependent on the sphere but
state-independent). The pressure flux for component ``c`` is
``\\{p\\}\\{ê_c ⋅ nvec\\}``.
"""
function kennedy_gruber_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    # Momentum pressure: `pm` = p (full conservative) or p' = p − p_ref
    # (stratified conservative, well-balanced over topography). Energy keeps
    # the full thermodynamic p in the enthalpy flux.
    p̄m = (y_a.pm + y_b.pm) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    return (
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn + p̄m * Ē1n,
        ρu2 = ρ̄ * ū2 * ūn + p̄m * Ē2n,
        ρu3 = ρ̄ * ū3 * ūn + p̄m * Ē3n,
    )
end

"""
    ln_mean(x, y)

Numerically-stable logarithmic mean ``(x-y)/(\\log x - \\log y)`` (Ismail & Roe
2009): switches to the convergent Taylor series in ``f^2=((x-y)/(x+y))^2`` when
``x≈y`` to avoid the ``0/0`` cancellation. The log mean is the building block of
entropy-conservative fluxes (it is what makes ``⟦w⟧·F^\\# = ⟦ψ⟧`` hold exactly).
"""
@inline function ln_mean(x, y)
    ε = oftype(x, 1e-4)
    f² = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)  # ((x−y)/(x+y))²
    # Build the series coefficients at the working precision: bare `2 / 3` etc.
    # are Float64 literals, which promote the Taylor branch to Float64 while the
    # `log` branch stays Float32. The resulting Union{Float32,Float64} boxes
    # every flux NamedTuple built on top of this and the GPU kernels fail to
    # compile (dynamic NamedTuple construction, gpu_gc_pool_alloc).
    c1, c2, c3 = oftype(f², 2 // 3), oftype(f², 2 // 5), oftype(f², 2 // 7)
    return f² < ε ?
           (x + y) / (2 + f² * (c1 + f² * (c2 + f² * c3))) :
           (y - x) / log(y / x)
end

"""
    ranocha_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Ranocha (2018, 2020) two-point flux for the (ρ, ρe, ρu⃗) system in GLOBAL
CARTESIAN momentum components — the *entropy-conservative* counterpart of
[`kennedy_gruber_cartesian_flux`](@ref). Unlike Kennedy-Gruber (which is only
kinetic-energy- and pressure-equilibrium-preserving), the Ranocha flux is
SIMULTANEOUSLY entropy-conservative (Tadmor `⟦w⟧·F# = ⟦ψ⟧`), kinetic-energy-
preserving, and pressure-equilibrium-preserving, so — paired with an
entropy-dissipative interface — it yields a discrete entropy inequality that
Kennedy-Gruber cannot.

It differs from KG in three places: the mass flux uses the logarithmic mean
``ρ^{ln}`` instead of ``ρ̄``; the internal energy uses ``1/((γ-1)(ρ/p)^{ln})``;
and the pressure-work uses the cross term ``½(p_a u_{n,b}+p_b u_{n,a})`` rather
than ``p̄ ū_n``. The kinetic part is the KEP cross term ``½\\,u_a·u_b``. The
geopotential (``Φ = e - e_{int} - K``, single-valued at a shared node, varying
horizontally only over terrain) is advected as a passive potential ``ρ^{ln}
ū_n\\,\\{Φ\\}``. Momentum pressure uses `pm` (= p, or p' for the stratified /
well-balanced split) exactly as KG, so it drops into the same volume-flux slot
and inherits the same reference-deviation well-balancedness. Consistency check:
for `y_a == y_b` it collapses to the physical fluxes ``ρu_n``,
``(ρe+p)u_n``, ``ρu_c u_n + pm\\,ê_c·n``. Same state fields as
[`kennedy_gruber_cartesian_flux`](@ref).
"""
function ranocha_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    γd = oftype(y_a.ρ, γ_dry)
    ρln = ln_mean(y_a.ρ, y_b.ρ)
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    mn = ρln * ūn                                   # entropy-consistent mass flux
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    p̄m = (y_a.pm + y_b.pm) / 2
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    # internal energy: 1/((γ−1)(ρ/p)^ln); KEP kinetic cross term; pressure work
    e_int = 1 / (ln_mean(y_a.ρ / y_a.p, y_b.ρ / y_b.p) * (γd - 1))
    K̃ = (y_a.u1 * y_b.u1 + y_a.u2 * y_b.u2 + y_a.u3 * y_b.u3) / 2
    una = y_a.uv' * nvec_a
    unb = y_b.uv' * nvec_b
    pv = (y_a.p * unb + y_b.p * una) / 2            # ½(p_a u_{n,b}+p_b u_{n,a})
    # geopotential per node (Φ = e − e_int − K), advected as a passive potential
    Φa = y_a.e - y_a.p / ((γd - 1) * y_a.ρ) - (y_a.u1^2 + y_a.u2^2 + y_a.u3^2) / 2
    Φb = y_b.e - y_b.p / ((γd - 1) * y_b.ρ) - (y_b.u1^2 + y_b.u2^2 + y_b.u3^2) / 2
    Φ̄ = (Φa + Φb) / 2
    return (
        ρ = mn,
        ρe = mn * (K̃ + e_int + Φ̄) + pv,
        ρu1 = mn * ū1 + p̄m * Ē1n,
        ρu2 = mn * ū2 + p̄m * Ē2n,
        ρu3 = mn * ū3 + p̄m * Ē3n,
    )
end

"""
    waruszewski_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Waruszewski et al. (2022, JCP 468:111507) entropy-conservative + WELL-BALANCED
two-point flux for the (ρ, ρe, ρu⃗) system WITH GRAVITY, in global Cartesian
momentum components. This is the only flux here that is EC *and* machine-precision
well-balanced over terrain SIMULTANEOUSLY: the geopotential is handled by a
non-conservative fluctuation term ``½ρ̂⟦φ⟧`` in the momentum flux — NOT by a
reference split. It satisfies the generalized (non-conservative) Tadmor condition
``β⁻·D(a;b) − β⁺·D(b;a) = ⟦u_kη⟧`` with the geopotential-augmented entropy
variables (β₁ carries the ``+2φb`` term; see [`entropy_variables`](@ref)).

Differs from Ranocha: the EC pressure is Chandrashekar's ``p* = {{ρ}}/(2{{b}})``,
``b = ρ/(2p)`` (not ``{{p}}``); the internal energy uses the log-mean of ``b``;
and the momentum pressure slot is ``p* + ½ρ̂⟦φ⟧`` with ``ρ̂ = {{b}}{{ρ}}_log/b⁻``
(NON-symmetric — uses the own/self state ``b⁻``, which is well-defined here since
the kernel passes the self node first). Verified: at ``y_a=y_b`` it reduces to the
physical fluxes, and the Tadmor residual over a geopotential jump is ~1e-15.

Hybrid adaptation: the horizontal DG advects only the horizontal momentum, so the
vertical kinetic energy ``w_c²/2`` rides as a passive potential bundled with ``φ``
in ``e*`` (via ``Ψ = e − e_int − K_h``), while the gravity fluctuation uses the
geopotential ``φ`` alone (state field `φ`). State fields: `ρ`, `e`, `p`, `uv`,
`u1`,`u2`,`u3`, `E1`,`E2`,`E3`, `φ`.
"""
function waruszewski_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    γd = oftype(y_a.ρ, γ_dry)
    ba = y_a.ρ / (2 * y_a.p)                         # inverse temperature b⁻ (self)
    bb = y_b.ρ / (2 * y_b.p)
    ρln = ln_mean(y_a.ρ, y_b.ρ)
    bln = ln_mean(ba, bb)
    b̄ = (ba + bb) / 2
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    mn = ρln * ūn                                    # (ρuₖ)* = ρ^ln {{u}}
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    p_star = ρ̄ / (2 * b̄)                             # Chandrashekar p* = {{ρ}}/2{{b}}
    ρ̂ = b̄ * ρln / ba                                # NON-symmetric (self b⁻)
    jφ = y_b.φ - y_a.φ                               # ⟦φ⟧
    pgrav = p_star + ρ̂ * jφ / 2                      # momentum pressure slot
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    # internal energy log-mean 1/(2(γ−1)b^ln); horizontal KEP kinetic cross term;
    # passive potential Ψ = φ + w_c²/2 = e − e_int − K_h (advected like {{φ}}).
    e_int = 1 / (2 * (γd - 1) * bln)
    K̃ = (y_a.u1 * y_b.u1 + y_a.u2 * y_b.u2 + y_a.u3 * y_b.u3) / 2
    Ψa = y_a.e - y_a.p / ((γd - 1) * y_a.ρ) - (y_a.u1^2 + y_a.u2^2 + y_a.u3^2) / 2
    Ψb = y_b.e - y_b.p / ((γd - 1) * y_b.ρ) - (y_b.u1^2 + y_b.u2^2 + y_b.u3^2) / 2
    e_star = e_int + (Ψa + Ψb) / 2 + K̃
    return (
        ρ = mn,
        ρe = e_star * mn + ūn * p_star,
        ρu1 = mn * ū1 + pgrav * Ē1n,
        ρu2 = mn * ū2 + pgrav * Ē2n,
        ρu3 = mn * ū3 + pgrav * Ē3n,
    )
end

"""
    kennedy_gruber_rusanov_cartesian(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe, ρu⃗-Cartesian) system:
[`kennedy_gruber_cartesian_flux`](@ref) as the central part plus a Rusanov
penalty scaled by the state field `λ` (jumps of the conserved variables;
momentum jumps via `ρ * u_c`). Additional state fields: `ρe`, `λ`.
"""
function kennedy_gruber_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
        ρu1 = F.ρu1 - λ / 2 * (y⁺.ρ * y⁺.u1 - y⁻.ρ * y⁻.u1),
        ρu2 = F.ρu2 - λ / 2 * (y⁺.ρ * y⁺.u2 - y⁻.ρ * y⁻.u2),
        ρu3 = F.ρu3 - λ / 2 * (y⁺.ρ * y⁺.u3 - y⁻.ρ * y⁻.u3),
    )
end

"""
    kennedy_gruber_roe_cartesian(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe, ρu⃗-Cartesian) system:
[`kennedy_gruber_cartesian_flux`](@ref) as the central part plus ROE-TYPE
wave-selective dissipation (Souza et al. 2023 interface choice): acoustic
waves are damped at ``|û_n ± ĉ|`` but entropy and shear jumps at
``max(|û_n|, ĉ/20)`` — so stationary balanced structure (contact/shear
jumps with ``u_n ≈ 0``) receives ~5% of Rusanov's uniform ``|u| + c``
dissipation (the Harten-type floor is required: see inline comment).
The energy eigen-component uses ``B = Ĥ - ĉ²/(γ-1)``, which absorbs the
geopotential and vertical-kinetic contributions of ``ρe`` without needing
them separately (Φ is single-valued at the face). Same state fields as
[`kennedy_gruber_rusanov_cartesian`](@ref); requires `γ` jumps consistent
with `p`/`e` (dry ideal gas).
"""
function kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    # keep the working precision of the state (Float32 fields stay Float32)
    γd = oftype(y⁻.ρ, γ_dry)
    # face normal in Cartesian components (E_c single-valued at the node)
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    # Roe-averaged state
    s⁻ = sqrt(y⁻.ρ)
    s⁺ = sqrt(y⁺.ρ)
    ρ̂ = s⁻ * s⁺
    a⁻ = s⁻ / (s⁻ + s⁺)
    a⁺ = 1 - a⁻
    û1 = a⁻ * y⁻.u1 + a⁺ * y⁺.u1
    û2 = a⁻ * y⁻.u2 + a⁺ * y⁺.u2
    û3 = a⁻ * y⁻.u3 + a⁺ * y⁺.u3
    Ĥ = a⁻ * (y⁻.e + y⁻.p / y⁻.ρ) + a⁺ * (y⁺.e + y⁺.p / y⁺.ρ)
    ĉ = a⁻ * sqrt(γd * y⁻.p / y⁻.ρ) + a⁺ * sqrt(γd * y⁺.p / y⁺.ρ)
    ûn = û1 * n1 + û2 * n2 + û3 * n3
    # jumps and wave amplitudes. The pressure jump uses the momentum pressure
    # `pm` (= p for full conservative, = p' for stratified) so the acoustic
    # amplitudes vanish at rest even over topography. (The entropy amplitude α₀
    # still uses the full Δρ, so stratified Roe leaves an O(Δρ_ref) contact-wave
    # residual over terrain — stable, not machine-precision; LMARS avoids it.)
    Δρ = y⁺.ρ - y⁻.ρ
    Δp = y⁺.pm - y⁻.pm
    Δu1 = y⁺.u1 - y⁻.u1
    Δu2 = y⁺.u2 - y⁻.u2
    Δu3 = y⁺.u3 - y⁻.u3
    Δun = Δu1 * n1 + Δu2 * n2 + Δu3 * n3
    α₊ = (Δp + ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₋ = (Δp - ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₀ = Δρ - Δp / ĉ^2
    s₊ = abs(ûn + ĉ)
    s₋ = abs(ûn - ĉ)
    # Harten-type entropy floor on the contact/shear speed: pure |û_n|
    # leaves density jumps in near-stagnant columns (e.g. the model top)
    # undamped, and the min-ρ cell can drain unchecked (observed: secular
    # min-ρ collapse from day ~2.3 of a perturbed baroclinic wave at
    # zelem = 30). ε = 0.05 retains 5% of the Rusanov ρ-jump dissipation
    # while keeping the spurious forcing of balanced jets ~20× below
    # Rusanov. The acoustic speeds need no floor (|û_n| ≪ ĉ here).
    s₀ = max(abs(ûn), ĉ / 20)
    Δut1 = Δu1 - Δun * n1
    Δut2 = Δu2 - Δun * n2
    Δut3 = Δu3 - Δun * n3
    B = Ĥ - ĉ^2 / (γd - 1)
    Dρ = s₊ * α₊ + s₋ * α₋ + s₀ * α₀
    Dρu1 =
        s₊ * α₊ * (û1 + ĉ * n1) + s₋ * α₋ * (û1 - ĉ * n1) +
        s₀ * (α₀ * û1 + ρ̂ * Δut1)
    Dρu2 =
        s₊ * α₊ * (û2 + ĉ * n2) + s₋ * α₋ * (û2 - ĉ * n2) +
        s₀ * (α₀ * û2 + ρ̂ * Δut2)
    Dρu3 =
        s₊ * α₊ * (û3 + ĉ * n3) + s₋ * α₋ * (û3 - ĉ * n3) +
        s₀ * (α₀ * û3 + ρ̂ * Δut3)
    Dρe =
        s₊ * α₊ * (Ĥ + ĉ * ûn) + s₋ * α₋ * (Ĥ - ĉ * ûn) +
        s₀ * (α₀ * B + ρ̂ * (û1 * Δut1 + û2 * Δut2 + û3 * Δut3))
    return (
        ρ = F.ρ - Dρ / 2,
        ρe = F.ρe - Dρe / 2,
        ρu1 = F.ρu1 - Dρu1 / 2,
        ρu2 = F.ρu2 - Dρu2 / 2,
        ρu3 = F.ρu3 - Dρu3 / 2,
    )
end

"""
    ranocha_rusanov_cartesian(normal, argvals⁻, argvals⁺)
    ranocha_roe_cartesian(normal, argvals⁻, argvals⁺)

Entropy-stable interface fluxes: the entropy-conservative
[`ranocha_cartesian_flux`](@ref) central part plus the same Rusanov / Roe
dissipation used by the Kennedy-Gruber interfaces. The dissipation is recovered
as ``(F_{diss} - F_{KG,central})`` (a cheap extra KG eval) and added to the
Ranocha central flux, so the tested wave-selective penalties are reused verbatim
while the volume/interface central pair is now entropy-conservative. Paired with
[`ranocha_cartesian_flux`](@ref) as the volume flux this gives an EC-volume +
dissipative-interface scheme — the ingredient Kennedy-Gruber lacks for a discrete
entropy inequality. (The dissipation is in conserved, not entropy, variables, so
this is entropy-stable in the sense of an EC volume flux + a positive dissipation,
not a certified entropy-variable dissipation matrix.)
"""
function ranocha_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    Fr = ranocha_cartesian_flux(normal, normal, y⁻, y⁺)
    Fkg = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    Fd = kennedy_gruber_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    return (
        ρ = Fr.ρ + (Fd.ρ - Fkg.ρ),
        ρe = Fr.ρe + (Fd.ρe - Fkg.ρe),
        ρu1 = Fr.ρu1 + (Fd.ρu1 - Fkg.ρu1),
        ρu2 = Fr.ρu2 + (Fd.ρu2 - Fkg.ρu2),
        ρu3 = Fr.ρu3 + (Fd.ρu3 - Fkg.ρu3),
    )
end

function ranocha_roe_cartesian(normal, (y⁻,), (y⁺,))
    Fr = ranocha_cartesian_flux(normal, normal, y⁻, y⁺)
    Fkg = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    Fd = kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    return (
        ρ = Fr.ρ + (Fd.ρ - Fkg.ρ),
        ρe = Fr.ρe + (Fd.ρe - Fkg.ρe),
        ρu1 = Fr.ρu1 + (Fd.ρu1 - Fkg.ρu1),
        ρu2 = Fr.ρu2 + (Fd.ρu2 - Fkg.ρu2),
        ρu3 = Fr.ρu3 + (Fd.ρu3 - Fkg.ρu3),
    )
end

"""
    waruszewski_rusanov_cartesian(normal, argvals⁻, argvals⁺)
    waruszewski_roe_cartesian(normal, argvals⁻, argvals⁺)
    waruszewski_es_cartesian(normal, argvals⁻, argvals⁺)

Interface fluxes pairing the well-balanced entropy-conservative
[`waruszewski_cartesian_flux`](@ref) central part with Rusanov / Roe / entropy-
variable ([`entropy_stable_dissipation`](@ref)) dissipation. The dissipation is
recovered as ``(F_{diss} − F_{KG,central})`` (a cheap KG eval) so the tested
penalties are reused verbatim; the WB-EC central flux carries the pressure and
gravity. With the entropy-variable (`es`) dissipation this is the genuinely
entropy-stable AND well-balanced-over-terrain scheme.
"""
function waruszewski_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    Fw = waruszewski_cartesian_flux(normal, normal, y⁻, y⁺)
    Fkg = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    Fd = kennedy_gruber_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    return (
        ρ = Fw.ρ + (Fd.ρ - Fkg.ρ),
        ρe = Fw.ρe + (Fd.ρe - Fkg.ρe),
        ρu1 = Fw.ρu1 + (Fd.ρu1 - Fkg.ρu1),
        ρu2 = Fw.ρu2 + (Fd.ρu2 - Fkg.ρu2),
        ρu3 = Fw.ρu3 + (Fd.ρu3 - Fkg.ρu3),
    )
end

function waruszewski_roe_cartesian(normal, (y⁻,), (y⁺,))
    Fw = waruszewski_cartesian_flux(normal, normal, y⁻, y⁺)
    Fkg = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    Fd = kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    return (
        ρ = Fw.ρ + (Fd.ρ - Fkg.ρ),
        ρe = Fw.ρe + (Fd.ρe - Fkg.ρe),
        ρu1 = Fw.ρu1 + (Fd.ρu1 - Fkg.ρu1),
        ρu2 = Fw.ρu2 + (Fd.ρu2 - Fkg.ρu2),
        ρu3 = Fw.ρu3 + (Fd.ρu3 - Fkg.ρu3),
    )
end

function waruszewski_es_cartesian(normal, (y⁻,), (y⁺,))
    Fw = waruszewski_cartesian_flux(normal, normal, y⁻, y⁺)
    D = entropy_stable_dissipation(y⁻, y⁺)
    return (
        ρ = Fw.ρ - D.ρ,
        ρe = Fw.ρe - D.ρe,
        ρu1 = Fw.ρu1 - D.ρu1,
        ρu2 = Fw.ρu2 - D.ρu2,
        ρu3 = Fw.ρu3 - D.ρu3,
    )
end
# dry-air ratio of specific heats used by the Roe linearization
const γ_dry = 7 / 5

"""
    entropy_variables(ρ, u1, u2, u3, p)

Entropy variables ``w = ∂S/∂U`` for the ideal-gas Euler system with the
mathematical (convex) entropy ``S = -ρs/(γ-1)``, ``s = \\log p - γ\\log ρ``
(thermal frame). With ``β = ρ/(2p)``,

    w = ((γ-s)/(γ-1) - β|u|²,  2βu1,  2βu2,  2βu3,  -2β).

Additive constants in `s` drop under the jump `⟦w⟧`, so they are irrelevant to
the dissipation built from these.
"""
@inline function entropy_variables(ρ, u1, u2, u3, p)
    γd = oftype(ρ, γ_dry)
    β = ρ / (2 * p)
    s = log(p) - γd * log(ρ)
    wρ = (γd - s) / (γd - 1) - β * (u1^2 + u2^2 + u3^2)
    return (wρ, 2 * β * u1, 2 * β * u2, 2 * β * u3, -2 * β)
end

"""
    entropy_stable_dissipation(y⁻, y⁺)

Lax-Friedrichs dissipation in ENTROPY variables, ``½ λ Ĥ ⟦w⟧``, where
``Ĥ = ∂U/∂w`` is the (symmetric positive-definite) entropy Jacobian at the
arithmetic-mean state and ``λ = \\max(|u|+c)``. Because `Ĥ` is SPD,
``⟦w⟧·(Ĥ⟦w⟧) ≥ 0``, so subtracting this from ANY entropy-conservative
([`ranocha_cartesian_flux`](@ref)) or kinetic-energy-preserving
([`kennedy_gruber_cartesian_flux`](@ref)) central flux gives a discrete entropy
inequality (entropy stability) — the guarantee that conserved-variable
Rusanov/Roe penalties do not provide. To leading order `Ĥ⟦w⟧ = ⟦U⟧`, so this is
an entropy-consistent Rusanov. The geopotential (single-valued at the shared
node, `⟦Φ⟧ = 0`) is handled by forming `Ĥ⟦w⟧` in the thermal frame and shifting
the energy component by `Φ·(mass dissipation)` — an identity-preserving change of
variables. Returns the conserved-variable dissipation `(ρ, ρe, ρu1, ρu2, ρu3)`.
The `Ĥ = ∂U/∂w` form is verified numerically (symmetry, SPD, `Ĥ·(∂w/∂U)=I`).
"""
@inline function entropy_stable_dissipation(y⁻, y⁺)
    γd = oftype(y⁻.ρ, γ_dry)
    w⁻ = entropy_variables(y⁻.ρ, y⁻.u1, y⁻.u2, y⁻.u3, y⁻.p)
    w⁺ = entropy_variables(y⁺.ρ, y⁺.u1, y⁺.u2, y⁺.u3, y⁺.p)
    v1 = w⁺[1] - w⁻[1]
    v2 = w⁺[2] - w⁻[2]
    v3 = w⁺[3] - w⁻[3]
    v4 = w⁺[4] - w⁻[4]
    v5 = w⁺[5] - w⁻[5]
    # arithmetic-mean state for Ĥ = ∂U/∂w
    ρ = (y⁻.ρ + y⁺.ρ) / 2
    u1 = (y⁻.u1 + y⁺.u1) / 2
    u2 = (y⁻.u2 + y⁺.u2) / 2
    u3 = (y⁻.u3 + y⁺.u3) / 2
    p = (y⁻.p + y⁺.p) / 2
    k = (u1^2 + u2^2 + u3^2) / 2
    E = p / ((γd - 1) * ρ) + k            # thermal total energy per mass
    H = E + p / ρ                         # thermal enthalpy per mass
    c2 = γd * p / ρ
    # Ĥ v (thermal frame), Ĥ = ∂U/∂w SPD
    HvR = ρ * v1 + ρ * u1 * v2 + ρ * u2 * v3 + ρ * u3 * v4 + ρ * E * v5
    Hv1 =
        ρ * u1 * v1 + (ρ * u1^2 + p) * v2 + ρ * u1 * u2 * v3 +
        ρ * u1 * u3 * v4 + ρ * u1 * H * v5
    Hv2 =
        ρ * u2 * v1 + ρ * u1 * u2 * v2 + (ρ * u2^2 + p) * v3 +
        ρ * u2 * u3 * v4 + ρ * u2 * H * v5
    Hv3 =
        ρ * u3 * v1 + ρ * u1 * u3 * v2 + ρ * u2 * u3 * v3 +
        (ρ * u3^2 + p) * v4 + ρ * u3 * H * v5
    HvE =
        ρ * E * v1 + ρ * u1 * H * v2 + ρ * u2 * H * v3 + ρ * u3 * H * v4 +
        (ρ * H^2 - c2 * p / (γd - 1)) * v5
    λ = max(y⁻.λ, y⁺.λ)
    # geopotential (single-valued at the node ⇒ Φ⁻ = Φ⁺); shift thermal→total
    Φ = y⁻.e - y⁻.p / ((γd - 1) * y⁻.ρ) - (y⁻.u1^2 + y⁻.u2^2 + y⁻.u3^2) / 2
    half = λ / 2
    Dρ = half * HvR
    return (
        ρ = Dρ,
        ρe = half * HvE + Φ * Dρ,
        ρu1 = half * Hv1,
        ρu2 = half * Hv2,
        ρu3 = half * Hv3,
    )
end

"""
    kennedy_gruber_es_cartesian(normal, argvals⁻, argvals⁺)
    ranocha_es_cartesian(normal, argvals⁻, argvals⁺)

Entropy-stable interface fluxes: a central two-point flux (Kennedy-Gruber or
Ranocha) minus [`entropy_stable_dissipation`](@ref) (dissipation in the entropy
variables). With the Ranocha EC central flux this is a genuinely entropy-stable
scheme (discrete `dS/dt ≤` boundary); with the KG (KEP, not EC) central flux the
dissipation is still entropy-decreasing but the KG volume error remains. Both
share the identical dissipation, so the penalty is decoupled from the choice of
central flux.
"""
function kennedy_gruber_es_cartesian(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    D = entropy_stable_dissipation(y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ,
        ρe = F.ρe - D.ρe,
        ρu1 = F.ρu1 - D.ρu1,
        ρu2 = F.ρu2 - D.ρu2,
        ρu3 = F.ρu3 - D.ρu3,
    )
end

function ranocha_es_cartesian(normal, (y⁻,), (y⁺,))
    F = ranocha_cartesian_flux(normal, normal, y⁻, y⁺)
    D = entropy_stable_dissipation(y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ,
        ρe = F.ρe - D.ρe,
        ρu1 = F.ρu1 - D.ρu1,
        ρu2 = F.ρu2 - D.ρu2,
        ρu3 = F.ρu3 - D.ρu3,
    )
end

"""
    lmars_cartesian(normal, argvals⁻, argvals⁺)

Low-Mach Approximate Riemann Solver (LMARS; Chen et al. 2013, the FV3 flux) for
the conservative (ρ, ρe, ρu⃗-Cartesian) system. A two-wave acoustic Riemann
solve gives an interface normal velocity and pressure from the reference
impedance ``C = ρ̄ ĉ`` (ĉ = mean of a state-provided, floorable sound speed
`c`):

    u* = ½(uₙ⁻+uₙ⁺) − (p⁺−p⁻)/(2C),   p* = ½(p⁻+p⁺) − ½C(uₙ⁺−uₙ⁻),

then every advected quantity is upwinded at `u*` (flow speed, NOT `|u|+c`), so
acoustic dissipation scales with the impedance `C` while advective dissipation
scales with `|u*|` — wave-selective like Roe, but with no eigen-decomposition
and no `sqrt(γp/ρ)` (robust where `p` dips negative). State fields: `ρ`, `ρe`,
`p`, `u1`,`u2`,`u3` (Cartesian velocity), `E1`,`E2`,`E3` (Cartesian projections
of the face normal, single-valued at the node), and `c` (sound speed). It is a
complete numerical flux (no separate central+penalty), consistent with the
Kennedy-Gruber volume flux `kennedy_gruber_cartesian_flux`.
"""
function lmars_cartesian(normal, (y⁻,), (y⁺,))
    # face normal in Cartesian components (ê_c single-valued at the node)
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    unL = y⁻.u1 * n1 + y⁻.u2 * n2 + y⁻.u3 * n3
    unR = y⁺.u1 * n1 + y⁺.u2 * n2 + y⁺.u3 * n3
    C = (y⁻.ρ + y⁺.ρ) / 2 * (y⁻.c + y⁺.c) / 2      # reference impedance ρ̄ĉ
    # Acoustic solve on the momentum pressure `pm` (= p full / p' stratified) so
    # u*, p* vanish at rest even over topography; enthalpy below keeps full p.
    ustar = (unL + unR) / 2 - (y⁺.pm - y⁻.pm) / (2 * C)
    pstar = (y⁻.pm + y⁺.pm) / 2 - C * (unR - unL) / 2
    # upwind (branchless) the advected quantities at u*
    pos = ustar >= 0
    ρup = ifelse(pos, y⁻.ρ, y⁺.ρ)
    ρeup = ifelse(pos, y⁻.ρe, y⁺.ρe)
    pup = ifelse(pos, y⁻.p, y⁺.p)
    u1up = ifelse(pos, y⁻.u1, y⁺.u1)
    u2up = ifelse(pos, y⁻.u2, y⁺.u2)
    u3up = ifelse(pos, y⁻.u3, y⁺.u3)
    return (
        ρ = ustar * ρup,
        ρe = ustar * (ρeup + pup),                 # enthalpy flux (full p)
        ρu1 = ustar * (ρup * u1up) + pstar * n1,
        ρu2 = ustar * (ρup * u2up) + pstar * n2,
        ρu3 = ustar * (ρup * u3up) + pstar * n3,
    )
end

"""
    lmars_tracer(normal, argvals⁻, argvals⁺)

Interface flux for a passive tracer `ρq` that is **consistent with the LMARS mass
flux**: the tracer is upwinded at the SAME low-Mach contact velocity `u*` that
[`lmars_cartesian`](@ref) uses for continuity/momentum, so a uniform `q` reproduces
`q·(u*·ρ_up)` = `q` × the LMARS continuity flux (free-stream / constancy
preserving). Use this for `ρq_tot` whenever the dynamics use `INTERFACE_FLUX=lmars`,
so mass and tracer share one interface velocity (a `kennedy_gruber_rusanov_tracer`
here would advect moisture with a *different* mass flux and inject spurious tracer).
`u*` is `√(γp/ρ)`-free and vanishes at rest over terrain. State fields required:
`ρ`, `c`, `pm`, `u1/u2/u3`, `E1/E2/E3`, `q`.
"""
function lmars_tracer(normal, (y⁻,), (y⁺,))
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    unL = y⁻.u1 * n1 + y⁻.u2 * n2 + y⁻.u3 * n3
    unR = y⁺.u1 * n1 + y⁺.u2 * n2 + y⁺.u3 * n3
    C = (y⁻.ρ + y⁺.ρ) / 2 * (y⁻.c + y⁺.c) / 2      # reference impedance ρ̄ĉ
    ustar = (unL + unR) / 2 - (y⁺.pm - y⁻.pm) / (2 * C)
    ρqup = ifelse(ustar >= 0, y⁻.ρ * y⁻.q, y⁺.ρ * y⁺.q)
    return (ρq = ustar * ρqup,)
end

"""
    lmars_cartesian_advective(normal, argvals⁻, argvals⁺)

Advection-only counterpart of [`lmars_cartesian`](@ref): keeps LMARS's low-Mach
contact velocity `u* = ½(uₙ⁻+uₙ⁺) − (pm⁺−pm⁻)/(2C)` and upwinds the advected
quantities (`ρ`, `ρe`, `ρu_c`) at `u*`, but OMITS the conservative pressure flux
`p* n`. Used with a non-conservative (Exner-perturbation) pressure-gradient force
(`kennedy_gruber_cartesian_advective_flux` volume flux): the interface supplies
LMARS's wave-selective, `sqrt(γp/ρ)`-free advective dissipation (impedance
`C = ρ̄ĉ`, `ĉ = √(γR_d T_ref)`) while the PGF is handled separately, exactly as
the Roe/Rusanov advective counterparts. Well-balanced: at a shared node `pm⁻=pm⁺`
at rest ⇒ `u*=0` ⇒ zero interface flux.
"""
function lmars_cartesian_advective(normal, (y⁻,), (y⁺,))
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    unL = y⁻.u1 * n1 + y⁻.u2 * n2 + y⁻.u3 * n3
    unR = y⁺.u1 * n1 + y⁺.u2 * n2 + y⁺.u3 * n3
    C = (y⁻.ρ + y⁺.ρ) / 2 * (y⁻.c + y⁺.c) / 2      # reference impedance ρ̄ĉ
    ustar = (unL + unR) / 2 - (y⁺.pm - y⁻.pm) / (2 * C)
    pos = ustar >= 0
    ρup = ifelse(pos, y⁻.ρ, y⁺.ρ)
    ρeup = ifelse(pos, y⁻.ρe, y⁺.ρe)
    pup = ifelse(pos, y⁻.p, y⁺.p)
    u1up = ifelse(pos, y⁻.u1, y⁺.u1)
    u2up = ifelse(pos, y⁻.u2, y⁺.u2)
    u3up = ifelse(pos, y⁻.u3, y⁺.u3)
    return (
        ρ = ustar * ρup,
        ρe = ustar * (ρeup + pup),                 # enthalpy flux (full p)
        ρu1 = ustar * (ρup * u1up),                # NO pressure flux (Exner PGF)
        ρu2 = ustar * (ρup * u2up),
        ρu3 = ustar * (ρup * u3up),
    )
end

"""
    kennedy_gruber_cartesian_advective_flux(nvec_a, nvec_b, y_a, y_b)

Advection-only variant of [`kennedy_gruber_cartesian_flux`](@ref): the momentum
flux omits the pressure term ``p̄ \\{ê_c ⋅ n\\}``, leaving the pure kinetic
Kennedy-Gruber flux ``ρ̄ ū_c ūn``. Used when the pressure-gradient force is
supplied separately in non-conservative (Exner-perturbation) form (Yatunin et
al. 2026): momentum conservation is traded for a well-balanced pressure
gradient, while the KEP property of the advective flux — and the mass and
energy (enthalpy) fluxes — are unchanged.
"""
function kennedy_gruber_cartesian_advective_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    return (
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn,
        ρu2 = ρ̄ * ū2 * ūn,
        ρu3 = ρ̄ * ū3 * ūn,
    )
end

"""
    kennedy_gruber_rusanov_cartesian_advective(normal, argvals⁻, argvals⁺)

Advection-only counterpart of [`kennedy_gruber_rusanov_cartesian`](@ref): the
central part omits the momentum pressure flux (see
[`kennedy_gruber_cartesian_advective_flux`](@ref)); the Rusanov dissipation is
unchanged.
"""
function kennedy_gruber_rusanov_cartesian_advective(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_cartesian_advective_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
        ρu1 = F.ρu1 - λ / 2 * (y⁺.ρ * y⁺.u1 - y⁻.ρ * y⁻.u1),
        ρu2 = F.ρu2 - λ / 2 * (y⁺.ρ * y⁺.u2 - y⁻.ρ * y⁻.u2),
        ρu3 = F.ρu3 - λ / 2 * (y⁺.ρ * y⁺.u3 - y⁻.ρ * y⁻.u3),
    )
end

"""
    kennedy_gruber_roe_cartesian_advective(normal, argvals⁻, argvals⁺)

Advection-only counterpart of [`kennedy_gruber_roe_cartesian`](@ref): the full
Roe flux minus its central momentum pressure term ``p̄ \\{ê_c ⋅ n\\}`` (mass,
energy and all wave-selective dissipation unchanged). `ê_c` is single-valued at
the shared node, so ``\\{ê_c ⋅ n\\} = ((E_c⁻ + E_c⁺)/2) ⋅ n``.
"""
function kennedy_gruber_roe_cartesian_advective(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    p̄ = (y⁻.p + y⁺.p) / 2
    return (
        ρ = F.ρ,
        ρe = F.ρe,
        ρu1 = F.ρu1 - p̄ * (((y⁻.E1 + y⁺.E1) / 2)' * normal),
        ρu2 = F.ρu2 - p̄ * (((y⁻.E2 + y⁺.E2) / 2)' * normal),
        ρu3 = F.ρu3 - p̄ * (((y⁻.E3 + y⁺.E3) / 2)' * normal),
    )
end

"""
    kg_massflux_fluctuation(nvec_a, nvec_b, y_a, y_b)

Non-symmetric two-point FLUCTUATION form for the advective operator
``(u·∇_h)u_c`` acting on velocity components in velocity (non-conservative)
form, driven by the Kennedy-Gruber mass flux:
``P^\\#_c(a, b) = F^\\#_ρ(a, b)\\,(u_{c,b} - u_{c,a})/2`` with
``F^\\#_ρ = \\{ρ\\}\\{u ⋅ nvec\\}`` (contravariant nodal fluxes averaged).

Pass to [`add_flux_differencing_divergence!`](@ref); the own-side boundary
lifts evaluate to zero (the jump vanishes for `y_a == y_b`), so the kernel
degenerates to the pure strong-form fluctuation sum. The mass-weighted
result divided by ``ρ\\,WJ`` is ``-(u·∇_h)u_c``, replacing BOTH the
relative-vorticity cross product and the horizontal-KE gradient of the
vector-invariant form.

KE compatibility with the KG mass flux (the fluctuation-form analog of the
KEP property): ``K_i F^\\#_ρ(i,j) + u_{c,i} P^\\#_c(i,j)`` (summed over
components) equals ``F^\\#_ρ(i,j)\\,(u_i · u_j)/2``, which is symmetric, so
the volume kinetic-energy production telescopes to face terms; complete them
with [`advective_fluctuation_lift`](@ref). The advected components must be
in a globally constant frame (e.g. Cartesian) — position-dependent frames
reintroduce curvature terms the jumps cannot see.

State fields required: `ρ`, `uv`, `u1`, `u2`, `u3`.
"""
function kg_massflux_fluctuation(nvec_a, nvec_b, y_a, y_b)
    F = ((y_a.ρ + y_b.ρ) / 2) * ((y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2)
    return (
        u1 = F * (y_b.u1 - y_a.u1) / 2,
        u2 = F * (y_b.u2 - y_a.u2) / 2,
        u3 = F * (y_b.u3 - y_a.u3) / 2,
    )
end

"""
    advective_fluctuation_lift(normal, argvals⁻, argvals⁺)

Per-component face SAT completing [`kg_massflux_fluctuation`](@ref): the
EXACT velocity-variables transform of the flux-form KG central face
treatment, ``δu_c = (δ(ρ u_c) - u_c\\,δρ)/ρ`` applied to the central
interface fluxes — the own-flux terms cancel, leaving each side
``-\\{ρ\\}(\\{uv\\} ⋅ n̂_{side})\\,(u_c^{other} - u_c^{side})/2``.
Because it is the exact transform of a KE-consistent face treatment, the
face kinetic-energy bookkeeping is identical to the flux form's. (The same
transform of the Rusanov jumps reproduces the λ velocity-jump penalties
with a ``ρ^{other}/ρ^{side}`` weight; the plain penalties are their
constant-ρ limit and provide the face dissipation.) NOTE the sign: the
naive "central lifting" sign (+) is anti-consistent and exponentially
unstable at O(jump). Use through [`lifting_correction`](@ref) with argument
fields `(u_c, ρ, uv)`; divide the result by ``ρ`` for the velocity
tendency.
"""
function advective_fluctuation_lift(normal, argvals⁻, argvals⁺)
    u_c⁻, ρ⁻, uv⁻ = argvals⁻
    u_c⁺, ρ⁺, uv⁺ = argvals⁺
    F = ((ρ⁻ + ρ⁺) / 2) * (((uv⁻ + uv⁺) / 2)' * normal)
    return -F * (u_c⁺ - u_c⁻) / 2
end

"""
    kennedy_gruber_height_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber-style two-point mass flux ``\\{h\\}\\{u ⋅ nvec\\}`` for the
shallow-water height equation (contravariant nodal fluxes averaged). State
fields required: `h`, `uv`.
"""
kennedy_gruber_height_flux(nvec_a, nvec_b, y_a, y_b) =
    ((y_a.h + y_b.h) / 2) * ((y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2)

"""
    kennedy_gruber_rusanov_height(normal, argvals⁻, argvals⁺)

Interface flux for the shallow-water height equation:
[`kennedy_gruber_height_flux`](@ref) central part plus a Rusanov penalty
scaled by the state field `λ`.
"""
function kennedy_gruber_rusanov_height(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    return kennedy_gruber_height_flux(normal, normal, y⁻, y⁺) -
           λ / 2 * (y⁺.h - y⁻.h)
end
