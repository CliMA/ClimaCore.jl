#=
CUDA implementations of the DG internal-face and flux-differencing volume
operators (see src/Operators/numericalflux.jl for the CPU methods and the
operator contracts).

Kernel design:
- Flux-differencing volume (`_add_flux_differencing_divergence!`), element-local.
- Internal-face fluxes (`_add_numerical_flux_internal!`,
  `_add_lifting_flux_internal!`): two-pass staging + gather, follows `DSS` GPU kernels.
- Ghost (inter-rank) faces: face-strip halo exchange
  (`Topologies.GhostFaceExchange`, one pack kernel per argument) started
  before the interior kernels (overlapping communication with compute), then
  the same staging + gather over `Operators.dg_ghost_connectivity`, with the
  plus side read from the recv strips.
- Boundary faces (`_add_numerical_flux_boundary!`): one-sided staging + the
  same gather, over `Operators.dg_boundary_connectivity`.
- `tensor_product!` (cutoff filter): per-element enabling in-place assignment.
=#

import ClimaCore: Operators, Topologies, Quadratures, Grids, DataLayouts
import ClimaCore.Operators: DGConnectivity
import UnrolledUtilities: unrolled_map

# ---------------------------------------------------------------------------
# Flux-differencing volume divergence
# ---------------------------------------------------------------------------

function Operators._add_flux_differencing_divergence!(
    ::ClimaComms.CUDADevice,
    fn2pt::F,
    dydt,
    y,
) where {F}
    space = axes(dydt)
    grid = Spaces.grid(space)
    if grid isa Grids.ExtrudedFiniteDifferenceGrid
        @assert grid.horizontal_grid isa Grids.SpectralElementGrid2D
    else
        @assert grid isa Grids.SpectralElementGrid2D
    end
    Nv = Spaces.nlevels(space)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, quadrature_style)
    (_, w) = Quadratures.quadrature_points(FT, quadrature_style)
    Nh = Topologies.nlocalelems(Spaces.topology(space))

    dydt_data = Fields.field_values(dydt)
    y_data = Fields.field_values(y)
    lg_data = Spaces.local_geometry_data(space)

    Nvt = max(1, min(fld(_max_threads_cuda(), Nq * Nq), Nv))
    args = (dydt_data, y_data, lg_data, fn2pt, D, w, Val(Nq), Nv)
    auto_launch!(
        dg_fddg_volume_kernel!,
        args;
        threads_s = (Nq, Nq, Nvt),
        blocks_s = (Nh, cld(Nv, Nvt)),
    )
    return dydt
end

function dg_fddg_volume_kernel!(
    dydt_data,
    y_data,
    lg_data,
    fn2pt::F,
    D,
    w,
    ::Val{Nq},
    Nv,
) where {F, Nq}
    i = threadIdx().x
    j = threadIdx().y
    h = blockIdx().x
    v = threadIdx().z + (blockIdx().y - Int32(1)) * blockDim().z
    if v ≤ Nv
        CI = CartesianIndex
        y_at = (a, b) -> y_data[CI(v, a, b, h)]
        lg_at = (a, b) -> lg_data[CI(v, a, b, h)]
        total = Operators._fd_volume_node_total(
            fn2pt,
            y_at,
            lg_at,
            D,
            w,
            Val(Nq),
            i,
            j,
        )
        I = CI(v, i, j, h)
        dydt_data[I] = Operators._fd_add(dydt_data[I], total)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Internal-face numerical flux and symmetric lifting (staging + gather)
# ---------------------------------------------------------------------------

Operators._add_numerical_flux_internal!(
    ::ClimaComms.CUDADevice,
    fn::F,
    dydt,
    args...;
    ghost_exchange = nothing,
) where {F} = _dg_face_apply!(fn, dydt, args, Val(:numflux); ghost_exchange)

Operators._add_lifting_flux_internal!(
    ::ClimaComms.CUDADevice,
    fn::F,
    dydt,
    args...;
    ghost_exchange = nothing,
) where {F} = _dg_face_apply!(fn, dydt, args, Val(:lifting); ghost_exchange)

# The face-strip halo exchange starts before the interior-face kernels, so the
# communication overlaps with the interior compute; each argument is packed by
# a single kernel. See `Operators.start_dg_ghost_exchange` and
# `Topologies.GhostFaceExchange` for the exchange semantics.
function Operators._start_dg_ghost_exchange_handle(
    ::ClimaComms.CUDADevice,
    space,
    args,
)
    topology = Spaces.topology(space)
    ClimaComms.context(topology) isa ClimaComms.SingletonCommsContext &&
        return Operators.NO_DG_GHOST_EXCHANGE
    Nv = Spaces.nlevels(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    args_data =
        unrolled_map(a -> a isa Fields.Field ? Fields.field_values(a) : a, args)
    bufs = ntuple(Val(length(args_data))) do i
        a = args_data[i]
        a isa DataLayouts.DataLayout || return nothing
        Operators._dg_face_exchange(space, a, i)
    end
    Operators._claim_dg_face_exchanges!(bufs)
    foreach(ntuple(identity, Val(length(args_data)))) do i
        ex = bufs[i]
        isnothing(ex) && return
        nstrips = length(ex.slot_lidx)
        p = linear_partition(Nq * Nv * nstrips, _max_threads_cuda())
        auto_launch!(
            dg_face_pack_kernel!,
            (
                ex.send_data,
                args_data[i],
                ex.slot_lidx,
                ex.slot_face,
                Val(Nq),
                Nv,
                nstrips,
            );
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    end
    # MPI reads the device-side send buffers on the host timeline, so the
    # pack kernels must have completed (the DSS-start pattern).
    Spaces.cuda_synchronize(ClimaComms.device(space); blocking = true)
    foreach(ex -> isnothing(ex) || ClimaComms.start(ex.graph_context), bufs)
    return Operators.DGGhostExchange(bufs)
end

Operators._dg_shared_or_start(
    device::ClimaComms.CUDADevice,
    space,
    args,
    ::Nothing,
) = Operators._start_dg_ghost_exchange_handle(device, space, args)

function _dg_face_apply!(
    fn::F,
    dydt,
    args,
    mode::Val;
    ghost_exchange = nothing,
) where {F}
    space = axes(dydt)
    topology = Spaces.topology(space)
    grid = Spaces.grid(space)
    if grid isa Grids.ExtrudedFiniteDifferenceGrid
        @assert grid.horizontal_grid isa Grids.SpectralElementGrid2D
    else
        @assert grid isa Grids.SpectralElementGrid2D
    end
    Nv = Spaces.nlevels(space)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    conn = Operators.dg_connectivity(space)
    distributed =
        !(ClimaComms.context(topology) isa ClimaComms.SingletonCommsContext)
    gconn = distributed ? Operators.dg_ghost_connectivity(space) : nothing
    !distributed && conn.nfaces == 0 && return dydt

    dydt_data = Fields.field_values(dydt)
    args_data =
        unrolled_map(a -> a isa Fields.Field ? Fields.field_values(a) : a, args)
    T = eltype(dydt_data)
    DA = ClimaComms.array_type(topology)

    ghost = Operators._dg_shared_or_start(
        ClimaComms.device(space),
        space,
        args,
        ghost_exchange,
    )

    if conn.nfaces > 0
        nsides = mode isa Val{:lifting} ? 2 : 1
        staging = Operators._dg_staging_buffer(space, T, nsides, conn.nfaces)

        nitemsA = Nq * Nv * conn.nfaces
        pA = linear_partition(nitemsA, _max_threads_cuda())
        auto_launch!(
            dg_face_flux_kernel!,
            (
                staging,
                fn,
                args_data,
                conn.faces,
                conn.sgeom,
                Val(Nq),
                Nv,
                conn.nfaces,
                mode,
            );
            threads_s = pA.threads,
            blocks_s = pA.blocks,
        )

        nitemsB = conn.nbnodes * Nv
        pB = linear_partition(nitemsB, _max_threads_cuda())
        auto_launch!(
            dg_face_gather_kernel!,
            (
                dydt_data,
                staging,
                conn.node_elem,
                conn.node_i,
                conn.node_j,
                conn.node_offset,
                conn.contrib_face,
                conn.contrib_side,
                conn.contrib_q,
                Nv,
                conn.nbnodes,
                mode,
            );
            threads_s = pB.threads,
            blocks_s = pB.blocks,
        )
    end

    ghost_bufs = Operators._consume_dg_ghost_exchange!(ghost)
    if !isnothing(ghost_bufs)
        isnothing(gconn) && return dydt
        recv = unrolled_map(
            ex -> isnothing(ex) ? nothing : ex.recv_data,
            ghost_bufs,
        )

        # Only the minus side is staged (the mirror ghost face on the
        # neighbouring rank accumulates the other side), so one staging slot
        # per face node serves both modes, and the gather kernel is reused
        # with the ghost gather map, whose contributions are all side 1.
        gstaging = Operators._dg_ghost_staging_buffer(space, T, gconn.nfaces)

        nitemsC = Nq * Nv * gconn.nfaces
        pC = linear_partition(nitemsC, _max_threads_cuda())
        auto_launch!(
            dg_ghost_face_flux_kernel!,
            (
                gstaging,
                fn,
                args_data,
                recv,
                gconn.faces,
                gconn.sgeom,
                Val(Nq),
                Nv,
                gconn.nfaces,
            );
            threads_s = pC.threads,
            blocks_s = pC.blocks,
        )

        nitemsD = gconn.nbnodes * Nv
        pD = linear_partition(nitemsD, _max_threads_cuda())
        auto_launch!(
            dg_face_gather_kernel!,
            (
                dydt_data,
                gstaging,
                gconn.node_elem,
                gconn.node_i,
                gconn.node_j,
                gconn.node_offset,
                gconn.contrib_face,
                gconn.contrib_side,
                gconn.contrib_q,
                Nv,
                gconn.nbnodes,
                mode,
            );
            threads_s = pD.threads,
            blocks_s = pD.blocks,
        )
    end
    return dydt
end

# Shared prologue for the two face-flux kernels: decode face `f` at node `q`
# and level `v`, gather the minus-side argument values, and fetch the surface
# geometry. Returns the plus-side node index `(idx⁺, i⁺, j⁺)` — where `idx⁺`
# is a local element (interior) or a ghost receive index (ghost), both in the
# third `faces` row — for the caller to read the plus-side values from.
# `@propagate_inbounds` so the caller's `@inbounds` reaches the array accesses.
Base.@propagate_inbounds function _dg_flux_minus(
    args_data,
    faces,
    sgeom,
    ::Val{Nq},
    q,
    v,
    f,
) where {Nq}
    elem⁻ = Int(faces[1, f])
    face⁻ = Int(faces[2, f])
    idx⁺ = Int(faces[3, f])
    face⁺ = Int(faces[4, f])
    reversed = faces[5, f] == Int32(1)
    i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
    i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)
    CI = CartesianIndex
    argvals⁻ = unrolled_map(
        a -> a isa DataLayouts.DataLayout ? a[CI(v, i⁻, j⁻, elem⁻)] : a,
        args_data,
    )
    return sgeom[q, v, f], argvals⁻, idx⁺, i⁺, j⁺
end

Base.@propagate_inbounds function dg_face_flux_kernel!(
    staging,
    fn::F,
    args_data,
    faces,
    sgeom,
    ::Val{Nq},
    Nv,
    nfaces,
    mode,
) where {F, Nq}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    @inbounds if gidx ≤ Nq * Nv * nfaces
        (q, v, f) = Utilities.cart_ind((Nq, Nv, nfaces), gidx).I
        sg, argvals⁻, elem⁺, i⁺, j⁺ =
            _dg_flux_minus(args_data, faces, sgeom, Val(Nq), q, v, f)
        CI = CartesianIndex
        argvals⁺ = unrolled_map(
            a -> a isa DataLayouts.DataLayout ? a[CI(v, i⁺, j⁺, elem⁺)] : a,
            args_data,
        )
        if mode isa Val{:numflux}
            val = fn(sg.normal, argvals⁻, argvals⁺)
            staging[q, v, 1, f] = Operators._fd_scale(sg.sWJ, val)
        else
            lift⁻ = fn(sg.normal, argvals⁻, argvals⁺)
            lift⁺ = fn(-sg.normal, argvals⁺, argvals⁻)
            staging[q, v, 1, f] = Operators._fd_scale(sg.sWJ, lift⁻)
            staging[q, v, 2, f] = Operators._fd_scale(sg.sWJ, lift⁺)
        end
    end
    return nothing
end

# Pack one argument's ghost-face strips for the `Topologies.GhostFaceExchange`
# (the single-kernel analog of `Topologies.fill_face_send_buffer!`): strip `s`
# holds the `Nq` face-node values of local face `(slot_lidx[s], slot_face[s])`,
# in the face's natural node order.
function dg_face_pack_kernel!(
    send_data,
    data,
    slot_lidx,
    slot_face,
    ::Val{Nq},
    Nv,
    nstrips,
) where {Nq}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if gidx ≤ Nq * Nv * nstrips
        (q, v, s) = Utilities.cart_ind((Nq, Nv, nstrips), gidx).I
        i, j = Topologies.face_node_index(Int(slot_face[s]), Nq, q, false)
        CI = CartesianIndex
        send_data[CI(v, q, 1, s)] = data[CI(v, i, j, Int(slot_lidx[s]))]
    end
    return nothing
end

# Ghost-face variant of `dg_face_flux_kernel!`: the plus side is read from the
# recv strips of each argument's ghost exchange — the third `faces` row holds
# the strip slot, and the value at loop node `q` is strip node
# `reversed ? Nq - q + 1 : q` (the sender packs in its natural face order) —
# instead of a local element; arguments without a recv buffer are identical on
# both sides. Only the minus side is staged — `fn(n̂⁻, ·⁻, ·⁺)` is the staged
# value for both the antisymmetric numerical flux and the symmetric lifting,
# and the gather kernel applies the mode-dependent sign at side 1 — so no
# `mode` argument is needed.
function dg_ghost_face_flux_kernel!(
    staging,
    fn::F,
    args_data,
    recv_data,
    faces,
    sgeom,
    ::Val{Nq},
    Nv,
    nfaces,
) where {F, Nq}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if gidx ≤ Nq * Nv * nfaces
        (q, v, f) = Utilities.cart_ind((Nq, Nv, nfaces), gidx).I
        sg, argvals⁻, slot, _i⁺, _j⁺ =
            _dg_flux_minus(args_data, faces, sgeom, Val(Nq), q, v, f)
        reversed = faces[5, f] == Int32(1)
        q′ = reversed ? Nq - q + 1 : q
        CI = CartesianIndex
        argvals⁺ = unrolled_map(
            (a, r) -> isnothing(r) ? a : r[CI(v, q′, 1, slot)],
            args_data,
            recv_data,
        )
        staging[q, v, 1, f] =
            Operators._fd_scale(sg.sWJ, fn(sg.normal, argvals⁻, argvals⁺))
    end
    return nothing
end

function dg_face_gather_kernel!(
    dydt_data,
    staging,
    node_elem,
    node_i,
    node_j,
    node_offset,
    contrib_face,
    contrib_side,
    contrib_q,
    Nv,
    nbnodes,
    mode,
)
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if gidx ≤ nbnodes * Nv
        (v, n) = Utilities.cart_ind((Nv, nbnodes), gidx).I
        e = Int(node_elem[n])
        i = Int(node_i[n])
        j = Int(node_j[n])
        I = CartesianIndex(v, i, j, e)
        acc = dydt_data[I]
        for c in Int(node_offset[n]):(Int(node_offset[n + 1]) - 1)
            f = Int(contrib_face[c])
            q = Int(contrib_q[c])
            side = Int(contrib_side[c])
            if mode isa Val{:numflux}
                s = staging[q, v, 1, f]
                # minus side subtracts, plus side adds (antisymmetric flux)
                acc = Operators._fd_add(
                    acc,
                    Operators._fd_scale(side == 1 ? -1 : 1, s),
                )
            else
                # symmetric lifting: each side adds its own lift
                acc = Operators._fd_add(acc, staging[q, v, side, f])
            end
        end
        dydt_data[I] = acc
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Boundary-face numerical flux (staging + gather)
# ---------------------------------------------------------------------------

Operators._add_numerical_flux_boundary!(
    ::ClimaComms.CUDADevice,
    fn::F,
    dydt,
    args...,
) where {F} = _dg_boundary_apply!(fn, dydt, args)

function _dg_boundary_apply!(fn::F, dydt, args) where {F}
    space = axes(dydt)
    bconn = Operators.dg_boundary_connectivity(space)
    isnothing(bconn) && return dydt
    grid = Spaces.grid(space)
    if grid isa Grids.ExtrudedFiniteDifferenceGrid
        @assert grid.horizontal_grid isa Grids.SpectralElementGrid2D
    else
        @assert grid isa Grids.SpectralElementGrid2D
    end
    Nv = Spaces.nlevels(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))

    dydt_data = Fields.field_values(dydt)
    args_data =
        unrolled_map(a -> a isa Fields.Field ? Fields.field_values(a) : a, args)
    T = eltype(dydt_data)
    staging = Operators._dg_boundary_staging_buffer(space, T, bconn.nfaces)

    nitemsA = Nq * Nv * bconn.nfaces
    pA = linear_partition(nitemsA, _max_threads_cuda())
    auto_launch!(
        dg_boundary_face_flux_kernel!,
        (staging, fn, args_data, bconn.faces, bconn.sgeom, Val(Nq), Nv, bconn.nfaces);
        threads_s = pA.threads,
        blocks_s = pA.blocks,
    )

    # The gather kernel in `:numflux` mode subtracts side-1 staging, which is
    # the boundary accumulation `dydt -= sWJ * fn(n̂, ·⁻)` (all boundary
    # contributions are side 1).
    nitemsB = bconn.nbnodes * Nv
    pB = linear_partition(nitemsB, _max_threads_cuda())
    auto_launch!(
        dg_face_gather_kernel!,
        (
            dydt_data,
            staging,
            bconn.node_elem,
            bconn.node_i,
            bconn.node_j,
            bconn.node_offset,
            bconn.contrib_face,
            bconn.contrib_side,
            bconn.contrib_q,
            Nv,
            bconn.nbnodes,
            Val(:numflux),
        );
        threads_s = pB.threads,
        blocks_s = pB.blocks,
    )
    return dydt
end

# Boundary faces have no plus side and no `reversed` flag (the `faces` matrix
# holds `(elem⁻, face⁻)` rows only), so this does not share `_dg_flux_minus`
# with the interior/ghost kernels.
function dg_boundary_face_flux_kernel!(
    staging,
    fn::F,
    args_data,
    faces,
    sgeom,
    ::Val{Nq},
    Nv,
    nfaces,
) where {F, Nq}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if gidx ≤ Nq * Nv * nfaces
        (q, v, f) = Utilities.cart_ind((Nq, Nv, nfaces), gidx).I
        elem⁻ = Int(faces[1, f])
        face⁻ = Int(faces[2, f])
        i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
        CI = CartesianIndex
        argvals⁻ = unrolled_map(
            a -> a isa DataLayouts.DataLayout ? a[CI(v, i⁻, j⁻, elem⁻)] : a,
            args_data,
        )
        sg = sgeom[q, v, f]
        staging[q, v, 1, f] =
            Operators._fd_scale(sg.sWJ, fn(sg.normal, argvals⁻))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# tensor_product! (cutoff filter); square matrices only
# ---------------------------------------------------------------------------

function Operators.tensor_product!(
    out::DataLayouts.VIJHWithF{S, Nv, Nij, Nij, Nh, F, Sc, A},
    indata::DataLayouts.VIJHWithF{S, Nv, Nij, Nij, Nh, F, Sc, A},
    M::SMatrix{Nij, Nij},
) where {S, Nv, Nij, Nh, F, Sc, A <: CUDA.CuArray}
    Nh_runtime = DataLayouts.nelems(out)
    @assert Nh_runtime == DataLayouts.nelems(indata)
    Nvt = max(1, min(fld(_max_threads_cuda(), Nij * Nij), Nv))
    auto_launch!(
        dg_tensor_product_kernel!,
        (out, indata, M, Val(Nij), Val(Nvt), Nv);
        threads_s = (Nij, Nij, Nvt),
        blocks_s = (Nh_runtime, cld(Nv, Nvt)),
    )
    return out
end

function Operators.tensor_product!(
    inout::DataLayouts.VIJHWithF{S, Nv, Nij, Nij, Nh, F, Sc, A},
    M::SMatrix{Nij, Nij},
) where {S, Nv, Nij, Nh, F, Sc, A <: CUDA.CuArray}
    return Operators.tensor_product!(inout, inout, M)
end

function dg_tensor_product_kernel!(
    out,
    indata,
    M,
    ::Val{Nij},
    ::Val{Nvt},
    Nv,
) where {Nij, Nvt}
    S = eltype(out)
    work = DataLayouts.scoped_static_array(ThisBlock(), S, (Nij, Nij, Nvt))
    i = threadIdx().x
    j = threadIdx().y
    k = threadIdx().z
    h = blockIdx().x
    v = k + (blockIdx().y - Int32(1)) * blockDim().z
    CI = CartesianIndex
    if v ≤ Nv
        work[i, j, k] = indata[CI(v, i, j, h)]
    end
    DataLayouts.synchronize(ThisBlock())
    if v ≤ Nv
        r = Operators._fd_scale(M[i, 1] * M[j, 1], work[1, 1, k])
        for jj in 1:Nij, ii in 1:Nij
            (ii == 1 && jj == 1) && continue
            r = Operators._fd_add(
                r,
                Operators._fd_scale(M[i, ii] * M[j, jj], work[ii, jj, k]),
            )
        end
        out[CI(v, i, j, h)] = r
    end
    return nothing
end
