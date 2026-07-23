#=
CUDA implementations of the DG internal-face and flux-differencing volume
operators (see src/Operators/numericalflux.jl for the CPU methods and the
operator contracts).

Kernel design:
- Flux-differencing volume (`_add_flux_differencing_divergence!`): one thread
  per node per level, one block-column per element (the spectral-partition
  layout); element-local, so race-free. The per-node math is
  `Operators._fd_volume_node_total`, shared verbatim with the CPU path.
- Internal-face fluxes (`_add_numerical_flux_internal!`,
  `_add_lifting_flux_internal!`): two-pass staging + gather, following the
  structure of the DSS GPU kernels (topologies_dss.jl). Pass A computes each
  face-node flux exactly once into a staging array (race-free by
  construction); pass B gathers the (1 or 2) face contributions of each
  element-boundary node via the deterministic map in
  `Operators.DGConnectivity` (no atomics; bitwise-reproducible).
- `tensor_product!` (cutoff filter): per-element kernel with a shared-memory
  stage of the input slab so in-place application is safe.

Note: on the GPU path, flux/lift functions receive *plain* values (no
`AutoBroadcaster` wrappers); all Operators-provided flux functions satisfy
this.
=#

import ClimaCore: Operators, Topologies, Quadratures, Grids
import ClimaCore.Operators: DGConnectivity

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
        Nv = Spaces.nlevels(space)
    else
        @assert grid isa Grids.SpectralElementGrid2D
        Nv = 1
    end
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
        y_at = (a, b) -> y_data[CI(a, b, 1, v, h)]
        lg_at = (a, b) -> lg_data[CI(a, b, 1, v, h)]
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
        I = CI(i, j, 1, v, h)
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
    args...,
) where {F} = _dg_face_apply!(fn, dydt, args, Val(:numflux))

Operators._add_lifting_flux_internal!(
    ::ClimaComms.CUDADevice,
    fn::F,
    dydt,
    args...,
) where {F} = _dg_face_apply!(fn, dydt, args, Val(:lifting))

function _dg_face_apply!(fn::F, dydt, args, mode::Val) where {F}
    space = axes(dydt)
    grid = Spaces.grid(space)
    if grid isa Grids.ExtrudedFiniteDifferenceGrid
        @assert grid.horizontal_grid isa Grids.SpectralElementGrid2D
        Nv = Spaces.nlevels(space)
    else
        @assert grid isa Grids.SpectralElementGrid2D
        Nv = 1
    end
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    conn = Operators.dg_connectivity(space)
    conn.nfaces == 0 && return dydt

    dydt_data = Fields.field_values(dydt)
    args_data =
        map(a -> a isa Fields.Field ? Fields.field_values(a) : a, args)
    T = eltype(dydt_data)
    DA = ClimaComms.array_type(Spaces.topology(space))
    nsides = mode isa Val{:lifting} ? 2 : 1
    staging = DA{T}(undef, Nq, Nv, nsides, conn.nfaces)

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
    return dydt
end

function dg_face_flux_kernel!(
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
    if gidx ≤ Nq * Nv * nfaces
        (q, v, f) = cart_ind((Nq, Nv, nfaces), gidx).I
        elem⁻ = Int(faces[1, f])
        face⁻ = Int(faces[2, f])
        elem⁺ = Int(faces[3, f])
        face⁺ = Int(faces[4, f])
        reversed = faces[5, f] == Int32(1)
        i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
        i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)
        CI = CartesianIndex
        argvals⁻ = map(
            a ->
                a isa DataLayouts.AbstractData ?
                a[CI(i⁻, j⁻, 1, v, elem⁻)] : a,
            args_data,
        )
        argvals⁺ = map(
            a ->
                a isa DataLayouts.AbstractData ?
                a[CI(i⁺, j⁺, 1, v, elem⁺)] : a,
            args_data,
        )
        sg = sgeom[q, v, f]
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
        (v, n) = cart_ind((Nv, nbnodes), gidx).I
        e = Int(node_elem[n])
        i = Int(node_i[n])
        j = Int(node_j[n])
        I = CartesianIndex(i, j, 1, v, e)
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
# tensor_product! (cutoff filter); square matrices only
# ---------------------------------------------------------------------------

function Operators._tensor_product!(
    ::DataLayouts.ToCUDA,
    out::DataLayouts.Data2DX{S, Nv, Nij},
    indata::DataLayouts.Data2DX{S, Nv, Nij},
    M::SMatrix{Nij, Nij},
) where {S, Nv, Nij}
    (_, _, _, _, Nh) = DataLayouts.universal_size(out)
    Nvt = max(1, min(fld(_max_threads_cuda(), Nij * Nij), Nv))
    auto_launch!(
        dg_tensor_product_kernel!,
        (out, indata, M, Val(Nij), Val(Nvt), Nv);
        threads_s = (Nij, Nij, Nvt),
        blocks_s = (Nh, cld(Nv, Nvt)),
    )
    return out
end

function Operators._tensor_product!(
    ::DataLayouts.ToCUDA,
    out::DataLayouts.Data2D{S, Nij},
    indata::DataLayouts.Data2D{S, Nij},
    M::SMatrix{Nij, Nij},
) where {S, Nij}
    (_, _, _, _, Nh) = DataLayouts.universal_size(out)
    auto_launch!(
        dg_tensor_product_kernel!,
        (out, indata, M, Val(Nij), Val(1), 1);
        threads_s = (Nij, Nij, 1),
        blocks_s = (Nh, 1),
    )
    return out
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
    work = CUDA.CuStaticSharedArray(S, (Nij, Nij, Nvt))
    i = threadIdx().x
    j = threadIdx().y
    k = threadIdx().z
    h = blockIdx().x
    v = k + (blockIdx().y - Int32(1)) * blockDim().z
    CI = CartesianIndex
    if v ≤ Nv
        work[i, j, k] = indata[CI(i, j, 1, v, h)]
    end
    CUDA.sync_threads()
    if v ≤ Nv
        r = M[i, 1] * M[j, 1] * work[1, 1, k]
        for jj in 1:Nij, ii in 1:Nij
            (ii == 1 && jj == 1) && continue
            r = r + M[i, ii] * M[j, jj] * work[ii, jj, k]
        end
        out[CI(i, j, 1, v, h)] = r
    end
    return nothing
end
