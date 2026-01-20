import ClimaCore.Operators:
    columnwise!,
    device_sync_threads,
    columnwise_kernel!,
    universal_index_columnwise,
    local_mem

device_sync_threads(device::ClimaComms.CUDADevice) = CUDA.sync_threads()

local_mem(
    device::ClimaComms.CUDADevice,
    ::Type{T},
    ::Val{dims},
) where {T, dims} = CUDA.CuStaticSharedArray(T, dims)

function columnwise!(
    device::ClimaComms.CUDADevice,
    ᶜf::ᶜF,
    ᶠf::ᶠF,
    ᶜYₜ::Fields.Field,
    ᶠYₜ::Fields.Field,
    ᶜY::Fields.Field,
    ᶠY::Fields.Field,
    p,
    t,
    ::Val{localmem_lg} = Val(true),
    ::Val{localmem_state} = Val(true),
) where {ᶜF, ᶠF, localmem_lg, localmem_state}
    ᶜspace = axes(ᶜY)
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠNv = Spaces.nlevels(ᶠspace)
    ᶜcf = Fields.coordinate_field(ᶜspace)
    us = DataLayouts.UniversalSize(Fields.field_values(ᶜcf))
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    nitems = Ni * Nj * 1 * ᶠNv * Nh

    threads_per_column = ᶠNv
    threads_per_block = 256 # figure out a better way to estimate this
    ntotal_columns = Nh * Ni * Nj
    n_columns_per_block = fld(threads_per_block, threads_per_column)
    blocks = cld(ntotal_columns, n_columns_per_block)
    threads = threads_per_block

    kernel = CUDA.@cuda(
        always_inline = true,
        launch = false,
        columnwise_kernel!(
            device,
            ᶜf,
            ᶠf,
            ᶜYₜ,
            ᶠYₜ,
            ᶜY,
            ᶠY,
            p,
            t,
            nothing,
            Val(localmem_lg),
            Val(localmem_state),
        )
    )
    # threads = (ᶠNv,)
    # blocks = (Nh, Ni * Nj)
    kernel(
        device,
        ᶜf,
        ᶠf,
        ᶜYₜ,
        ᶠYₜ,
        ᶜY,
        ᶠY,
        p,
        t,
        nothing,
        Val(localmem_lg),
        Val(localmem_state);
        threads,
        blocks,
    )
end

# @inline function universal_index_columnwise(
#     device::ClimaComms.CUDADevice,
#     UI,
#     (v,) = CUDA.threadIdx()
#     (h, ij) = CUDA.blockIdx()
#     (Ni, Nj, _, _, _) = DataLayouts.universal_size(us)
#     Ni * Nj < ij && return CartesianIndex((-1, -1, 1, -1, -1))
#     @inbounds (i, j) = CartesianIndices((Ni, Nj))[ij].I
#     return CartesianIndex((i, j, 1, v, h))
# end

@inline function universal_index_columnwise(device::ClimaComms.CUDADevice, UI, us, ::Val{ᶠNv}) where {ᶠNv}
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    (v, i, j, h) = get_indices(Ni, Nj, ᶠNv, Nh)

    v_safe = 1 ≤ v ≤ ᶠNv ? v : 1
    i_safe = 1 ≤ i ≤ Ni ? i : 1
    j_safe = 1 ≤ j ≤ Nj ? j : 1
    h_safe = 1 ≤ h ≤ Nh ? h : 1

    # (1 ≤ v ≤ ᶠNv && oob(v, i, j, h)) || CUDA.@cuprint("bad v=$v, i=$i, j=$j, h=$h, ᶠNv=$ᶠNv\n")
    # (1 ≤ i ≤ Ni && oob(v, i, j, h)) || CUDA.@cuprint("bad i = $i\n")
    # (1 ≤ j ≤ Nj && oob(v, i, j, h)) || CUDA.@cuprint("bad j = $j\n")
    # (1 ≤ h ≤ Nh && oob(v, i, j, h)) || CUDA.@cuprint("bad h = $h, Nh = $Nh\n")

    ui = CartesianIndex((i_safe, j_safe, 1, v_safe, h_safe))
    return ui
end

oob(v, i, j, h) = v == -1 && i == -1 && j == -1 && h == -1

function get_indices(Ni, Nj, ᶠNv, Nh)
    # static parameters
    threads_per_column = ᶠNv # number of threads per column
    threads_per_block = 256 # number of threads per block
    ntotal_columns = Nh * Ni * Nj # total number of columns
    n_columns_per_block = fld(threads_per_block, threads_per_column) # number of columns per block
    n_blocks = cld(ntotal_columns, n_columns_per_block) # number of blocks (which have multiple columns)
    @assert gridDim().x == n_blocks
    @assert n_blocks * n_columns_per_block ≥ ntotal_columns
    # Indices
    tv = threadIdx().x # thread index
    vblock_idx = div(tv-1,ᶠNv)+1 # "column index" per block
    if tv > ᶠNv * n_columns_per_block # oob
        return (-1, -1, -1, -1)
    end
    v = mod(tv - 1, ᶠNv) + 1 # block-local column vertical index
    bidx = blockIdx().x # block index
    block_inds = LinearIndices((1:n_blocks, 1:n_columns_per_block)) # linearized block indices
    # if bidx > n_blocks || vblock_idx > n_columns_per_block
    if !(1 ≤ bidx*vblock_idx ≤ length(block_inds))
        return (-1, -1, -1, -1)
    else
        gbi = block_inds[bidx,vblock_idx] # global block index
        CI = CartesianIndices((1:Ni,1:Nj,1:Nh))
        if 1 ≤ gbi ≤ length(CI)
            (i, j, h) = CI[gbi].I
            return (v, i, j, h)
        else
            return (-1, -1, -1, -1)
        end
    end
end

function get_indices_gemini(Ni, Nj, fNv, Nh)
    # Get global thread index
    global_thread_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # Calculate v (1-based)
    v = mod(threadIdx().x - 1, fNv) + 1

    # Calculate the linear index of the column (1-based)
    # Each column requires fNv threads.
    column_linear_idx = fld(global_thread_idx - 1, fNv) + 1

    # Calculate h, j, i from the linear column index (0-based for calculation, then convert to 1-based)
    # The total number of unique (i, j, h) combinations is Ni * Nj * Nh
    # h varies slowest, then j, then i.
    # column_linear_idx is 1-based, so subtract 1 for 0-based calculations
    zero_based_column_idx = column_linear_idx - 1

    # Calculate h (1-based)
    h = fld(zero_based_column_idx, (Ni * Nj)) + 1

    # Calculate remaining index for j and i
    remaining_idx_for_ji = mod(zero_based_column_idx, (Ni * Nj))

    # Calculate j (1-based)
    j = fld(remaining_idx_for_ji, Ni) + 1

    # Calculate i (1-based)
    i = mod(remaining_idx_for_ji, Ni) + 1

    return (v, i, j, h)
end
