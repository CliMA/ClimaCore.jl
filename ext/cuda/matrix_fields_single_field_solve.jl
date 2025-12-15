import CUDA
import CUDA.CUSOLVER
import CUDA.CUSPARSE
import Base: Csize_t
import ClimaComms
import LinearAlgebra: UniformScaling
import ClimaCore.Operators
import ClimaCore.Fields: Field
import ClimaCore.Fields
import ClimaCore.Spaces
import ClimaCore.Topologies
import ClimaCore.MatrixFields
import ClimaCore.MatrixFields: single_field_solve!, TridiagonalMatrixRow
import ClimaCore.MatrixFields: _single_field_solve!
import ClimaCore.MatrixFields: band_matrix_solve!, unzip_tuple_field_values
import ClimaCore.DataLayouts: vindex, nlevels
import ClimaCore.RecursiveApply: ⊠, ⊞, ⊟, rmap, rzero, rdiv

# Cap threads per block to keep more blocks resident on the GPU.
const SINGLEFIELD_THREADS_CAP = 256

function single_field_solve!(device::ClimaComms.CUDADevice, cache, x, A, b)
    Ni, Nj, _, Nv, Nh = size(Fields.field_values(A))
    us = UniversalSize(Fields.field_values(A))
    mask = Spaces.get_mask(axes(x))

    # Use block/thread layout that promotes memory coalescing:
    # Each thread block handles multiple columns, threads within a block
    # access consecutive memory when loading vertical column data
    nitems = Ni * Nj * Nh

    # Aim for warp-sized or larger blocks for better coalescing
    threads_per_block = min(SINGLEFIELD_THREADS_CAP, nitems)
    blocks = cld(nitems, threads_per_block)

    args = (device, cache, x, A, b, us, mask, Ni, Nj, Nh)
    auto_launch!(
        single_field_solve_kernel_coalesced!,
        args;
        threads_s = threads_per_block,
        blocks_s = blocks,
    )
    call_post_op_callback() && post_op_callback(x, device, cache, x, A, b)
end

function single_field_solve_kernel_coalesced!(device, cache, x, A, b, us, mask, Ni, Nj, Nh)
    # Compute flat thread index across all blocks
    tidx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x

    # Total number of columns
    ncols = Ni * Nj * Nh

    if tidx <= ncols
        # Convert linear index to (i, j, h) using column-major ordering
        # This ensures adjacent threads access adjacent columns in memory
        tidx_0 = tidx - 1  # 0-indexed
        i = (tidx_0 % Ni) + 1
        j = ((tidx_0 ÷ Ni) % Nj) + 1
        h = (tidx_0 ÷ (Ni * Nj)) + 1

        ui = CartesianIndex((i, j, 1, 1, h))
        DataLayouts.should_compute(mask, ui) || return nothing

        _single_field_solve!(
            device,
            Spaces.column(cache, i, j, h),
            Spaces.column(x, i, j, h),
            Spaces.column(A, i, j, h),
            Spaces.column(b, i, j, h),
        )
    end
    return nothing
end

# Keep old kernel for compatibility
function single_field_solve_kernel!(device, cache, x, A, b, us, mask, cart_inds)
    tidx = linear_thread_idx()
    if linear_is_valid_index(tidx, us) && tidx ≤ length(unval(cart_inds))
        I = unval(cart_inds)[tidx]
        (i, j, h) = I.I
        ui = CartesianIndex((i, j, 1, 1, h))
        DataLayouts.should_compute(mask, ui) || return nothing
        _single_field_solve!(
            device,
            Spaces.column(cache, i, j, h),
            Spaces.column(x, i, j, h),
            Spaces.column(A, i, j, h),
            Spaces.column(b, i, j, h),
        )
    end
    return nothing
end

@inline unrolled_unzip_tuple_field_values(data) =
    unrolled_unzip_tuple_field_values(data, propertynames(data))
@inline unrolled_unzip_tuple_field_values(data, pn::Tuple) = (
    getproperty(data, Val(first(pn))),
    unrolled_unzip_tuple_field_values(data, Base.tail(pn))...,
)
@inline unrolled_unzip_tuple_field_values(data, pn::Tuple{Any}) =
    (getproperty(data, Val(first(pn))),)
@inline unrolled_unzip_tuple_field_values(data, pn::Tuple{}) = ()

# TODO: get this working, it doesn't work yet due to InvalidIR
function _single_field_solve_diag_matrix_row!(
    device::ClimaComms.CUDADevice,
    cache,
    x,
    A,
    b,
)
    Aⱼs = unrolled_unzip_tuple_field_values(Fields.field_values(A.entries))
    (A₀,) = Aⱼs
    vi = vindex
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)
    Nv = DataLayouts.nlevels(x_data)
    @inbounds for v in 1:Nv
        x_data[vi(v)] = inv(A₀[vi(v)]) ⊠ b_data[vi(v)]
    end
end

function _single_field_solve!(
    device::ClimaComms.CUDADevice,
    cache::Fields.Field,
    x::Fields.Field,
    A::Fields.Field,
    b::Fields.Field,
)
    if eltype(A) <: MatrixFields.DiagonalMatrixRow
        _single_field_solve_diag_matrix_row!(device, cache, x, A, b)
    else
        band_matrix_solve_local_mem!(
            eltype(A),
            unzip_tuple_field_values(Fields.field_values(cache)),
            Fields.field_values(x),
            unzip_tuple_field_values(Fields.field_values(A.entries)),
            Fields.field_values(b),
        )
    end
end

function _single_field_solve!(
    ::ClimaComms.CUDADevice,
    cache::Fields.ColumnField,
    x::Fields.ColumnField,
    A::UniformScaling,
    b::Fields.ColumnField,
)
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)
    Nv = DataLayouts.nlevels(x_data)
    @inbounds for v in 1:Nv
        x_data[vindex(v)] = inv(A.λ) ⊠ b_data[vindex(v)]
    end
end

function _single_field_solve!(
    ::ClimaComms.CUDADevice,
    cache::Fields.PointDataField,
    x::Fields.PointDataField,
    A::UniformScaling,
    b::Fields.PointDataField,
)
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)
    x_data[] = inv(A.λ) ⊠ b_data[]
end

using StaticArrays: MArray
function band_matrix_solve_local_mem!(
    t::Type{<:MatrixFields.TridiagonalMatrixRow},
    cache,
    x,
    Aⱼs,
    b,
)
    Nv = DataLayouts.nlevels(x)
    Ux, U₊₁ = cache
    A₋₁, A₀, A₊₁ = Aⱼs
    vi = vindex

    Ux_local = MArray{Tuple{Nv}, eltype(Ux)}(undef)
    U₊₁_local = MArray{Tuple{Nv}, eltype(U₊₁)}(undef)
    x_local = MArray{Tuple{Nv}, eltype(x)}(undef)
    A₋₁_local = MArray{Tuple{Nv}, eltype(A₋₁)}(undef)
    A₀_local = MArray{Tuple{Nv}, eltype(A₀)}(undef)
    A₊₁_local = MArray{Tuple{Nv}, eltype(A₊₁)}(undef)
    b_local = MArray{Tuple{Nv}, eltype(b)}(undef)
    @inbounds for v in 1:Nv
        A₋₁_local[v] = A₋₁[vi(v)]
        A₀_local[v] = A₀[vi(v)]
        A₊₁_local[v] = A₊₁[vi(v)]
        b_local[v] = b[vi(v)]
    end
    cache_local = (Ux_local, U₊₁_local)
    Aⱼs_local = (A₋₁_local, A₀_local, A₊₁_local)
    band_matrix_solve!(t, cache_local, x_local, Aⱼs_local, b_local, identity)
    @inbounds for v in 1:Nv
        x[vi(v)] = x_local[v]
    end
    return nothing
end

function band_matrix_solve_local_mem!(
    t::Type{<:MatrixFields.PentadiagonalMatrixRow},
    cache,
    x,
    Aⱼs,
    b,
)
    vi = vindex
    Nv = DataLayouts.nlevels(x)
    Ux, U₊₁, U₊₂ = cache
    A₋₂, A₋₁, A₀, A₊₁, A₊₂ = Aⱼs
    Ux_local = MArray{Tuple{Nv}, eltype(Ux)}(undef)
    U₊₁_local = MArray{Tuple{Nv}, eltype(U₊₁)}(undef)
    U₊₂_local = MArray{Tuple{Nv}, eltype(U₊₂)}(undef)
    x_local = MArray{Tuple{Nv}, eltype(x)}(undef)
    A₋₂_local = MArray{Tuple{Nv}, eltype(A₋₂)}(undef)
    A₋₁_local = MArray{Tuple{Nv}, eltype(A₋₁)}(undef)
    A₀_local = MArray{Tuple{Nv}, eltype(A₀)}(undef)
    A₊₁_local = MArray{Tuple{Nv}, eltype(A₊₁)}(undef)
    A₊₂_local = MArray{Tuple{Nv}, eltype(A₊₂)}(undef)
    b_local = MArray{Tuple{Nv}, eltype(b)}(undef)
    @inbounds for v in 1:Nv
        A₋₂_local[v] = A₋₂[vi(v)]
        A₋₁_local[v] = A₋₁[vi(v)]
        A₀_local[v] = A₀[vi(v)]
        A₊₁_local[v] = A₊₁[vi(v)]
        A₊₂_local[v] = A₊₂[vi(v)]
        b_local[v] = b[vi(v)]
    end
    cache_local = (Ux_local, U₊₁_local, U₊₂_local)
    Aⱼs_local = (A₋₂_local, A₋₁_local, A₀_local, A₊₁_local, A₊₂_local)
    band_matrix_solve!(t, cache_local, x_local, Aⱼs_local, b_local, identity)
    @inbounds for v in 1:Nv
        x[vi(v)] = x_local[v]
    end
    return nothing
end

function band_matrix_solve_local_mem!(
    t::Type{<:MatrixFields.DiagonalMatrixRow},
    cache,
    x,
    Aⱼs,
    b,
)
    Nv = DataLayouts.nlevels(x)
    (A₀,) = Aⱼs
    @inbounds for v in 1:Nv
        x[vindex(v)] = inv(A₀[vindex(v)]) ⊠ b[vindex(v)]
    end
    return nothing
end

# Batched tridiagonal solver using cuSOLVER for improved GPU utilization
# This function solves multiple independent tridiagonal systems in parallel
function batched_tridiag_solve_cusolver!(
    dl::CUDA.CuVector,  # Lower diagonal
    d::CUDA.CuVector,   # Main diagonal
    du::CUDA.CuVector,  # Upper diagonal
    x::CUDA.CuVector,   # Solution/RHS (overwritten with solution)
    batch_size::Int,
    n::Int,
)
    """
    Solve batch_size independent tridiagonal systems using cuSPARSE's
    gtsv2StridedBatch routine (double precision).

    Each system i has the form: A_i * x_i = b_i
    where A_i is tridiagonal and stored in strided format:
    - dl[i*n+1:(i+1)*n] contains lower diagonal (sub-diagonal)
    - d[i*n+1:(i+1)*n] contains main diagonal
    - du[i*n+1:(i+1)*n] contains upper diagonal
    - x[i*n+1:(i+1)*n] contains RHS b_i on input, solution x_i on output

    Stride is n (number of unknowns per system).
    """
    handle = CUDA.CUSPARSE.handle()

    # Stride across batches
    stride = n

    # Workspace size (bytes)
    workspace_bytes = Ref{Csize_t}()
    CUDA.CUSPARSE.cusparseDgtsv2StridedBatch_bufferSizeExt(
        handle,
        n,
        dl,
        d,
        du,
        x,
        batch_size,
        stride,
        workspace_bytes,
    )

    work = CUDA.CuVector{UInt8}(undef, Int(workspace_bytes[]))

    # Solve in-place on x
    CUDA.CUSPARSE.cusparseDgtsv2StridedBatch(
        handle,
        n,
        dl,
        d,
        du,
        x,
        batch_size,
        stride,
        work,
    )

    return nothing
end

# Wrapper for batched tridiagonal solve that integrates with ClimaCore's field structure
function band_matrix_solve_batched_cusolver!(
    t::Type{<:MatrixFields.TridiagonalMatrixRow},
    device::ClimaComms.CUDADevice,
    x::CUDA.CuArray,
    Aⱼs::Tuple,
    b::CUDA.CuArray,
    n_batch::Int,  # Number of independent columns (batch size)
    n::Int,        # Size of each tridiagonal system (number of vertical levels)
)
    """
    Batched solver for tridiagonal matrices using cuSOLVER.
    Converts ClimaCore field data to strided format for cuSOLVER's batched solver.

    Args:
        t: Type indicator (TridiagonalMatrixRow)
        device: CUDA device context
        x: Solution vector (overwritten with result)
        Aⱼs: Tuple of (lower_diag, main_diag, upper_diag) matrix components
        b: Right-hand side vector
        n_batch: Number of independent systems to solve (horizontal columns)
        n: Size of each system (vertical levels)
    """
    A₋₁, A₀, A₊₁ = Aⱼs

    # Create strided arrays for cuSOLVER
    # Reshape data from (vertical_levels, n_columns) to (vertical_levels * n_columns,)
    # with stride = vertical_levels between systems
    dl = CUDA.vec(A₋₁)  # Lower diagonal
    d = CUDA.vec(A₀)    # Main diagonal
    du = CUDA.vec(A₊₁)  # Upper diagonal
    x_vec = CUDA.vec(x) # Solution vector
    b_vec = CUDA.vec(b) # RHS vector

    # Copy RHS to solution vector (cuSOLVER solves in-place)
    copyto!(x_vec, b_vec)

    # Call batched solver
    batched_tridiag_solve_cusolver!(dl, d, du, x_vec, n_batch, n)

    return nothing
end

# Cyclic Reduction kernel for tridiagonal systems
# Uses parallel cyclic reduction algorithm with O(log n) depth
function cyclic_reduction_kernel!(
    a::CUDA.CuDeviceArray{T, 2},  # lower diagonal (Nv, n_batch)
    b::CUDA.CuDeviceArray{T, 2},  # main diagonal (Nv, n_batch)
    c::CUDA.CuDeviceArray{T, 2},  # upper diagonal (Nv, n_batch)
    d::CUDA.CuDeviceArray{T, 2},  # RHS/solution (Nv, n_batch)
    n::Int,                        # system size (must be power of 2 for simplicity)
    n_batch::Int,
) where {T}
    # Each block handles one system
    col_idx = blockIdx().x
    if col_idx > n_batch
        return nothing
    end

    tid = threadIdx().x
    stride = blockDim().x

    # Shared memory for current system
    s_a = CUDA.CuDynamicSharedArray(T, n)
    s_b = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * n)
    s_c = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * 2n)
    s_d = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * 3n)

    # Load data into shared memory
    for i in tid:stride:n
        s_a[i] = a[i, col_idx]
        s_b[i] = b[i, col_idx]
        s_c[i] = c[i, col_idx]
        s_d[i] = d[i, col_idx]
    end
    CUDA.sync_threads()

    # Forward reduction phase
    step = 1
    while step < n
        for i in tid:stride:n
            if i > step && i <= n - step
                k1 = -s_a[i] / s_b[i - step]
                k2 = -s_c[i] / s_b[i + step]

                s_b[i] = s_b[i] + k1 * s_c[i - step] + k2 * s_a[i + step]
                s_d[i] = s_d[i] + k1 * s_d[i - step] + k2 * s_d[i + step]
                s_a[i] = k1 * s_a[i - step]
                s_c[i] = k2 * s_c[i + step]
            end
        end
        CUDA.sync_threads()
        step *= 2
    end

    # Solve the reduced system
    if tid == 1
        s_d[step] = s_d[step] / s_b[step]
    end
    CUDA.sync_threads()

    # Backward substitution phase
    while step > 1
        step = div(step, 2)
        for i in tid:stride:n
            if i > step && i <= n - step && mod(i - 1, 2 * step) == step - 1
                s_d[i] = (s_d[i] - s_a[i] * s_d[i - step] - s_c[i] * s_d[i + step]) / s_b[i]
            end
        end
        CUDA.sync_threads()
    end

    # Write solution back
    for i in tid:stride:n
        d[i, col_idx] = s_d[i]
    end

    return nothing
end

# Parallel Cyclic Reduction (PCR) - more stable variant
function pcr_kernel!(
    a::CUDA.CuDeviceArray{T, 2},  # lower diagonal (Nv, n_batch)
    b::CUDA.CuDeviceArray{T, 2},  # main diagonal (Nv, n_batch)
    c::CUDA.CuDeviceArray{T, 2},  # upper diagonal (Nv, n_batch)
    d::CUDA.CuDeviceArray{T, 2},  # RHS/solution (Nv, n_batch)
    n::Int,
    n_batch::Int,
) where {T}
    col_idx = blockIdx().x
    if col_idx > n_batch
        return nothing
    end

    i = threadIdx().x
    if i > n
        return nothing
    end

    # Shared memory for working arrays
    s_a = CUDA.CuDynamicSharedArray(T, n)
    s_b = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * n)
    s_c = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * 2n)
    s_d = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * 3n)
    s_a2 = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * 4n)
    s_b2 = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * 5n)
    s_c2 = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * 6n)
    s_d2 = CUDA.CuDynamicSharedArray(T, n, sizeof(T) * 7n)

    # Load into shared memory
    s_a[i] = a[i, col_idx]
    s_b[i] = b[i, col_idx]
    s_c[i] = c[i, col_idx]
    s_d[i] = d[i, col_idx]
    CUDA.sync_threads()

    # PCR iterations
    stride = 1
    iterations = ceil(Int, log2(n))

    for iter in 1:iterations
        i_minus = max(i - stride, 1)
        i_plus = min(i + stride, n)

        # Compute elimination factors
        k1 = (i > stride) ? -s_a[i] / s_b[i_minus] : zero(T)
        k2 = (i <= n - stride) ? -s_c[i] / s_b[i_plus] : zero(T)

        # Update coefficients
        s_a2[i] = k1 * s_a[i_minus]
        s_b2[i] = s_b[i] + k1 * s_c[i_minus] + k2 * s_a[i_plus]
        s_c2[i] = k2 * s_c[i_plus]
        s_d2[i] = s_d[i] + k1 * s_d[i_minus] + k2 * s_d[i_plus]

        CUDA.sync_threads()

        # Copy back for next iteration
        s_a[i] = s_a2[i]
        s_b[i] = s_b2[i]
        s_c[i] = s_c2[i]
        s_d[i] = s_d2[i]

        CUDA.sync_threads()
        stride *= 2
    end

    # Final solve
    d[i, col_idx] = s_d[i] / s_b[i]

    return nothing
end

# Wrapper for PCR solver
function cyclic_reduction_solve!(
    x::CUDA.CuArray{T, 2},
    a::CUDA.CuArray{T, 2},
    b::CUDA.CuArray{T, 2},
    c::CUDA.CuArray{T, 2},
    d::CUDA.CuArray{T, 2},
    n::Int,
    n_batch::Int,
) where {T}
    # Use PCR which is more numerically stable
    threads_per_block = min(n, 256)
    n_blocks = n_batch
    shmem_size = 8 * n * sizeof(T)  # 8 arrays of size n

    @cuda threads = threads_per_block blocks = n_blocks shmem = shmem_size pcr_kernel!(
        a, b, c, d, n, n_batch,
    )

    return nothing
end

# Public entry used by BatchedTridiagonalSolve algorithm
function MatrixFields.single_field_solve_batched!(
    cache,
    x::Fields.Field,
    A::Fields.Field,
    b::Fields.Field,
)
    device = ClimaComms.device(x)
    device isa ClimaComms.CUDADevice || error("Batched solver only supports CUDA devices")

    # Fallback if matrix is not tridiagonal
    if !(eltype(A) <: MatrixFields.TridiagonalMatrixRow)
        return single_field_solve!(device, cache, x, A, b)
    end

    # Get field dimensions
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    n_batch = Ni * Nj * Nh
    Nv = DataLayouts.nlevels(Fields.field_values(x))

    # Use cuSPARSE batched solver for tridiagonal systems
    # cuSPARSE's dgtsv2StridedBatch is optimized for exactly this case
    # It handles strided batch storage efficiently in hardware
    try
        solve_tridiag_cusparse_strided_batch!(x, A, b, Nv, n_batch)
        call_post_op_callback() && post_op_callback(x, device, cache, x, A, b)
        return nothing
    catch e
        # If cuSPARSE fails, log and fall back to cyclic reduction
        @warn "cuSPARSE batched solver failed: $e. Falling back to cyclic reduction." maxlog =
            1
        solve_tridiag_cyclic_reduction!(x, A, b, Nv, n_batch)
        call_post_op_callback() && post_op_callback(x, device, cache, x, A, b)
        return nothing
    end
end

"""
    solve_tridiag_cusparse_strided_batch!(x, A, b, Nv, n_batch)

Solve tridiagonal systems using cuSPARSE's optimized batched solver.

This expects data in the following layout:
- For each column (batch element), the vertical levels (Nv) are stored contiguously
- The diagonals are stored as flat vectors where:
  - dl[k*Nv+v] = lower diagonal at level v of column k
  - d[k*Nv+v]  = main diagonal at level v of column k
  - du[k*Nv+v] = upper diagonal at level v of column k
  - b[k*Nv+v]  = RHS at level v of column k
  - x[k*Nv+v]  = solution at level v of column k (output)
"""
function solve_tridiag_cusparse_strided_batch!(x, A, b, Nv::Int, n_batch::Int)
    # Extract field data
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)

    # Get matrix band entries
    Aⱼs = unzip_tuple_field_values(Fields.field_values(A.entries))
    A₋₁, A₀, A₊₁ = Aⱼs

    # Verify shapes match expectations
    # Expected shape: (Nv, n_batch) for banded matrices stored column-wise
    x_shape = size(parent(x_data))
    b_shape = size(parent(b_data))
    A_shape = size(parent(A₀))

    # For column-major storage: first dimension is Nv, but could have extra dims
    # Flatten to (Nv*n_batch,) for cuSPARSE
    n_total = Nv * n_batch

    # Extract and reshape to (Nv, n_batch), then flatten to contiguous vectors
    # We need actual contiguous CuArrays, not views/ReshapedArrays, for cuSPARSE
    a_flat = CUDA.CuArray(reshape(parent(A₋₁), Nv, n_batch))
    b_flat = CUDA.CuArray(reshape(parent(A₀), Nv, n_batch))
    c_flat = CUDA.CuArray(reshape(parent(A₊₁), Nv, n_batch))
    d_flat = CUDA.CuArray(reshape(parent(b_data), Nv, n_batch))
    x_flat = CUDA.CuArray(reshape(parent(x_data), Nv, n_batch))

    # Create flat vectors from the (Nv, n_batch) matrices
    dl_vec = vec(a_flat)  # Lower diagonal (size n_total)
    d_vec = vec(b_flat)   # Main diagonal (size n_total)
    du_vec = vec(c_flat)  # Upper diagonal (size n_total)
    b_vec = vec(d_flat)   # RHS (size n_total)
    x_vec = vec(x_flat)   # Solution (size n_total)

    # Validate vector sizes
    if length(dl_vec) != n_total || length(d_vec) != n_total ||
       length(du_vec) != n_total || length(b_vec) != n_total || length(x_vec) != n_total
        error(
            "Shape mismatch in cuSPARSE batched solver: " *
            "expected size $n_total for all vectors, got " *
            "dl=$(length(dl_vec)), d=$(length(d_vec)), du=$(length(du_vec)), " *
            "b=$(length(b_vec)), x=$(length(x_vec))",
        )
    end

    # Copy RHS to solution vector (cuSPARSE solves in-place)
    copyto!(x_vec, b_vec)

    # Call cuSPARSE batched solver
    # Stride is Nv: each batch element's data is separated by Nv entries
    handle = CUDA.CUSPARSE.handle()

    # Get workspace size
    workspace_bytes = Ref{Csize_t}()
    CUDA.CUSPARSE.cusparseDgtsv2StridedBatch_bufferSizeExt(
        handle,
        Nv,
        dl_vec,
        d_vec,
        du_vec,
        x_vec,
        n_batch,
        Nv,  # stride
        workspace_bytes,
    )

    # Allocate workspace
    work = CUDA.CuVector{UInt8}(undef, Int(workspace_bytes[]))

    # Solve
    CUDA.CUSPARSE.cusparseDgtsv2StridedBatch(
        handle,
        Nv,
        dl_vec,
        d_vec,
        du_vec,
        x_vec,
        n_batch,
        Nv,  # stride
        work,
    )

    # Copy solution back to original x_flat array, then to parent x_data
    copyto!(x_flat, x_vec)
    copyto!(reshape(parent(x_data), Nv, n_batch), x_flat)

    return nothing
end

"""
    solve_tridiag_cyclic_reduction!(x, A, b, Nv, n_batch)

Fallback solver using cyclic reduction (parallel cyclic reduction algorithm).
"""
function solve_tridiag_cyclic_reduction!(x, A, b, Nv::Int, n_batch::Int)
    # For very large systems or if shared memory is insufficient, fall back
    if Nv > 256
        device = ClimaComms.device(x)
        return single_field_solve!(device, nothing, x, A, b)
    end

    # Extract matrix bands
    Aⱼs = unzip_tuple_field_values(Fields.field_values(A.entries))
    A₋₁, A₀, A₊₁ = Aⱼs
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)

    # Reshape to (Nv, n_batch) matrices for kernel processing
    a_matrix = CUDA.CuArray(reshape(parent(A₋₁), Nv, n_batch))
    b_matrix = CUDA.CuArray(reshape(parent(A₀), Nv, n_batch))
    c_matrix = CUDA.CuArray(reshape(parent(A₊₁), Nv, n_batch))
    d_matrix = CUDA.CuArray(reshape(parent(b_data), Nv, n_batch))
    x_matrix = CUDA.CuArray(reshape(parent(x_data), Nv, n_batch))

    # Copy RHS to solution array
    copyto!(x_matrix, d_matrix)

    # Solve using cyclic reduction
    cyclic_reduction_solve!(x_matrix, a_matrix, b_matrix, c_matrix, x_matrix, Nv, n_batch)

    # Copy solution back to original array structure
    copyto!(reshape(parent(x_data), Nv, n_batch), x_matrix)

    return nothing
end
