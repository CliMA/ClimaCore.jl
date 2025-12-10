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

function single_field_solve!(device::ClimaComms.CUDADevice, cache, x, A, b)
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    us = UniversalSize(Fields.field_values(A))
    mask = Spaces.get_mask(axes(x))
    cart_inds = cartesian_indices_columnwise(us)
    args = (device, cache, x, A, b, us, mask, cart_inds)
    threads = threads_via_occupancy(single_field_solve_kernel!, args)
    nitems = Ni * Nj * Nh
    n_max_threads = min(threads, nitems)
    p = linear_partition(nitems, n_max_threads)
    auto_launch!(
        single_field_solve_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
    call_post_op_callback() && post_op_callback(x, device, cache, x, A, b)
end

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

    # For now, reuse the stable column-wise solver until a numerically robust
    # batched path is available.
    return single_field_solve!(device, cache, x, A, b)
end
