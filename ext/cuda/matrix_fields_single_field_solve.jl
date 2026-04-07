import CUDA
import ClimaComms
import LinearAlgebra: UniformScaling
import ClimaCore.Operators
import ClimaCore.Fields: Field
import ClimaCore.Fields
import ClimaCore.Spaces
import ClimaCore.Topologies
import ClimaCore.MatrixFields
import ClimaCore.DataLayouts: vindex, universal_size
import ClimaCore.MatrixFields: single_field_solve!
import ClimaCore.MatrixFields: _single_field_solve!
import ClimaCore.MatrixFields: band_matrix_solve!, unzip_tuple_field_values

function single_field_solve!(device::ClimaComms.CUDADevice, cache, x, A, b)
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    us = UniversalSize(Fields.field_values(A))
    mask = Spaces.get_mask(axes(x))
    cart_inds = cartesian_indices_columnwise(us)
    args = (device, cache, x, A, b, us, mask, cart_inds)
    nitems = Ni * Nj * Nh
    (; threads, blocks) = config_via_occupancy(single_field_solve_kernel!, nitems, args)
    auto_launch!(
        single_field_solve_kernel!,
        args;
        threads_s = threads,
        blocks_s = blocks,
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
        x_data[vi(v)] = inv(A₀[vi(v)]) * b_data[vi(v)]
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
        x_data[vindex(v)] = inv(A.λ) * b_data[vindex(v)]
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
    x_data[] = inv(A.λ) * b_data[]
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
        x[vindex(v)] = inv(A₀[vindex(v)]) * b[vindex(v)]
    end
    return nothing
end


function tridiag_pcr_kernel!(
    x, a, b, c, d, ::Val{Nv}, ::Val{n_iter},
) where {Nv, n_iter}
    (idx_i, idx_j, idx_h) = blockIdx()
    i = threadIdx().x
    if i > Nv
        return nothing
    end

    s_a = CUDA.CuStaticSharedArray(eltype(a), Nv)
    s_b = CUDA.CuStaticSharedArray(eltype(b), Nv)
    s_c = CUDA.CuStaticSharedArray(eltype(c), Nv)
    s_d = CUDA.CuStaticSharedArray(eltype(d), Nv)

    idx = CartesianIndex(idx_i, idx_j, 1, i, idx_h)

    # Load into shared memory
    @inbounds begin
        local_ai = a[idx]
        local_bi = b[idx]
        local_ci = c[idx]
        local_di = d[idx]

        s_a[i] = local_ai
        s_b[i] = local_bi
        s_c[i] = local_ci
        s_d[i] = local_di
    end
    CUDA.sync_threads()

    # PCR iterations
    stride = 1

    for _ in 1:n_iter
        i_minus = max(i - stride, 1)
        i_plus = min(i + stride, Nv)

        # Compute elimination factors
        @inbounds begin
            k1 = (i > stride) ? -local_ai ⊠ inv(s_b[i_minus]) : zero(eltype(a))
            k2 = (i <= Nv - stride) ? -local_ci ⊠ inv(s_b[i_plus]) : zero(eltype(a))

            # Update coefficients
            local_ai = k1 ⊠ s_a[i_minus]
            local_bi = local_bi ⊞ k1 ⊠ s_c[i_minus] ⊞ k2 ⊠ s_a[i_plus]
            local_ci = k2 ⊠ s_c[i_plus]
            local_di = local_di ⊞ k1 ⊠ s_d[i_minus] ⊞ k2 ⊠ s_d[i_plus]
        end

        CUDA.sync_threads()

        # Copy back for next iteration
        @inbounds begin
            s_a[i] = local_ai
            s_b[i] = local_bi
            s_c[i] = local_ci
            s_d[i] = local_di
        end

        CUDA.sync_threads()
        stride *= 2
    end

    #  Final solve into x
    @inbounds x[idx] = inv(s_b[i]) ⊠ s_d[i]
    return nothing
end


"""
    single_field_solve_tridiagonal!(cache, x, A, b)

Specialized solver for the tridiagonal MatrixField. Solves each column in
parallel launching Nv threads per block where Nv is the number of vertical levels.
Works best if Nv is multiple of 32. There is an upper limit on the size of Nv
due to resource limits of the GPU (register and shared memory usage). For
A100 it is 1024, but may differ depending on the hardware.
"""
function single_field_solve_tridiagonal!(cache, x, A, b)

    device = ClimaComms.device(x)
    device isa ClimaComms.CUDADevice || error("This solver supports only CUDA devices.")

    eltype(A) <: MatrixFields.TridiagonalMatrixRow || error(
        "This function expects a tridiagonal matrix field, but got a field with element type $(eltype(A))",
    )

    # Get field dimensions
    Ni, Nj, _, Nv, Nh = universal_size(Fields.field_values(A))

    # Prepare data
    Aⱼs = unzip_tuple_field_values(Fields.field_values(A.entries))
    A₋₁, A₀, A₊₁ = Aⱼs
    x_data = Fields.field_values(x)
    b_data = Fields.field_values(b)

    # Solve
    threads_per_block = Nv
    n_iter = ceil(Int, log2(Nv))
    args = (x_data, A₋₁, A₀, A₊₁, b_data, Val(Nv), Val(n_iter))

    auto_launch!(
        tridiag_pcr_kernel!,
        args;
        threads_s = (threads_per_block,),
        blocks_s = (Ni, Nj, Nh),
    )

    call_post_op_callback() && post_op_callback(x, device, cache, x, A, b)
    return nothing
end
