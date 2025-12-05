import CUDA
import ClimaComms
import LinearAlgebra: UniformScaling
import ClimaCore.Operators
import ClimaCore.Fields: Field
import ClimaCore.Fields
import ClimaCore.Spaces
import ClimaCore.Topologies
import ClimaCore.MatrixFields
import ClimaCore.DataLayouts: vindex
import ClimaCore.MatrixFields: single_field_solve!
import ClimaCore.MatrixFields: _single_field_solve!
import ClimaCore.MatrixFields: band_matrix_solve!, unzip_tuple_field_values
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
const REGISTER_PATH_MAX_NV = 16

@inline use_register_path(Nv::Integer) = Nv ≤ REGISTER_PATH_MAX_NV

@inline function _band_matrix_solve_tridi_static!(Nv, cache, x, Aⱼs, b)
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
    band_matrix_solve!(
        MatrixFields.TridiagonalMatrixRow,
        cache_local,
        x_local,
        Aⱼs_local,
        b_local,
        identity,
    )
    @inbounds for v in 1:Nv
        x[vi(v)] = x_local[v]
    end
end

@inline function _band_matrix_solve_tridi_stream!(Nv, cache, x, Aⱼs, b)
    Ux, U₊₁ = cache
    A₋₁, A₀, A₊₁ = Aⱼs
    vi = vindex

    @inbounds begin
        inv_D₀ = inv(A₀[vi(1)])
        U₊₁_curr = inv_D₀ ⊠ A₊₁[vi(1)]
        Ux_curr = inv_D₀ ⊠ b[vi(1)]
        Ux[vi(1)] = Ux_curr
        U₊₁[vi(1)] = U₊₁_curr

        for i in 2:Nv
            A₋₁_curr = A₋₁[vi(i)]
            inv_D = inv(A₀[vi(i)] ⊟ A₋₁_curr ⊠ U₊₁_curr)
            Ux_curr = inv_D ⊠ (b[vi(i)] ⊟ A₋₁_curr ⊠ Ux_curr)
            Ux[vi(i)] = Ux_curr
            if i < Nv
                U₊₁_curr = inv_D ⊠ A₊₁[vi(i)]
                U₊₁[vi(i)] = U₊₁_curr
            end
        end

        x[vi(Nv)] = Ux[vi(Nv)]
        for i in (Nv - 1):-1:1
            x[vi(i)] = Ux[vi(i)] ⊟ U₊₁[vi(i)] ⊠ x[vi(i + 1)]
        end
    end
end

function band_matrix_solve_local_mem!(
    t::Type{<:MatrixFields.TridiagonalMatrixRow},
    cache,
    x,
    Aⱼs,
    b,
)
    Nv = DataLayouts.nlevels(x)
    if use_register_path(Nv)
        _band_matrix_solve_tridi_static!(Nv, cache, x, Aⱼs, b)
    else
        _band_matrix_solve_tridi_stream!(Nv, cache, x, Aⱼs, b)
    end
    return nothing
end

@inline function _band_matrix_solve_penta_static!(Nv, cache, x, Aⱼs, b)
    vi = vindex
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
    band_matrix_solve!(
        MatrixFields.PentadiagonalMatrixRow,
        cache_local,
        x_local,
        Aⱼs_local,
        b_local,
        identity,
    )
    @inbounds for v in 1:Nv
        x[vi(v)] = x_local[v]
    end
end

@inline function _band_matrix_solve_penta_stream!(Nv, cache, x, Aⱼs, b)
    vi = vindex
    Ux, U₊₁, U₊₂ = cache
    A₋₂, A₋₁, A₀, A₊₁, A₊₂ = Aⱼs

    @inbounds begin
        inv_D₀ = inv(A₀[vi(1)])
        Ux_i = inv_D₀ ⊠ b[vi(1)]
        U₊₁_i = inv_D₀ ⊠ A₊₁[vi(1)]
        U₊₂_i = inv_D₀ ⊠ A₊₂[vi(1)]
        Ux[vi(1)] = Ux_i
        U₊₁[vi(1)] = U₊₁_i
        U₊₂[vi(1)] = U₊₂_i

        inv_D₀ = inv(A₀[vi(2)] ⊟ A₋₁[vi(2)] ⊠ U₊₁_i)
        Ux_i_prev = Ux_i
        U₊₁_i_prev = U₊₁_i
        U₊₂_i_prev = U₊₂_i
        Ux_i = inv_D₀ ⊠ (b[vi(2)] ⊟ A₋₁[vi(2)] ⊠ Ux_i_prev)
        U₊₁_i = inv_D₀ ⊠ (A₊₁[vi(2)] ⊟ A₋₁[vi(2)] ⊠ U₊₂_i_prev)
        U₊₂_i = inv_D₀ ⊠ A₊₂[vi(2)]
        Ux[vi(2)] = Ux_i
        U₊₁[vi(2)] = U₊₁_i
        U₊₂[vi(2)] = U₊₂_i

        for i in 3:Nv
            L₋₁ = A₋₁[vi(i)] ⊟ A₋₂[vi(i)] ⊠ U₊₁_i_prev
            inv_D₀ = inv(A₀[vi(i)] ⊟ L₋₁ ⊠ U₊₁_i ⊟ A₋₂[vi(i)] ⊠ U₊₂_i_prev)
            Ux_i_prev_prev = Ux_i_prev
            Ux_i_prev = Ux_i
            U₊₁_i_prev = U₊₁_i
            U₊₂_i_prev = U₊₂_i
            Ux_i = inv_D₀ ⊠ (b[vi(i)] ⊟ L₋₁ ⊠ Ux_i_prev ⊟ A₋₂[vi(i)] ⊠ Ux_i_prev_prev)
            if i < Nv
                U₊₁_i = inv_D₀ ⊠ (A₊₁[vi(i)] ⊟ L₋₁ ⊠ U₊₂_i_prev)
            end
            if i < Nv - 1
                U₊₂_i = inv_D₀ ⊠ A₊₂[vi(i)]
            end
            Ux[vi(i)] = Ux_i
            if i < Nv
                U₊₁[vi(i)] = U₊₁_i
            end
            if i < Nv - 1
                U₊₂[vi(i)] = U₊₂_i
            end
        end

        x[vi(Nv)] = Ux[vi(Nv)]
        x[vi(Nv - 1)] = Ux[vi(Nv - 1)] ⊟ U₊₁[vi(Nv - 1)] ⊠ x[vi(Nv)]
        for i in (Nv - 2):-1:1
            x[vi(i)] = Ux[vi(i)] ⊟ U₊₁[vi(i)] ⊠ x[vi(i + 1)] ⊟ U₊₂[vi(i)] ⊠ x[vi(i + 2)]
        end
    end
end

function band_matrix_solve_local_mem!(
    t::Type{<:MatrixFields.PentadiagonalMatrixRow},
    cache,
    x,
    Aⱼs,
    b,
)
    Nv = DataLayouts.nlevels(x)
    if use_register_path(Nv)
        _band_matrix_solve_penta_static!(Nv, cache, x, Aⱼs, b)
    else
        _band_matrix_solve_penta_stream!(Nv, cache, x, Aⱼs, b)
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
