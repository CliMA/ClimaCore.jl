import ClimaCore: Topologies, Spaces, Fields
import ClimaComms
import CUDA
using CUDA: @cuda
import ClimaCore.Remapping: _set_interpolated_values_device!, bilinear, linear

function ClimaCore.Remapping._set_interpolated_values_bilinear!(
    out::CUDA.CuArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_corners,
    local_horiz_indices,
    vert_interpolation_weights::AbstractArray,
    vert_bounding_indices::AbstractArray,
    local_bilinear_s,
    ::Nothing,
    local_bilinear_i,
    ::Nothing,
)
    field_values = tuple(map(f -> Fields.field_values(f), fields)...)
    num_horiz = length(local_horiz_indices)
    num_vert = length(vert_bounding_indices)
    num_fields = length(field_values)
    nitems = length(out)
    args = (
        out,
        local_horiz_indices,
        local_bilinear_s,
        local_bilinear_i,
        vert_interpolation_weights,
        vert_bounding_indices,
        field_values,
    )
    threads = threads_via_occupancy(set_interpolated_values_linear_2d_kernel!, args)
    p = linear_partition(nitems, threads)
    auto_launch!(
        set_interpolated_values_linear_2d_kernel!,
        args;
        threads_s = (p.threads,),
        blocks_s = (p.blocks,),
    )
end

function set_interpolated_values_linear_2d_kernel!(
    out,
    local_horiz_indices,
    local_bilinear_s,
    local_bilinear_i,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
)
    num_horiz = length(local_horiz_indices)
    num_vert = length(vert_bounding_indices)
    num_fields = length(field_values)
    inds = (num_horiz, num_vert, num_fields)
    i_thread = thread_index()
    1 ≤ i_thread ≤ prod(inds) || return nothing
    (i_out, j_v, k) = CartesianIndices(map(x -> Base.OneTo(x), inds))[i_thread].I
    @inbounds begin
        CI = CartesianIndex
        h = local_horiz_indices[i_out]
        v_lo, v_hi = vert_bounding_indices[j_v]
        A, B = vert_interpolation_weights[j_v]
        s, ii = local_bilinear_s[i_out], local_bilinear_i[i_out]
        fvals = field_values[k]
        out[i_out, j_v, k] =
            A * linear(fvals[CI(ii, 1, 1, v_lo, h)], fvals[CI(ii + 1, 1, 1, v_lo, h)], s) +
            B * linear(fvals[CI(ii, 1, 1, v_hi, h)], fvals[CI(ii + 1, 1, 1, v_hi, h)], s)
    end
    return nothing
end

function ClimaCore.Remapping._set_interpolated_values_bilinear!(
    out::CUDA.CuArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_corners,
    local_horiz_indices,
    vert_interpolation_weights::AbstractArray,
    vert_bounding_indices::AbstractArray,
    local_bilinear_s,
    local_bilinear_t,
    local_bilinear_i,
    local_bilinear_j,
)
    field_values = tuple(map(f -> Fields.field_values(f), fields)...)
    num_horiz = length(local_horiz_indices)
    num_vert = length(vert_bounding_indices)
    num_fields = length(field_values)
    nitems = length(out)
    args = (
        out,
        local_horiz_indices,
        local_bilinear_s,
        local_bilinear_t,
        local_bilinear_i,
        local_bilinear_j,
        vert_interpolation_weights,
        vert_bounding_indices,
        field_values,
    )
    threads = threads_via_occupancy(set_interpolated_values_bilinear_3d_kernel!, args)
    p = linear_partition(nitems, threads)
    auto_launch!(
        set_interpolated_values_bilinear_3d_kernel!,
        args;
        threads_s = (p.threads,),
        blocks_s = (p.blocks,),
    )
end

function set_interpolated_values_bilinear_3d_kernel!(
    out,
    local_horiz_indices,
    local_bilinear_s,
    local_bilinear_t,
    local_bilinear_i,
    local_bilinear_j,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
)
    num_horiz = length(local_horiz_indices)
    num_vert = length(vert_bounding_indices)
    num_fields = length(field_values)
    inds = (num_horiz, num_vert, num_fields)
    i_thread = thread_index()
    1 ≤ i_thread ≤ prod(inds) || return nothing
    (i_out, j_v, k) = CartesianIndices(map(x -> Base.OneTo(x), inds))[i_thread].I
    @inbounds begin
        CI = CartesianIndex
        h = local_horiz_indices[i_out]
        v_lo, v_hi = vert_bounding_indices[j_v]
        A, B = vert_interpolation_weights[j_v]
        s = local_bilinear_s[i_out]
        t = local_bilinear_t[i_out]
        ii = local_bilinear_i[i_out]
        jj = local_bilinear_j[i_out]
        fvals = field_values[k]
        # Horizontal bilinear at v_lo (level by level), then at v_hi, then vertical blend
        c11_lo = fvals[CI(ii, jj, 1, v_lo, h)]
        c21_lo = fvals[CI(ii + 1, jj, 1, v_lo, h)]
        c22_lo = fvals[CI(ii + 1, jj + 1, 1, v_lo, h)]
        c12_lo = fvals[CI(ii, jj + 1, 1, v_lo, h)]
        f_lo = bilinear(c11_lo, c21_lo, c22_lo, c12_lo, s, t)
        c11_hi = fvals[CI(ii, jj, 1, v_hi, h)]
        c21_hi = fvals[CI(ii + 1, jj, 1, v_hi, h)]
        c22_hi = fvals[CI(ii + 1, jj + 1, 1, v_hi, h)]
        c12_hi = fvals[CI(ii, jj + 1, 1, v_hi, h)]
        f_hi = bilinear(c11_hi, c21_hi, c22_hi, c12_hi, s, t)
        out[i_out, j_v, k] = A * f_lo + B * f_hi
    end
    return nothing
end

# 1D linear: horizontal-only.
function ClimaCore.Remapping._set_interpolated_values_bilinear!(
    out::CUDA.CuArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_corners,
    local_horiz_indices,
    ::Nothing,
    ::Nothing,
    local_bilinear_s,
    ::Nothing,
    local_bilinear_i,
    ::Nothing,
)
    field_values = tuple(map(f -> Fields.field_values(f), fields)...)
    num_horiz = length(local_horiz_indices)
    num_fields = length(field_values)
    nitems = length(out)
    args = (
        out,
        local_horiz_indices,
        local_bilinear_s,
        local_bilinear_i,
        field_values,
    )
    threads = threads_via_occupancy(set_interpolated_values_linear_1d_kernel!, args)
    p = linear_partition(nitems, threads)
    auto_launch!(
        set_interpolated_values_linear_1d_kernel!,
        args;
        threads_s = (p.threads,),
        blocks_s = (p.blocks,),
    )
end

function set_interpolated_values_linear_1d_kernel!(
    out,
    local_horiz_indices,
    local_bilinear_s,
    local_bilinear_i,
    field_values,
)
    num_horiz = length(local_horiz_indices)
    num_fields = length(field_values)
    inds = (num_horiz, num_fields)
    i_thread = thread_index()
    1 ≤ i_thread ≤ prod(inds) || return nothing
    (i_out, k) = CartesianIndices(map(x -> Base.OneTo(x), inds))[i_thread].I
    @inbounds begin
        CI = CartesianIndex
        h, s, ii =
            local_horiz_indices[i_out], local_bilinear_s[i_out], local_bilinear_i[i_out]
        fvals = field_values[k]
        out[i_out, k] = linear(fvals[CI(ii, 1, 1, 1, h)], fvals[CI(ii + 1, 1, 1, 1, h)], s)
    end
    return nothing
end

function ClimaCore.Remapping._set_interpolated_values_bilinear!(
    out::CUDA.CuArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_corners,
    local_horiz_indices,
    ::Nothing,
    ::Nothing,
    local_bilinear_s,
    local_bilinear_t,
    local_bilinear_i,
    local_bilinear_j,
)
    field_values = tuple(map(f -> Fields.field_values(f), fields)...)
    num_horiz = length(local_horiz_indices)
    num_fields = length(field_values)
    nitems = length(out)
    args = (
        out,
        local_horiz_indices,
        local_bilinear_s,
        local_bilinear_t,
        local_bilinear_i,
        local_bilinear_j,
        field_values,
    )
    threads = threads_via_occupancy(set_interpolated_values_bilinear_2d_kernel!, args)
    p = linear_partition(nitems, threads)
    auto_launch!(
        set_interpolated_values_bilinear_2d_kernel!,
        args;
        threads_s = (p.threads,),
        blocks_s = (p.blocks,),
    )
end

function set_interpolated_values_bilinear_2d_kernel!(
    out,
    local_horiz_indices,
    local_bilinear_s,
    local_bilinear_t,
    local_bilinear_i,
    local_bilinear_j,
    field_values,
)
    num_horiz = length(local_horiz_indices)
    num_fields = length(field_values)
    inds = (num_horiz, num_fields)
    i_thread = thread_index()
    1 ≤ i_thread ≤ prod(inds) || return nothing
    (i_out, k) = CartesianIndices(map(x -> Base.OneTo(x), inds))[i_thread].I
    @inbounds begin
        CI = CartesianIndex
        h = local_horiz_indices[i_out]
        s = local_bilinear_s[i_out]
        t = local_bilinear_t[i_out]
        ii = local_bilinear_i[i_out]
        jj = local_bilinear_j[i_out]
        fvals = field_values[k]
        # Four nodes of 2-point cell: (ii,jj), (ii+1,jj), (ii+1,jj+1), (ii,jj+1)
        c11 = fvals[CI(ii, jj, 1, 1, h)]
        c21 = fvals[CI(ii + 1, jj, 1, 1, h)]
        c22 = fvals[CI(ii + 1, jj + 1, 1, 1, h)]
        c12 = fvals[CI(ii, jj + 1, 1, 1, h)]
        out[i_out, k] = bilinear(c11, c21, c22, c12, s, t)
    end
    return nothing
end


function _set_interpolated_values_device!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_field_values,
    local_horiz_indices,
    interpolation_matrix,
    vert_interpolation_weights::AbstractArray,
    vert_bounding_indices::AbstractArray,
    dev::ClimaComms.CUDADevice,
)
    # FIXME: Avoid allocation of tuple
    field_values = tuple(map(f -> Fields.field_values(f), fields)...)

    purely_vert_space = isnothing(interpolation_matrix)
    nitems = length(out)

    # For purely vertical spaces, `Nq` is not used, so we pass in -1 here.
    _, Nq = purely_vert_space ? (-1, -1) : size(interpolation_matrix[1])
    args = (
        out,
        interpolation_matrix,
        local_horiz_indices,
        vert_interpolation_weights,
        vert_bounding_indices,
        field_values,
        Val(Nq),
    )
    threads = threads_via_occupancy(set_interpolated_values_kernel!, args)
    p = linear_partition(nitems, threads)
    auto_launch!(
        set_interpolated_values_kernel!,
        args;
        threads_s = (p.threads),
        blocks_s = (p.blocks),
    )
    call_post_op_callback() && post_op_callback(
        out,
        out,
        fields,
        scratch_field_values,
        local_horiz_indices,
        interpolation_matrix,
        vert_interpolation_weights,
        vert_bounding_indices,
        dev,
    )
end

# GPU, 3D case
function set_interpolated_values_kernel!(
    out::AbstractArray,
    (I1, I2)::NTuple{2},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
    ::Val{Nq},
) where {Nq}
    num_horiz = length(local_horiz_indices)
    num_vert = length(vert_bounding_indices)
    num_fields = length(field_values)

    @inbounds begin
        i_thread = thread_index()
        inds = (num_vert, num_horiz, num_fields)

        1 ≤ i_thread ≤ prod(inds) || return nothing

        # TODO: Check the memory access pattern, we should maximize coalesced memory
        (j, i, k) = CartesianIndices(map(x -> Base.OneTo(x), inds))[i_thread].I

        CI = CartesianIndex
        h = local_horiz_indices[i]
        v_lo, v_hi = vert_bounding_indices[j]
        A, B = vert_interpolation_weights[j]
        fvals = field_values[k]
        out[i, j, k] = 0
        for t in 1:Nq, s in 1:Nq
            out[i, j, k] +=
                I1[i, t] *
                I2[i, s] *
                (
                    A * fvals[CI(t, s, 1, v_lo, h)] +
                    B * fvals[CI(t, s, 1, v_hi, h)]
                )
        end
    end
    return nothing
end

# GPU, 2D case
function set_interpolated_values_kernel!(
    out::AbstractArray,
    (I,)::NTuple{1},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
    ::Val{Nq},
) where {Nq}
    # TODO: Check the memory access pattern. This was not optimized and likely inefficient!
    num_horiz = length(local_horiz_indices)
    num_vert = length(vert_bounding_indices)
    num_fields = length(field_values)

    @inbounds begin
        i_thread = thread_index()
        inds = (num_vert, num_horiz, num_fields)

        1 ≤ i_thread ≤ prod(inds) || return nothing

        # TODO: Check the memory access pattern, we should maximize coalesced memory
        (j, i, k) = CartesianIndices(map(x -> Base.OneTo(x), inds))[i_thread].I

        CI = CartesianIndex
        h = local_horiz_indices[i]
        v_lo, v_hi = vert_bounding_indices[j]
        A, B = vert_interpolation_weights[j]
        out[i, j, k] = 0
        for t in 1:Nq
            out[i, j, k] +=
                I[i, t] * (
                    A * field_values[k][CI(t, 1, 1, v_lo, h)] +
                    B * field_values[k][CI(t, 1, 1, v_hi, h)]
                )
        end
    end
    return nothing
end

# GPU, vertical case
function set_interpolated_values_kernel!(
    out::AbstractArray,
    ::Nothing,
    ::Nothing,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
    ::Val{Nq},
) where {Nq}
    # TODO: Check the memory access pattern. This was not optimized and likely inefficient!
    num_fields = length(field_values)
    num_vert = length(vert_bounding_indices)

    @inbounds begin
        i_thread = thread_index()
        inds = (num_vert, num_fields)

        1 ≤ i_thread ≤ prod(inds) || return nothing

        # TODO: Check the memory access pattern, we should maximize coalesced memory
        (j, k) = CartesianIndices(map(x -> Base.OneTo(x), inds))[i_thread].I

        CI = CartesianIndex
        v_lo, v_hi = vert_bounding_indices[j]
        A, B = vert_interpolation_weights[j]
        out[j, k] = (
            A * field_values[k][CI(1, 1, 1, v_lo, 1)] +
            B * field_values[k][CI(1, 1, 1, v_hi, 1)]
        )
    end
    return nothing
end

function _set_interpolated_values_device!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    _scratch_field_values,
    local_horiz_indices,
    local_horiz_interpolation_weights,
    ::Nothing,
    ::Nothing,
    ::ClimaComms.CUDADevice,
)

    space = axes(first(fields))
    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quad)

    hdims = length(local_horiz_interpolation_weights)
    hdims in (1, 2) || error("Cannot handle $hdims horizontal dimensions")

    # FIXME: Avoid allocation of tuple
    field_values = tuple(map(f -> Fields.field_values(f), fields)...)
    nitems = length(out)

    args = (
        out,
        local_horiz_interpolation_weights,
        local_horiz_indices,
        field_values,
        Val(Nq),
    )
    threads = threads_via_occupancy(set_interpolated_values_kernel!, args)
    p = linear_partition(nitems, threads)
    auto_launch!(
        set_interpolated_values_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
end

function set_interpolated_values_kernel!(
    out,
    (I1, I2)::NTuple{2},
    local_horiz_indices,
    field_values,
    ::Val{Nq},
) where {Nq}
    # TODO: Check the memory access pattern. This was not optimized and likely inefficient!
    num_horiz = length(local_horiz_indices)
    num_fields = length(field_values)

    @inbounds begin
        i_thread = thread_index()
        inds = (num_horiz, num_fields)

        1 ≤ i_thread ≤ prod(inds) || return nothing

        # TODO: Check the memory access pattern, we should maximize coalesced memory
        (i, k) = CartesianIndices(map(x -> Base.OneTo(x), inds))[i_thread].I

        h = local_horiz_indices[i]
        out[i, k] = 0
        for t in 1:Nq, s in 1:Nq
            out[i, k] +=
                I1[i, t] *
                I2[i, s] *
                field_values[k][CartesianIndex(t, s, 1, 1, h)]
        end
    end
    return nothing
end

function set_interpolated_values_kernel!(
    out::AbstractArray,
    (I,)::NTuple{1},
    local_horiz_indices,
    field_values,
    ::Val{Nq},
) where {Nq}
    # TODO: Check the memory access pattern. This was not optimized and likely inefficient!
    num_horiz = length(local_horiz_indices)
    num_fields = length(field_values)

    @inbounds begin
        i_thread = thread_index()
        inds = (num_horiz, num_fields)

        1 ≤ i_thread ≤ prod(inds) || return nothing

        # TODO: Check the memory access pattern, we should maximize coalesced memory
        (i, k) = CartesianIndices(map(x -> Base.OneTo(x), inds))[i_thread].I

        h = local_horiz_indices[i]
        out[i, k] = 0
        for t in 1:Nq
            out[i, k] +=
                I[i, t] * field_values[k][CartesianIndex(t, 1, 1, 1, h)]
        end
    end
    return nothing
end
