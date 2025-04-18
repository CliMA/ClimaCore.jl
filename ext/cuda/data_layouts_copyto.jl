DataLayouts.device_dispatch(x::CUDA.CuArray) = ToCUDA()

function knl_copyto!(dest, src, us, mask, cart_inds)
    tidx = linear_thread_idx()
    if linear_is_valid_index(tidx, us) && tidx ≤ length(unval(cart_inds))
        I = if mask isa NoMask
            unval(cart_inds)[tidx]
        else
            masked_universal_index(mask, cart_inds)
        end
        @inbounds dest[I] = src[I]
    end
    return nothing
end

function knl_copyto_linear!(dest, src, us)
    i = linear_thread_idx()
    if linear_is_valid_index(i, us)
        @inbounds dest[i] = src[i]
    end
    return nothing
end

if VERSION ≥ v"1.11.0-beta"
    # https://github.com/JuliaLang/julia/issues/56295
    # Julia 1.11's Base.Broadcast currently requires
    # multiple integer indexing, wheras Julia 1.10 did not.
    # This means that we cannot reserve linear indexing to
    # special-case fixes for https://github.com/JuliaLang/julia/issues/28126
    # (including the GPU-variant related issue resolution efforts:
    # JuliaGPU/GPUArrays.jl#454, JuliaGPU/GPUArrays.jl#464).
    function Base.copyto!(dest::AbstractData, bc, to::ToCUDA, mask = NoMask())
        (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
        us = DataLayouts.UniversalSize(dest)
        if Nv > 0 && Nh > 0
            cart_inds = if mask isa NoMask
                cartesian_indices(us)
            else
                cartesian_indicies_mask(us, mask)
            end
            args = (dest, bc, us, mask, cart_inds)
            threads = threads_via_occupancy(knl_copyto!, args)
            n_max_threads = min(threads, get_N(us))
            p = if mask isa NoMask
                linear_partition(prod(size(dest)), n_max_threads)
            else
                masked_partition(mask, n_max_threads, us)
            end
            auto_launch!(
                knl_copyto!,
                args;
                threads_s = p.threads,
                blocks_s = p.blocks,
            )
        end
        call_post_op_callback() && post_op_callback(dest, dest, bc, to, mask)
        return dest
    end
else
    function Base.copyto!(dest::AbstractData, bc, to::ToCUDA, mask = NoMask())
        (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
        us = DataLayouts.UniversalSize(dest)
        if Nv > 0 && Nh > 0
            if DataLayouts.has_uniform_datalayouts(bc) &&
               dest isa DataLayouts.EndsWithField &&
               mask isa NoMask
                bc′ = Base.Broadcast.instantiate(
                    DataLayouts.to_non_extruded_broadcasted(bc),
                )
                args = (dest, bc′, us)
                threads = threads_via_occupancy(knl_copyto_linear!, args)
                n_max_threads = min(threads, get_N(us))
                p = linear_partition(prod(size(dest)), n_max_threads)
                auto_launch!(
                    knl_copyto_linear!,
                    args;
                    threads_s = p.threads,
                    blocks_s = p.blocks,
                )
            else
                cart_inds = if mask isa NoMask
                    cartesian_indices(us)
                else
                    cartesian_indicies_mask(us, mask)
                end
                args = (dest, bc, us, mask, cart_inds)
                threads = threads_via_occupancy(knl_copyto!, args)
                n_max_threads = min(threads, get_N(us))
                p = if mask isa NoMask
                    linear_partition(prod(size(dest)), n_max_threads)
                else
                    masked_partition(mask, n_max_threads, us)
                end
                auto_launch!(
                    knl_copyto!,
                    args;
                    threads_s = p.threads,
                    blocks_s = p.blocks,
                )
            end
        end
        call_post_op_callback() && post_op_callback(dest, dest, bc, to, mask)
        return dest
    end
end

# broadcasting scalar assignment
# Performance optimization for the common identity scalar case: dest .= val
# And this is valid for the CPU or GPU, since the broadcasted object
# is a scalar type.
function Base.copyto!(
    dest::AbstractData,
    bc::Base.Broadcast.Broadcasted{Style},
    to::ToCUDA,
    mask = NoMask(),
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
}
    bc = Base.Broadcast.instantiate(
        Base.Broadcast.Broadcasted{Style}(bc.f, bc.args, ()),
    )
    @inbounds bc0 = bc[]
    fill!(dest, bc0, mask)
    call_post_op_callback() && post_op_callback(dest, dest, bc, to, mask)
end

# For field-vector operations
function DataLayouts.copyto_per_field!(
    array::AbstractArray,
    bc::Union{AbstractArray, Base.Broadcast.Broadcasted},
    to::ToCUDA,
)
    bc′ = DataLayouts.to_non_extruded_broadcasted(bc)
    # All field variables are treated separately, so
    # we can parallelize across the field index, and
    # leverage linear indexing:
    nitems = prod(size(array))
    N = prod(size(array))
    args = (array, bc′, N)
    threads = threads_via_occupancy(copyto_per_field_kernel!, args)
    n_max_threads = min(threads, nitems)
    p = linear_partition(nitems, n_max_threads)
    auto_launch!(
        copyto_per_field_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
    call_post_op_callback() && post_op_callback(array, array, bc, to)
    return array
end
function copyto_per_field_kernel!(array, bc, N)
    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if 1 ≤ i ≤ N
        @inbounds array[i] = bc[i]
    end
    return nothing
end

# Need 2 methods here to avoid unbound arguments:
function DataLayouts.copyto_per_field_scalar!(
    array::AbstractArray,
    bc::Base.Broadcast.Broadcasted{Style},
    to::ToCUDA,
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
}
    bc′ = DataLayouts.to_non_extruded_broadcasted(bc)
    # All field variables are treated separately, so
    # we can parallelize across the field index, and
    # leverage linear indexing:
    nitems = prod(size(array))
    N = prod(size(array))
    args = (array, bc′, N)
    threads = threads_via_occupancy(copyto_per_field_kernel_0D!, args)
    n_max_threads = min(threads, nitems)
    p = linear_partition(nitems, n_max_threads)
    auto_launch!(
        copyto_per_field_kernel_0D!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
    call_post_op_callback() && post_op_callback(array, array, bc, to)
    return array
end
function DataLayouts.copyto_per_field_scalar!(
    array::AbstractArray,
    bc::Real,
    to::ToCUDA,
)
    bc′ = DataLayouts.to_non_extruded_broadcasted(bc)
    # All field variables are treated separately, so
    # we can parallelize across the field index, and
    # leverage linear indexing:
    nitems = prod(size(array))
    N = prod(size(array))
    args = (array, bc′, N)
    threads = threads_via_occupancy(copyto_per_field_kernel_0D!, args)
    n_max_threads = min(threads, nitems)
    p = linear_partition(nitems, n_max_threads)
    auto_launch!(
        copyto_per_field_kernel_0D!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
    call_post_op_callback() && post_op_callback(array, array, bc, to)
    return array
end
function copyto_per_field_kernel_0D!(array, bc, N)
    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if 1 ≤ i ≤ N
        @inbounds array[i] = bc[]
    end
    return nothing
end
