DataLayouts.device_dispatch(x::CUDA.CuArray) = ToCUDA()

##### Multi-dimensional launch configuration kernels

function knl_copyto!(dest, src)

    i = CUDA.threadIdx().x
    j = CUDA.threadIdx().y

    h = CUDA.blockIdx().x
    v = CUDA.blockDim().z * (CUDA.blockIdx().y - 1) + CUDA.threadIdx().z

    if v <= size(dest, 4)
        I = CartesianIndex((i, j, 1, v, h))
        @inbounds dest[I] = src[I]
    end
    return nothing
end

function Base.copyto!(
    dest::IJFH{S, Nij, Nh},
    bc::DataLayouts.BroadcastedUnionIJFH{S, Nij, Nh},
    ::ToCUDA,
) where {S, Nij, Nh}
    if Nh > 0
        auto_launch!(
            knl_copyto!,
            (dest, bc);
            threads_s = (Nij, Nij),
            blocks_s = (Nh, 1),
        )
    end
    return dest
end

function Base.copyto!(
    dest::VIJFH{S, Nv, Nij, Nh},
    bc::DataLayouts.BroadcastedUnionVIJFH{S, Nv, Nij, Nh},
    ::ToCUDA,
) where {S, Nv, Nij, Nh}
    if Nv > 0 && Nh > 0
        Nv_per_block = min(Nv, fld(256, Nij * Nij))
        Nv_blocks = cld(Nv, Nv_per_block)
        auto_launch!(
            knl_copyto!,
            (dest, bc);
            threads_s = (Nij, Nij, Nv_per_block),
            blocks_s = (Nh, Nv_blocks),
        )
    end
    return dest
end

import ClimaCore.DataLayouts: isascalar
function knl_copyto_flat!(dest::AbstractData, bc, us)
    @inbounds begin
        tidx = thread_index()
        if tidx ≤ get_N(us)
            n = size(dest)
            I = kernel_indexes(tidx, n)
            dest[I] = bc[I]
        end
    end
    return nothing
end

function cuda_copyto!(dest::AbstractData, bc)
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
    us = DataLayouts.UniversalSize(dest)
    if Nv > 0 && Nh > 0
        nitems = prod(DataLayouts.universal_size(dest))
        auto_launch!(knl_copyto_flat!, (dest, bc, us), nitems; auto = true)
    end
    return dest
end
Base.copyto!(
    dest::IFH{S, Ni, Nh},
    bc::DataLayouts.BroadcastedUnionIFH{S, Ni, Nh},
    ::ToCUDA,
) where {S, Ni, Nh} = cuda_copyto!(dest, bc)
Base.copyto!(
    dest::VIFH{S, Nv, Ni, Nh},
    bc::DataLayouts.BroadcastedUnionVIFH{S, Nv, Ni, Nh},
    ::ToCUDA,
) where {S, Nv, Ni, Nh} = cuda_copyto!(dest, bc)
#####

function knl_copyto!(dest, src, us)
    I = universal_index(dest)
    if is_valid_index(dest, I, us)
        @inbounds dest[I] = src[I]
    end
    return nothing
end

function knl_copyto_linear!(dest, src, us)
    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
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
    function Base.copyto!(dest::AbstractData, bc, ::ToCUDA)
        (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
        us = DataLayouts.UniversalSize(dest)
        if Nv > 0 && Nh > 0
            args = (dest, bc, us)
            threads = threads_via_occupancy(knl_copyto!, args)
            n_max_threads = min(threads, get_N(us))
            p = partition(dest, n_max_threads)
            auto_launch!(
                knl_copyto!,
                args;
                threads_s = p.threads,
                blocks_s = p.blocks,
            )
        end
        return dest
    end
else
    function Base.copyto!(dest::AbstractData, bc, ::ToCUDA)
        (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
        us = DataLayouts.UniversalSize(dest)
        if Nv > 0 && Nh > 0
            if DataLayouts.has_uniform_datalayouts(bc) &&
               dest isa DataLayouts.EndsWithField
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
                args = (dest, bc, us)
                threads = threads_via_occupancy(knl_copyto!, args)
                n_max_threads = min(threads, get_N(us))
                p = partition(dest, n_max_threads)
                auto_launch!(
                    knl_copyto!,
                    args;
                    threads_s = p.threads,
                    blocks_s = p.blocks,
                )
            end
        end
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
    ::ToCUDA,
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
}
    bc = Base.Broadcast.instantiate(
        Base.Broadcast.Broadcasted{Style}(bc.f, bc.args, ()),
    )
    @inbounds bc0 = bc[]
    fill!(dest, bc0)
end

# For field-vector operations
function DataLayouts.copyto_per_field!(
    array::AbstractArray,
    bc::Union{AbstractArray, Base.Broadcast.Broadcasted},
    ::ToCUDA,
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
    ::ToCUDA,
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
    return array
end
function DataLayouts.copyto_per_field_scalar!(
    array::AbstractArray,
    bc::Real,
    ::ToCUDA,
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
    return array
end
function copyto_per_field_kernel_0D!(array, bc, N)
    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if 1 ≤ i ≤ N
        @inbounds array[i] = bc[]
    end
    return nothing
end
