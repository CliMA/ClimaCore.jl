DataLayouts.device_dispatch(x::CUDA.CuArray) = ToCUDA()

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
