import CUDA
#####
##### CPU column
#####

#= e.g., any 2nd order interpolation / derivative operator =#
function column_op_2mul_1add!(x, y, D, U)
    y1 = @view y[1:(end - 1)]
    y2 = @view y[2:end]
    @inbounds for i in eachindex(x)
        x[i] = D[i] * y1[i] + U[i] * y2[i]
    end
    return nothing
end

#= e.g., div(grad(scalar)), div(interp(vec)) =#
function column_op_3mul_2add!(x, y, L, D, U)
    y1 = @view y[1:(end - 1)]
    y2 = @view y[2:(end - 1)]
    y3 = @view y[2:end]
    @inbounds for i in eachindex(x)
        i == 1 && continue
        i == length(x) && continue
        x[i] = L[i] * y1[i] + D[i] * y2[i] + U[i] * y3[i]
    end
    return nothing
end

#= e.g., curlC2F =#
function column_curl_like!(curluₕ, uₕ_x, uₕ_y, D, U)
    @inbounds for i in eachindex(curluₕ)
        curluₕ[i] = D[i] * uₕ_x[i] + U[i] * uₕ_y[i]
    end
    return nothing
end

#####
##### CUDA column
#####

# TODO: expand on this

function column_op_2mul_1add_cuda!(x, y, D, U)
    kernel =
        CUDA.@cuda always_inline = true launch = false op_2mul_1add_cuda_kernel!(
            x,
            y,
            D,
            U,
            Val(length(x)),
        )
    config = CUDA.launch_configuration(kernel.fun)
    nitems = length(x)
    threads = min(nitems, config.threads)
    blocks = cld(nitems, threads)
    kernel(x, y, D, U; threads, blocks) # This knows to use always_inline from above.
    return nothing
end

function op_2mul_1add_cuda_kernel!(x, y, D, U, ::Val{N}) where {N}
    @inbounds begin
        i = thread_index()
        if i ≤ N
            x[i] = D[i] * y[i] + U[i] * y[i + 1]
        end
    end
    return nothing
end


#####
##### CPU sphere
#####

# TODO

#####
##### CUDA sphere
#####

# TODO: move to CUDA utils
thread_index() =
    (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
Base.@propagate_inbounds kernel_indexes(tidx, Nv, Nij, Nh) =
    CartesianIndices(map(x -> Base.OneTo(x), (Nv, Nij, Nij, 1, Nh)))[tidx]
valid_range(tidx, n) = 1 ≤ tidx ≤ n


#= e.g., any 2nd order interpolation / derivative operator =#
function sphere_op_2mul_1add_cuda!(x, y, D, U)
    Nv = size(x, 1)
    Nij = size(x, 2)
    Nh = size(x, 5)
    N = length(x)
    kernel =
        CUDA.@cuda always_inline = true launch = false sphere_op_2mul_1add_cuda_kernel!(
            x,
            y,
            D,
            U,
            Val(Nv),
            Val(Nij),
            Val(Nh),
            Val(N),
        )
    config = CUDA.launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    kernel(x, y, D, U; threads, blocks)
    return nothing
end

function sphere_op_2mul_1add_cuda_kernel!(
    x,
    y,
    D,
    U,
    ::Val{Nv},
    ::Val{Nij},
    ::Val{Nh},
    ::Val{N},
) where {Nv, Nij, Nh, N}
    @inbounds begin
        tidx = thread_index()
        if valid_range(tidx, N)
            I = kernel_indexes(tidx, Nv, Nij, Nh)
            x[I] = D[I] * y[I] + U[I] * y[I + CartesianIndex(1, 0, 0, 0, 0)]
        end
    end
    return nothing
end
