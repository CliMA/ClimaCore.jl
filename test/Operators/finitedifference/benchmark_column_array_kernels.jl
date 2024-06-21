#####
##### CPU
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
        i==1 && continue
        i==length(x) && continue
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
##### CUDA
#####

# TODO

function column_op_2mul_1add_cuda!(x, y, D, U)
    kernel = CUDA.@cuda always_inline = true launch = false op_2mul_1add_cuda_kernel!(x, y, D, U, Val(length(x)))
    config = CUDA.launch_configuration(kernel.fun)
    nitems = length(x)
    threads = min(nitems, config.threads)
    blocks = cld(nitems, threads)
    kernel(x, y, D, U; threads, blocks) # This knows to use always_inline from above.
    return nothing
end

function op_2mul_1add_cuda_kernel!(x, y, D, U, ::Val{N}) where N
    @inbounds begin
        i = thread_index()
        if i ≤ N
            x[i] = D[i] * y[i] + U[i] * y[i+1]
        end
    end
    return nothing
end
