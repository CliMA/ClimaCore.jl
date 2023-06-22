#####
##### wdiv(grad(ϕ))
#####

#=
TODO:

We should add simple CuArray-based kernels that:
 - have the same (minimal) number of memory loads (read & writes)
 - have the same number of flops (add / mull etc.)
so that we have a transparent cost of abstraction overhead.

For now, we're tracking existing operators and heuristically
experimenting to minimize the cost. The cost of most kernels
can roughly be compared to `copyto!`, which is likely cheaper
that any spectral element operator, but still provides some
order of magnitude estimate per problem size.
=#

# TODO: add simple array-based kernels as a comparison
# TODO: use more accurate array-based kernels

##### copyto!
kernel_copyto!(args) = kernel_copyto!(args, args.device)
kernel_copyto!(args, ::ClimaComms.AbstractCPUDevice) =
    Base.copyto!(args.ϕ_arr, args.ψ_arr)
function kernel_copyto!(args, ::ClimaComms.CUDADevice)
    (; ϕ_arr, ψ_arr) = args
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(ϕ_arr)
        @inbounds ϕ_arr[i] = ψ_arr[i]
    end
    return nothing
end

##### wdiv(u)
kernel_spectral_wdiv_array!(args) = kernel_copyto!(args)
function kernel_spectral_wdiv!(args)
    (; u, ϕ) = args
    wdiv = Operators.WeakDivergence()
    @. ϕ = wdiv(u)
    return
end

##### grad(ϕ)
kernel_spectral_grad_array!(args) = kernel_copyto!(args)
function kernel_spectral_grad!(args)
    (; ϕ, du) = args
    grad = Operators.Gradient()
    @. du = grad(ϕ)
    return
end

##### grad(norm(ϕ))
kernel_spectral_grad_norm_array!(args) = kernel_copyto!(args)
function kernel_spectral_grad_norm!(args)
    (; ϕ, ψ, u, du) = args
    grad = Operators.Gradient()
    @. du = grad((ϕ + ψ) + LA.norm(u)^2 / 2)
    return
end

##### wdiv(grad(ϕ))
kernel_spectral_div_grad_array!(args) = kernel_copyto!(args)
function kernel_spectral_div_grad!(args)
    (; ϕ, ψ) = args
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    @. ϕ = wdiv(grad(ψ))
    return
end

##### wgrad(div(u))
kernel_spectral_wgrad_div_array!(args) = kernel_copyto!(args)
function kernel_spectral_wgrad_div!(args)
    (; u, du) = args
    wgrad = Operators.WeakGradient()
    div = Operators.Divergence()
    @. du = wgrad(div(u))
    return
end

##### Covariant12Vector(wcurl(Covariant3Vector(curl(u))))
kernel_spectral_wcurl_curl_array!(args) = kernel_copyto!(args)
function kernel_spectral_wcurl_curl!(args)
    (; u, du) = args
    curl = Operators.Curl()
    wcurl = Operators.WeakCurl()
    @. du =
        Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(u))))
    return
end

##### u × curl(u)
kernel_spectral_u_cross_curl_u_array!(args) = kernel_copyto!(args)
function kernel_spectral_u_cross_curl_u!(args)
    (; u, f, du) = args
    curl = Operators.Curl()
    CT12 = Geometry.Contravariant12Vector
    @. du = CT12(u) × (f + curl(u))
    return
end

##### scalar dss!
kernel_scalar_dss_array!(args) = kernel_copyto!(args)
function kernel_scalar_dss!(args)
    Spaces.weighted_dss!(args.ϕ, args.buffers.ϕ_buffer)
    return nothing
end

##### vector dss!
kernel_vector_dss_array!(args) = kernel_copyto!(args)
function kernel_vector_dss!(args)
    Spaces.weighted_dss!(args.u, args.buffers.u_buffer)
    return nothing
end

##### field dss!
kernel_field_dss_array!(args) = kernel_copyto!(args)
function kernel_field_dss!(args)
    Spaces.weighted_dss!(args.ϕψ, args.buffers.ϕψ_buffer)
    return nothing
end

##### ntuple_field dss!
kernel_ntuple_field_dss_array!(args) = kernel_copyto!(args)
function kernel_ntuple_field_dss!(args)
    Spaces.weighted_dss!(args.nt_ϕψ, args.buffers.nt_ϕψ_buffer)
    return nothing
end

##### ntuple_floats dss!
kernel_ntuple_floats_dss_array!(args) = kernel_copyto!(args)
function kernel_ntuple_floats_dss!(args)
    Spaces.weighted_dss!(args.nt_ϕψ_ft, args.buffers.nt_ϕψ_ft_buffer)
    return nothing
end

##### complicated_field dss!
kernel_complicated_field_dss_array!(args) = kernel_copyto!(args)
function kernel_complicated_field_dss!(args)
    Spaces.weighted_dss!(args.f_comp, args.buffers.f_comp_buffer)
    return nothing
end

##### complicated_field2 dss!
kernel_complicated_field2_dss_array!(args) = kernel_copyto!(args)
function kernel_complicated_field2_dss!(args)
    Spaces.weighted_dss!(args.f_comp2, args.buffers.f_comp2_buffer)
    return nothing
end
