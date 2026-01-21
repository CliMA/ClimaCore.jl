import ClimaCore.DataLayouts: data2array_rrtmgp!, array2data_rrtmgp!

function data2array_rrtmgp!(
    array::CUDA.CuArray,
    data::D,
    ::Val{trans},
) where {trans, D <: Union{VF, VIFH, VIHF, VIJFH, VIJHF}}
    (nl, ncol) = trans ? size(array) : reverse(size(array))
    Ni, Nj, Nk, Nv, Nh = Base.size(data)
    @assert nl * ncol == Ni * Nj * Nk * Nv * Nh
    @assert prod(size(parent(data))) == Ni * Nj * Nk * Nv * Nh # verify Nf == 1
    kernel = CUDA.@cuda launch = false data2array_rrtmgp_kernel!(
        array,
        data,
        Val(trans),
    )
    kernel_config = CUDA.launch_configuration(kernel.fun)
    nitems = Ni * Nj * Nk * Nh
    nthreads, nblocks = linear_partition(nitems, kernel_config.threads)
    CUDA.@cuda threads = nthreads blocks = nblocks data2array_rrtmgp_kernel!(
        array,
        data,
        Val(trans),
    )
    return nothing
end

function data2array_rrtmgp_kernel!(
    array::AbstractArray,
    data::AbstractData,
    ::Val{trans},
) where {trans}
    Ni, Nj, Nk, Nv, Nh = Base.size(data)
    ncol = Ni * Nj * Nk * Nh
    # obtain the column number processed by each thread
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if gidx ≤ ncol
        h = cld(gidx, Ni * Nj * Nk)
        idx = gidx - (h - 1) * Ni * Nj * Nk
        k = cld(idx, Ni * Nj)
        idx = idx - (k - 1) * Ni * Nj
        j = cld(idx, Ni)
        i = idx - (j - 1) * Ni
        @inbounds begin
            for v in 1:Nv
                cidx = CartesianIndex(i, j, k, v, h)
                trans ? (array[gidx, v] = data[cidx][1]) :
                (array[v, gidx] = data[cidx][1])
            end
        end
    end
    return nothing
end

_get_kernel_function(::VIJFH) = array2data_rrtmgp_VIJFH_kernel!
_get_kernel_function(::VIFH) = array2data_rrtmgp_VIFH_kernel!
_get_kernel_function(::VF) = array2data_rrtmgp_VF_kernel!

function array2data_rrtmgp!(
    data::D,
    array::CUDA.CuArray,
    ::Val{trans},
) where {trans, D <: Union{VF, VIFH, VIHF, VIJFH, VIJHF}}
    (nl, ncol) = trans ? size(array) : reverse(size(array))
    Ni, Nj, _, Nv, Nh = Base.size(data)
    @assert nl * ncol == Ni * Nj * Nv * Nh
    @assert prod(size(parent(data))) == Ni * Nj * Nv * Nh # verify Nf == 1

    kernelfun! = _get_kernel_function(data)

    kernel =
        CUDA.@cuda launch = false kernelfun!(parent(data), array, Val(trans))
    kernel_config = CUDA.launch_configuration(kernel.fun)
    nitems = Ni * Nj * Nh
    nthreads, nblocks = linear_partition(nitems, kernel_config.threads)
    CUDA.@cuda threads = nthreads blocks = nblocks kernelfun!(
        parent(data),
        array,
        Val(trans),
    )
    return nothing
end

function array2data_rrtmgp_VIJFH_kernel!(
    parentdata::AbstractArray,
    array::AbstractArray,
    ::Val{trans},
) where {trans}
    Nv, Ni, Nj, _, Nh = size(parentdata)
    ncol = Ni * Nj * Nh
    # obtain the column number processed by each thread
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if gidx ≤ ncol
        h = cld(gidx, Ni * Nj)
        j = cld(gidx - (h - 1) * Ni * Nj, Ni)
        i = gidx - (h - 1) * Ni * Nj - (j - 1) * Ni
        for v in 1:Nv
            @inbounds parentdata[v, i, j, 1, h] =
                trans ? array[gidx, v] : array[v, gidx]
        end
    end
    return nothing
end

function array2data_rrtmgp_VIFH_kernel!(
    parentdata::AbstractArray,
    array::AbstractArray,
    ::Val{trans},
) where {trans}
    Nv, Ni, _, Nh = size(parentdata)
    ncol = Ni * Nh
    # obtain the column number processed by each thread
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if gidx ≤ ncol
        h = cld(gidx, Ni)
        i = gidx - (h - 1) * Ni
        for v in 1:Nv
            @inbounds parentdata[v, i, 1, h] =
                trans ? array[gidx, v] : array[v, gidx]
        end
    end
    return nothing
end

function array2data_rrtmgp_VF_kernel!(
    parentdata::AbstractArray,
    array::AbstractArray,
    ::Val{trans},
) where {trans}
    Nv, _ = size(parentdata)
    # obtain the column number processed by each thread
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if gidx ≤ 1
        for v in 1:Nv
            @inbounds parentdata[v, 1] = trans ? array[gidx, v] : array[v, gidx]
        end
    end
    return nothing
end
