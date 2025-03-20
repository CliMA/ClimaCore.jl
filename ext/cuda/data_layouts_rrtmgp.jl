import ClimaCore.DataLayouts: data2array_rrtmgp!, array2data_rrtmgp!

function data2array_rrtmgp!(
    array::CUDA.CuArray,
    data::AbstractData,
    ::Val{trans},
) where {trans}
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
    gid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if gid ≤ ncol
        h = cld(gid, Ni * Nj * Nk)
        idx = gid - (h - 1) * Ni * Nj * Nk
        k = cld(idx, Ni * Nj)
        idx = idx - (k - 1) * Ni * Nj
        j = cld(idx, Ni)
        i = idx - (j - 1) * Ni
        @inbounds begin
            for v in 1:Nv
                colidx =
                    i +
                    (j - 1) * Ni +
                    (k - 1) * Ni * Nj +
                    (h - 1) * Ni * Nj * Nk
                cidx = CartesianIndex(i, j, k, v, h)
                trans ? (array[colidx, v] = data[cidx]) :
                (array[v, colidx] = data[cidx])
            end
        end
    end
    return nothing
end

function array2data_rrtmgp!(
    data::AbstractData,
    array::CUDA.CuArray,
    ::Val{trans},
) where {trans}
    (nl, ncol) = trans ? size(array) : reverse(size(array))
    Ni, Nj, Nk, Nv, Nh = Base.size(data)
    @assert nl * ncol == Ni * Nj * Nk * Nv * Nh
    @assert prod(size(parent(data))) == Ni * Nj * Nk * Nv * Nh # verify Nf == 1
    kernel = CUDA.@cuda launch = false array2data_rrtmgp_kernel!(
        data,
        array,
        Val(trans),
    )
    kernel_config = CUDA.launch_configuration(kernel.fun)
    nitems = Ni * Nj * Nk * Nh
    nthreads, nblocks = linear_partition(nitems, kernel_config.threads)
    CUDA.@cuda threads = nthreads blocks = nblocks array2data_rrtmgp_kernel!(
        data,
        array,
        Val(trans),
    )
    return nothing
end

function array2data_rrtmgp_kernel!(
    data::AbstractData,
    array::AbstractArray,
    ::Val{trans},
) where {trans}
    Ni, Nj, Nk, Nv, Nh = Base.size(data)
    ncol = Ni * Nj * Nk * Nh
    # obtain the column number processed by each thread
    gid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if gid ≤ ncol
        h = cld(gid, Ni * Nj * Nk)
        idx = gid - (h - 1) * Ni * Nj * Nk
        k = cld(idx, Ni * Nj)
        idx = idx - (k - 1) * Ni * Nj
        j = cld(idx, Ni)
        i = idx - (j - 1) * Ni
        colidx = i + (j - 1) * Ni + (k - 1) * Ni * Nj + (h - 1) * Ni * Nj * Nk
        @inbounds begin
            for v in 1:Nv
                data[CartesianIndex(i, j, k, v, h)] =
                    trans ? array[colidx, v] : array[v, colidx]
            end
        end
    end
    return nothing
end
