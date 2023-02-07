
function dss_load_perimeter_data!(
    ::ClimaComms.CUDA,
    dss_buffer::DSSBuffer,
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
    perimeter,
) where {S, Nij}
    pperimeter_data = parent(dss_buffer.perimeter_data)
    pdata = parent(data)
    (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)
    nitems = nlevels * nperimeter * nfid * nelems
    max_threads = 256
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)
    @cuda threads = (nthreads) blocks = (nblocks) dss_load_perimeter_data_kernel!(
        pperimeter_data,
        pdata,
        perimeter,
    )
    return nothing
end

function dss_load_perimeter_data_kernel!(
    pperimeter_data::AbstractArray{FT, 4},
    pdata::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    perimeter::Perimeter2D{Nq},
) where {FT <: AbstractFloat, Nq}
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    (nlevels, _, nfidx, nelems) = sizep = size(pperimeter_data) # size of perimeter data array
    sized = (nlevels, Nq, Nq, nfidx, nelems) # size of data

    if gidx ≤ prod(sizep)
        (level, p, fidx, elem) = _get_idx(sizep, gidx)
        (ip, jp) = perimeter[p]
        data_idx = _get_idx(sized, (level, ip, jp, fidx, elem))
        pperimeter_data[level, p, fidx, elem] = pdata[data_idx]
    end
    return nothing
end

function dss_unload_perimeter_data!(
    ::ClimaComms.CUDA,
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
    dss_buffer::DSSBuffer,
    perimeter,
) where {S, Nij}
    pperimeter_data = parent(dss_buffer.perimeter_data)
    pdata = parent(data)
    (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)
    nitems = nlevels * nperimeter * nfid * nelems
    max_threads = 256
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)
    @cuda threads = (nthreads) blocks = (nblocks) dss_unload_perimeter_data_kernel!(
        pdata,
        pperimeter_data,
        perimeter,
    )
    return nothing
end

function dss_unload_perimeter_data_kernel!(
    pdata::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    pperimeter_data::AbstractArray{FT, 4},
    perimeter::Perimeter2D{Nq},
) where {FT <: AbstractFloat, Nq}
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    (nlevels, _, nfidx, nelems) = sizep = size(pperimeter_data) # size of perimeter data array
    sized = (nlevels, Nq, Nq, nfidx, nelems) # size of data

    if gidx ≤ prod(sizep)
        (level, p, fidx, elem) = _get_idx(sizep, gidx)
        (ip, jp) = perimeter[p]
        data_idx = _get_idx(sized, (level, ip, jp, fidx, elem))
        pdata[data_idx] = pperimeter_data[level, p, fidx, elem]
    end
    return nothing
end

function dss_local!(
    ::ClimaComms.CUDA,
    perimeter_data::DataLayouts.VIFH,
    perimeter::Perimeter2D,
    topology::Topologies.Topology2D,
)
    nlocalvertices = length(topology.local_vertex_offset) - 1
    nlocalfaces = length(topology.interior_faces)
    pperimeter_data = parent(perimeter_data)
    (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)

    max_threads = 256
    nitems = nlevels * nfid * (nlocalfaces + nlocalvertices)
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)
    @cuda threads = (nthreads) blocks = (nblocks) dss_local_kernel!(
        pperimeter_data,
        topology.local_vertices,
        topology.local_vertex_offset,
        topology.interior_faces,
        perimeter,
    )
    return nothing
end

function dss_local_kernel!(
    pperimeter_data::AbstractArray{FT, 4},
    local_vertices::AbstractVector{Tuple{Int, Int}},
    local_vertex_offset::AbstractVector{Int},
    interior_faces::AbstractVector{Tuple{Int, Int, Int, Int, Bool}},
    perimeter::Perimeter2D{Nq},
) where {FT <: AbstractFloat, Nq}
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nlocalvertices = length(local_vertex_offset) - 1
    nlocalfaces = length(interior_faces)
    (nlevels, nperimeter, nfidx, _) = size(pperimeter_data)
    if gidx ≤ nlevels * nfidx * nlocalvertices # local vertices
        sizev = (nlevels, nfidx, nlocalvertices)
        (level, fidx, vertexid) = _get_idx(sizev, gidx)
        sum_data = FT(0)
        st, en =
            local_vertex_offset[vertexid], local_vertex_offset[vertexid + 1]
        for idx in st:(en - 1)
            (lidx, vert) = local_vertices[idx]
            ip = Topologies.perimeter_vertex_node_index(vert)
            sum_data += pperimeter_data[level, ip, fidx, lidx]
        end
        for idx in st:(en - 1)
            (lidx, vert) = local_vertices[idx]
            ip = Topologies.perimeter_vertex_node_index(vert)
            pperimeter_data[level, ip, fidx, lidx] = sum_data
        end
    elseif gidx ≤ nlevels * nfidx * (nlocalvertices + nlocalfaces) # interior faces
        nfacedof = div(nperimeter - 4, 4)
        sizef = (nlevels, nfidx, nlocalfaces)
        (level, fidx, faceid) =
            _get_idx(sizef, gidx - nlevels * nfidx * nlocalvertices)
        for idx in 1:nlocalfaces
            (lidx1, face1, lidx2, face2, reversed) = interior_faces[idx]
            (first1, inc1, last1) =
                Topologies.perimeter_face_indices_cuda(face1, nfacedof, false)
            (first2, inc2, last2) = Topologies.perimeter_face_indices_cuda(
                face2,
                nfacedof,
                reversed,
            )
            for i in 1:nfacedof
                ip1 = inc1 == 1 ? first1 + i - 1 : first1 - i + 1
                ip2 = inc2 == 1 ? first2 + i - 1 : first2 - i + 1
                val =
                    pperimeter_data[level, ip1, fidx, lidx1] +
                    pperimeter_data[level, ip2, fidx, lidx2]
                pperimeter_data[level, ip1, fidx, lidx1] = val
                pperimeter_data[level, ip2, fidx, lidx2] = val
            end
        end
    end

    return nothing
end

function dss_transform!(
    device::ClimaComms.CUDA,
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂ξ∂x::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂x∂ξ::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    weight::DataLayouts.IJFH,
    perimeter::Perimeter2D,
    scalarfidx::AbstractVector{Int},
    covariant12fidx::AbstractVector{Int},
    contravariant12fidx::AbstractVector{Int},
    localelems::AbstractVector{Int},
)
    nlocalelems = length(localelems)
    pdata = parent(data)
    pweight = parent(weight)
    p∂x∂ξ = parent(∂x∂ξ)
    p∂ξ∂x = parent(∂ξ∂x)
    pperimeter_data = parent(perimeter_data)
    nmetric = cld(length(p∂ξ∂x), prod(size(∂ξ∂x)))
    (nlevels, nperimeter, _, _) = size(pperimeter_data)
    nitems = nlevels * nperimeter * nlocalelems
    max_threads = 256
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)
    @cuda threads = (nthreads) blocks = (nblocks) dss_transform_kernel!(
        pperimeter_data,
        pdata,
        p∂ξ∂x,
        p∂x∂ξ,
        nmetric,
        pweight,
        perimeter,
        scalarfidx,
        covariant12fidx,
        contravariant12fidx,
        localelems,
    )
    return nothing
end

function dss_transform_kernel!(
    pperimeter_data::AbstractArray{FT, 4},
    pdata::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    p∂ξ∂x::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    p∂x∂ξ::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    nmetric::Int,
    pweight::AbstractArray{FT, 4},
    perimeter::Perimeter2D{Nq},
    scalarfidx::AbstractVector{Int},
    covariant12fidx::AbstractVector{Int},
    contravariant12fidx::AbstractVector{Int},
    localelems::AbstractVector{Int},
) where {FT <: AbstractFloat, Nq}
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)
    nlocalelems = length(localelems)
    if gidx ≤ nlevels * nperimeter * nlocalelems
        sizet = (nlevels, nperimeter, nlocalelems)
        sizet_data = (nlevels, Nq, Nq, nfid, nelems)
        sizet_wt = (Nq, Nq, 1, nelems)
        sizet_metric = (nlevels, Nq, Nq, nmetric, nelems)

        (level, p, elem) = _get_idx(sizet, gidx)
        (ip, jp) = perimeter[p]

        weight = pweight[_get_idx(sizet_wt, (ip, jp, 1, elem))]
        for fidx in scalarfidx
            data_idx = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
            pperimeter_data[level, p, fidx, elem] = pdata[data_idx] * weight
        end
        for fidx in covariant12fidx
            data_idx1 = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = _get_idx(sizet_data, (level, ip, jp, fidx + 1, elem))
            (idx11, idx12, idx21, idx22) =
                _get_idx_metric(sizet_metric, (level, ip, jp, elem))
            pperimeter_data[level, p, fidx, elem] =
                (
                    p∂ξ∂x[idx11] * pdata[data_idx1] +
                    p∂ξ∂x[idx12] * pdata[data_idx2]
                ) * weight
            pperimeter_data[level, p, fidx + 1, elem] =
                (
                    p∂ξ∂x[idx21] * pdata[data_idx1] +
                    p∂ξ∂x[idx22] * pdata[data_idx2]
                ) * weight
        end
        for fidx in contravariant12fidx
            data_idx1 = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = _get_idx(sizet_data, (level, ip, jp, fidx + 1, elem))
            (idx11, idx12, idx21, idx22) =
                _get_idx_metric(sizet_metric, (level, ip, jp, elem))
            pperimeter_data[level, p, fidx, elem] =
                (
                    p∂x∂ξ[idx11] * pdata[data_idx1] +
                    p∂x∂ξ[idx21] * pdata[data_idx2]
                ) * weight
            pperimeter_data[level, p, fidx + 1, elem] =
                (
                    p∂x∂ξ[idx12] * pdata[data_idx1] +
                    p∂x∂ξ[idx22] * pdata[data_idx2]
                ) * weight
        end
    end
    return nothing
end

function dss_untransform!(
    device::ClimaComms.CUDA,
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂ξ∂x::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂x∂ξ::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    perimeter::Perimeter2D{Nq},
    scalarfidx::AbstractVector{Int},
    covariant12fidx::AbstractVector{Int},
    contravariant12fidx::AbstractVector{Int},
    localelems::AbstractVector{Int},
) where {Nq}
    nlocalelems = length(localelems)
    if nlocalelems > 0
        pdata = parent(data)
        p∂x∂ξ = parent(∂x∂ξ)
        p∂ξ∂x = parent(∂ξ∂x)
        nmetric = cld(length(p∂ξ∂x), prod(size(∂ξ∂x)))
        pperimeter_data = parent(perimeter_data)
        (nlevels, nperimeter, _, _) = size(pperimeter_data)
        nitems = nlevels * nperimeter * nlocalelems
        max_threads = 256
        nthreads = min(max_threads, nitems)
        nblocks = cld(nitems, nthreads)
        @cuda threads = (nthreads) blocks = (nblocks) dss_untransform_kernel!(
            pperimeter_data,
            pdata,
            p∂ξ∂x,
            p∂x∂ξ,
            nmetric,
            perimeter,
            scalarfidx,
            covariant12fidx,
            contravariant12fidx,
            localelems,
        )
    end
    return nothing
end

function dss_untransform_kernel!(
    pperimeter_data::AbstractArray{FT, 4},
    pdata::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    p∂ξ∂x::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    p∂x∂ξ::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    nmetric::Int,
    perimeter::Perimeter2D{Nq},
    scalarfidx::AbstractVector{Int},
    covariant12fidx::AbstractVector{Int},
    contravariant12fidx::AbstractVector{Int},
    localelems::AbstractVector{Int},
) where {FT <: AbstractFloat, Nq}
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)
    nlocalelems = length(localelems)
    if gidx ≤ nlevels * nperimeter * nlocalelems
        sizet = (nlevels, nperimeter, nlocalelems)
        sizet_data = (nlevels, Nq, Nq, nfid, nelems)
        sizet_wt = (Nq, Nq, 1, nelems)
        sizet_metric = (nlevels, Nq, Nq, nmetric, nelems)

        (level, p, elem) = _get_idx(sizet, gidx)
        ip, jp = perimeter[p]
        for fidx in scalarfidx
            data_idx = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
            pdata[data_idx] = pperimeter_data[level, p, fidx, elem]
        end
        for fidx in covariant12fidx
            data_idx1 = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = _get_idx(sizet_data, (level, ip, jp, fidx + 1, elem))
            (idx11, idx12, idx21, idx22) =
                _get_idx_metric(sizet_metric, (level, ip, jp, elem))
            pdata[data_idx1] =
                p∂x∂ξ[idx11] * pperimeter_data[level, p, fidx, elem] +
                p∂x∂ξ[idx12] * pperimeter_data[level, p, fidx + 1, elem]
            pdata[data_idx2] =
                p∂x∂ξ[idx21] * pperimeter_data[level, p, fidx, elem] +
                p∂x∂ξ[idx22] * pperimeter_data[level, p, fidx + 1, elem]
        end
        for fidx in contravariant12fidx
            data_idx1 = _get_idx(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = _get_idx(sizet_data, (level, ip, jp, fidx + 1, elem))
            (idx11, idx12, idx21, idx22) =
                _get_idx_metric(sizet_metric, (level, ip, jp, elem))
            pdata[data_idx1] =
                p∂ξ∂x[idx11] * pperimeter_data[level, p, fidx, elem] +
                p∂ξ∂x[idx21] * pperimeter_data[level, p, fidx + 1, elem]
            pdata[data_idx2] =
                p∂ξ∂x[idx12] * pperimeter_data[level, p, fidx, elem] +
                p∂ξ∂x[idx22] * pperimeter_data[level, p, fidx + 1, elem]
        end
    end
    return nothing
end

# TODO: Function stubs, code to be implemented, needed only for distributed GPU runs
function dss_local_ghost!(
    ::ClimaComms.CUDA,
    perimeter_data::DataLayouts.VIFH,
    perimeter::AbstractPerimeter,
    topology::Topologies.AbstractTopology,
)
    return nothing
end

function fill_send_buffer!(::ClimaComms.CUDA, dss_buffer::DSSBuffer)
    return nothing
end

function load_from_recv_buffer!(::ClimaComms.CUDA, dss_buffer::DSSBuffer)
    return nothing
end

function dss_ghost!(
    ::ClimaComms.CUDA,
    perimeter_data::DataLayouts.VIFH,
    perimeter::AbstractPerimeter,
    topology::Topologies.AbstractTopology,
)
    return nothing
end
