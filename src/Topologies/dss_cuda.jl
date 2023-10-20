
_max_threads_cuda() = 256

function _configure_threadblock(max_threads, nitems)
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)
    return (nthreads, nblocks)
end

_configure_threadblock(nitems) =
    _configure_threadblock(_max_threads_cuda(), nitems)

function dss_load_perimeter_data!(
    ::ClimaComms.CUDADevice,
    dss_buffer::DSSBuffer,
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
    perimeter,
) where {S, Nij}
    pperimeter_data = parent(dss_buffer.perimeter_data)
    pdata = parent(data)
    (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)
    nitems = nlevels * nperimeter * nfid * nelems
    nthreads, nblocks = _configure_threadblock(nitems)
    CUDA.@cuda threads = (nthreads) blocks = (nblocks) dss_load_perimeter_data_kernel!(
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
    ::ClimaComms.CUDADevice,
    data::Union{DataLayouts.IJFH{S, Nij}, DataLayouts.VIJFH{S, Nij}},
    dss_buffer::DSSBuffer,
    perimeter,
) where {S, Nij}
    pperimeter_data = parent(dss_buffer.perimeter_data)
    pdata = parent(data)
    (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)
    nitems = nlevels * nperimeter * nfid * nelems
    nthreads, nblocks = _configure_threadblock(nitems)
    CUDA.@cuda threads = (nthreads) blocks = (nblocks) dss_unload_perimeter_data_kernel!(
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
    (nlevels, nperimeter, nfidx, nelems) = sizep = size(pperimeter_data) # size of perimeter data array
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
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    perimeter::Perimeter2D,
    topology::Topologies.Topology2D,
)
    nlocalvertices = length(topology.local_vertex_offset) - 1
    nlocalfaces = length(topology.interior_faces)
    if (nlocalvertices + nlocalfaces) > 0
        pperimeter_data = parent(perimeter_data)
        (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)

        nitems = nlevels * nfid * (nlocalfaces + nlocalvertices)
        nthreads, nblocks = _configure_threadblock(nitems)
        CUDA.@cuda threads = (nthreads) blocks = (nblocks) dss_local_kernel!(
            pperimeter_data,
            topology.local_vertices,
            topology.local_vertex_offset,
            topology.interior_faces,
            perimeter,
        )
    end
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
        (lidx1, face1, lidx2, face2, reversed) = interior_faces[faceid]
        (first1, inc1, last1) =
            Topologies.perimeter_face_indices_cuda(face1, nfacedof, false)
        (first2, inc2, last2) =
            Topologies.perimeter_face_indices_cuda(face2, nfacedof, reversed)
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

    return nothing
end

function dss_transform!(
    device::ClimaComms.CUDADevice,
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
    if nlocalelems > 0
        pdata = parent(data)
        pweight = parent(weight)
        p∂x∂ξ = parent(∂x∂ξ)
        p∂ξ∂x = parent(∂ξ∂x)
        pperimeter_data = parent(perimeter_data)
        nmetric = cld(length(p∂ξ∂x), prod(size(∂ξ∂x)))
        (nlevels, nperimeter, _, _) = size(pperimeter_data)
        nitems = nlevels * nperimeter * nlocalelems
        nthreads, nblocks = _configure_threadblock(nitems)
        CUDA.@cuda threads = (nthreads) blocks = (nblocks) dss_transform_kernel!(
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
    end
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

        (level, p, localelemno) = _get_idx(sizet, gidx)
        elem = localelems[localelemno]
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
    device::ClimaComms.CUDADevice,
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
        nthreads, nblocks = _configure_threadblock(nitems)
        CUDA.@cuda threads = (nthreads) blocks = (nblocks) dss_untransform_kernel!(
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

        (level, p, localelemno) = _get_idx(sizet, gidx)
        elem = localelems[localelemno]
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
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    perimeter::Perimeter2D,
    topology::Topologies.AbstractTopology,
)
    nghostvertices = length(topology.ghost_vertex_offset) - 1
    if nghostvertices > 0
        pperimeter_data = parent(perimeter_data)
        (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)
        max_threads = 256
        nitems = nlevels * nfid * nghostvertices
        nthreads, nblocks = _configure_threadblock(nitems)
        CUDA.@cuda threads = (nthreads) blocks = (nblocks) dss_local_ghost_kernel!(
            pperimeter_data,
            topology.ghost_vertices,
            topology.ghost_vertex_offset,
            perimeter,
        )
    end
    return nothing
end

function dss_local_ghost_kernel!(
    pperimeter_data::AbstractArray{FT, 4},
    ghost_vertices,
    ghost_vertex_offset,
    perimeter::Perimeter2D{Nq},
) where {FT <: AbstractFloat, Nq}
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    (nlevels, nperimeter, nfidx, _) = size(pperimeter_data)
    nghostvertices = length(ghost_vertex_offset) - 1
    if gidx ≤ nlevels * nfidx * nghostvertices
        sizev = (nlevels, nfidx, nghostvertices)
        (level, fidx, vertexid) = _get_idx(sizev, gidx)
        sum_data = FT(0)
        st, en =
            ghost_vertex_offset[vertexid], ghost_vertex_offset[vertexid + 1]
        for idx in st:(en - 1)
            isghost, lidx, vert = ghost_vertices[idx]
            if !isghost
                ip = Topologies.perimeter_vertex_node_index(vert)
                sum_data += pperimeter_data[level, ip, fidx, lidx]
            end
        end
        for idx in st:(en - 1)
            isghost, lidx, vert = ghost_vertices[idx]
            if !isghost
                ip = Topologies.perimeter_vertex_node_index(vert)
                pperimeter_data[level, ip, fidx, lidx] = sum_data
            end
        end
    end
    return nothing
end

function fill_send_buffer!(::ClimaComms.CUDADevice, dss_buffer::DSSBuffer)
    (; perimeter_data, send_buf_idx, send_data) = dss_buffer
    pperimeter_data = parent(perimeter_data)
    (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)
    nsend = size(send_buf_idx, 1)
    if nsend > 0
        nitems = nsend * nlevels * nfid
        nthreads, nblocks = _configure_threadblock(nitems)
        CUDA.synchronize() # CUDA MPI uses a separate stream. This will synchronize across streams
        CUDA.@cuda threads = (nthreads) blocks = (nblocks) fill_send_buffer_kernel!(
            send_data,
            send_buf_idx,
            pperimeter_data,
        )
        CUDA.synchronize() # CUDA MPI uses a separate stream. This will synchronize across streams
    end
    return nothing
end

function fill_send_buffer_kernel!(
    send_data::AbstractArray{FT, 1},
    send_buf_idx::AbstractArray{I, 2},
    pperimeter_data::AbstractArray{FT, 4},
) where {FT <: AbstractFloat, I <: Int}
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    (nlevels, _, nfid, nelems) = size(pperimeter_data)
    nsend = size(send_buf_idx, 1)
    #sizet = (nsend, nlevels, nfid)
    sizet = (nlevels, nfid, nsend)
    #if gidx ≤ nsend * nlevels * nfid
    if gidx ≤ nlevels * nfid * nsend
        #(isend, level, fidx) = _get_idx(sizet, gidx)
        (level, fidx, isend) = _get_idx(sizet, gidx)
        lidx = send_buf_idx[isend, 1]
        ip = send_buf_idx[isend, 2]
        idx = level + ((fidx - 1) + (isend - 1) * nfid) * nlevels
        send_data[idx] = pperimeter_data[level, ip, fidx, lidx]
    end
    return nothing
end

function load_from_recv_buffer!(::ClimaComms.CUDADevice, dss_buffer::DSSBuffer)
    (; perimeter_data, recv_buf_idx, recv_data) = dss_buffer
    pperimeter_data = parent(perimeter_data)
    (nlevels, nperimeter, nfid, nelems) = size(pperimeter_data)
    nrecv = size(recv_buf_idx, 1)
    if nrecv > 0
        nitems = nrecv * nlevels * nfid
        nthreads, nblocks = _configure_threadblock(nitems)
        CUDA.synchronize()
        CUDA.@cuda threads = (nthreads) blocks = (nblocks) load_from_recv_buffer_kernel!(
            pperimeter_data,
            recv_data,
            recv_buf_idx,
        )
        CUDA.synchronize()
    end
    return nothing
end

function load_from_recv_buffer_kernel!(
    pperimeter_data::AbstractArray{FT, 4},
    recv_data::AbstractArray{FT, 1},
    recv_buf_idx::AbstractArray{I, 2},
) where {FT <: AbstractFloat, I <: Int}
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nlevels, _, nfid, nelems = size(pperimeter_data)
    nrecv = size(recv_buf_idx, 1)
    #sizet = (nrecv, nlevels, nfid)
    sizet = (nlevels, nfid, nrecv)
    #if gidx ≤ nrecv * nlevels * nfid
    if gidx ≤ nlevels * nfid * nrecv
        #(irecv, level, fidx) = _get_idx(sizet, gidx)
        (level, fidx, irecv) = _get_idx(sizet, gidx)
        lidx = recv_buf_idx[irecv, 1]
        ip = recv_buf_idx[irecv, 2]
        idx = level + ((fidx - 1) + (irecv - 1) * nfid) * nlevels
        CUDA.@atomic pperimeter_data[level, ip, fidx, lidx] += recv_data[idx]
    end
    return nothing
end


function dss_ghost!(
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    perimeter::Perimeter2D,
    topology::Topologies.Topology2D,
)
    nghostvertices = length(topology.ghost_vertex_offset) - 1
    if nghostvertices > 0
        pperimeter_data = parent(perimeter_data)
        nlevels, _, nfidx, _ = size(pperimeter_data)
        nitems = nlevels * nfidx * nghostvertices
        nthreads, nblocks = _configure_threadblock(nitems)
        CUDA.@cuda threads = (nthreads) blocks = (nblocks) dss_ghost_kernel!(
            pperimeter_data,
            topology.ghost_vertices,
            topology.ghost_vertex_offset,
            topology.repr_ghost_vertex,
            perimeter,
        )
    end
    return nothing
end

function dss_ghost_kernel!(
    pperimeter_data::AbstractArray{FT, 4},
    ghost_vertices,
    ghost_vertex_offset,
    repr_ghost_vertex,
    perimeter::Perimeter2D{Nq},
) where {FT <: AbstractFloat, Nq}
    gidx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nlevels, _, nfidx, _ = size(pperimeter_data)
    nghostvertices = length(ghost_vertex_offset) - 1

    if gidx ≤ nlevels * nfidx * nghostvertices
        (level, fidx, ghostvertexidx) =
            _get_idx((nlevels, nfidx, nghostvertices), gidx)
        idxresult, lvertresult = repr_ghost_vertex[ghostvertexidx]
        ipresult = Topologies.perimeter_vertex_node_index(lvertresult)
        result = pperimeter_data[level, ipresult, fidx, idxresult]
        st, en = ghost_vertex_offset[ghostvertexidx],
        ghost_vertex_offset[ghostvertexidx + 1]
        for vertexidx in st:(en - 1)
            isghost, eidx, lvert = ghost_vertices[vertexidx]
            if !isghost
                ip = Topologies.perimeter_vertex_node_index(lvert)
                pperimeter_data[level, ip, fidx, eidx] = result
            end
        end
    end
    return nothing
end
