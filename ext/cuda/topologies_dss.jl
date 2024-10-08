import ClimaCore: DataLayouts, Topologies, Spaces, Fields
using CUDA
import ClimaCore.Topologies
import ClimaCore.Topologies: perimeter_vertex_node_index

_max_threads_cuda() = 256

function _configure_threadblock(max_threads, nitems)
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)
    return (nthreads, nblocks)
end

_configure_threadblock(nitems) =
    _configure_threadblock(_max_threads_cuda(), nitems)

function Topologies.dss_load_perimeter_data!(
    ::ClimaComms.CUDADevice,
    dss_buffer::Topologies.DSSBuffer,
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    perimeter::Topologies.Perimeter2D,
)
    (; perimeter_data) = dss_buffer
    nitems = prod(DataLayouts.farray_size(perimeter_data))
    nthreads, nblocks = _configure_threadblock(nitems)
    args = (perimeter_data, data, perimeter)
    auto_launch!(
        dss_load_perimeter_data_kernel!,
        args;
        threads_s = (nthreads),
        blocks_s = (nblocks),
    )
    return nothing
end

function dss_load_perimeter_data_kernel!(
    perimeter_data::DataLayouts.AbstractData,
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    perimeter::Topologies.Perimeter2D{Nq},
) where {Nq}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    (nlevels, _, nfidx, nelems) =
        sizep = DataLayouts.farray_size(perimeter_data) # size of perimeter data array
    sized = (nlevels, Nq, Nq, nfidx, nelems) # size of data
    pperimeter_data = parent(perimeter_data)
    pdata = parent(data)

    if gidx ≤ prod(sizep)
        (level, p, fidx, elem) = cart_ind(sizep, gidx).I
        (ip, jp) = perimeter[p]
        data_idx = linear_ind(sized, (level, ip, jp, fidx, elem))
        pperimeter_data[level, p, fidx, elem] = pdata[data_idx]
    end
    return nothing
end

function Topologies.dss_unload_perimeter_data!(
    ::ClimaComms.CUDADevice,
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    dss_buffer::Topologies.DSSBuffer,
    perimeter,
)
    (; perimeter_data) = dss_buffer
    nitems = prod(DataLayouts.farray_size(perimeter_data))
    nthreads, nblocks = _configure_threadblock(nitems)
    args = (data, perimeter_data, perimeter)
    auto_launch!(
        dss_unload_perimeter_data_kernel!,
        args;
        threads_s = (nthreads),
        blocks_s = (nblocks),
    )
    return nothing
end

function dss_unload_perimeter_data_kernel!(
    data::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    perimeter_data::AbstractData,
    perimeter::Topologies.Perimeter2D{Nq},
) where {Nq}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    (nlevels, nperimeter, nfidx, nelems) =
        sizep = DataLayouts.farray_size(perimeter_data) # size of perimeter data array
    sized = (nlevels, Nq, Nq, nfidx, nelems) # size of data
    pperimeter_data = parent(perimeter_data)
    pdata = parent(data)

    if gidx ≤ prod(sizep)
        (level, p, fidx, elem) = cart_ind(sizep, gidx).I
        (ip, jp) = perimeter[p]
        data_idx = linear_ind(sized, (level, ip, jp, fidx, elem))
        pdata[data_idx] = pperimeter_data[level, p, fidx, elem]
    end
    return nothing
end

function Topologies.dss_local!(
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    perimeter::Topologies.Perimeter2D,
    topology::Topologies.Topology2D,
)
    nlocalvertices = length(topology.local_vertex_offset) - 1
    nlocalfaces = length(topology.interior_faces)
    if (nlocalvertices + nlocalfaces) > 0
        (nlevels, nperimeter, nfid, nelems) =
            DataLayouts.farray_size(perimeter_data)

        nitems = nlevels * nfid * (nlocalfaces + nlocalvertices)
        nthreads, nblocks = _configure_threadblock(nitems)
        args = (
            perimeter_data,
            topology.local_vertices,
            topology.local_vertex_offset,
            topology.interior_faces,
            perimeter,
        )
        auto_launch!(
            dss_local_kernel!,
            args;
            threads_s = (nthreads),
            blocks_s = (nblocks),
        )
    end
    return nothing
end

function dss_local_kernel!(
    perimeter_data::DataLayouts.VIFH,
    local_vertices::AbstractVector{Tuple{Int, Int}},
    local_vertex_offset::AbstractVector{Int},
    interior_faces::AbstractVector{Tuple{Int, Int, Int, Int, Bool}},
    perimeter::Topologies.Perimeter2D,
)
    FT = eltype(parent(perimeter_data))
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    nlocalvertices = length(local_vertex_offset) - 1
    nlocalfaces = length(interior_faces)
    pperimeter_data = parent(perimeter_data)
    FT = eltype(pperimeter_data)
    (nlevels, nperimeter, nfidx, _) = DataLayouts.farray_size(perimeter_data)
    if gidx ≤ nlevels * nfidx * nlocalvertices # local vertices
        sizev = (nlevels, nfidx, nlocalvertices)
        (level, fidx, vertexid) = cart_ind(sizev, gidx).I
        sum_data = FT(0)
        st, en =
            local_vertex_offset[vertexid], local_vertex_offset[vertexid + 1]
        for idx in st:(en - 1)
            (lidx, vert) = local_vertices[idx]
            ip = perimeter_vertex_node_index(vert)
            sum_data += pperimeter_data[level, ip, fidx, lidx]
        end
        for idx in st:(en - 1)
            (lidx, vert) = local_vertices[idx]
            ip = perimeter_vertex_node_index(vert)
            pperimeter_data[level, ip, fidx, lidx] = sum_data
        end
    elseif gidx ≤ nlevels * nfidx * (nlocalvertices + nlocalfaces) # interior faces
        nfacedof = div(nperimeter - 4, 4)
        sizef = (nlevels, nfidx, nlocalfaces)
        (level, fidx, faceid) =
            cart_ind(sizef, gidx - nlevels * nfidx * nlocalvertices).I
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

function Topologies.dss_transform!(
    device::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    perimeter::Topologies.Perimeter2D,
    local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    weight::DataLayouts.IJFH,
    localelems::AbstractVector{Int},
)
    nlocalelems = length(localelems)
    if nlocalelems > 0
        (nperimeter, _, _, nlevels, _) =
            DataLayouts.universal_size(perimeter_data)
        nitems = nlevels * nperimeter * nlocalelems
        nthreads, nblocks = _configure_threadblock(nitems)
        args = (
            perimeter_data,
            data,
            perimeter,
            local_geometry,
            weight,
            localelems,
            Val(nlocalelems),
        )
        auto_launch!(
            dss_transform_kernel!,
            args;
            threads_s = (nthreads),
            blocks_s = (nblocks),
        )
    end
    return nothing
end

function dss_transform_kernel!(
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    perimeter::Topologies.Perimeter2D,
    local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    weight::DataLayouts.IJFH,
    localelems::AbstractVector{Int},
    ::Val{nlocalelems},
) where {nlocalelems}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    (nperimeter, _, _, nlevels, nelems) =
        DataLayouts.universal_size(perimeter_data)
    CI = CartesianIndex
    if gidx ≤ nlevels * nperimeter * nlocalelems
        sizet = (nlevels, nperimeter, nlocalelems)
        (level, p, localelemno) = cart_ind(sizet, gidx).I
        elem = localelems[localelemno]
        (ip, jp) = perimeter[p]
        loc = CI(ip, jp, 1, level, elem)
        src = Topologies.dss_transform(
            data[loc],
            local_geometry[loc],
            weight[loc],
        )
        perimeter_data[CI(p, 1, 1, level, elem)] =
            Topologies.drop_vert_dim(eltype(perimeter_data), src)
    end
    return nothing
end

function Topologies.dss_untransform!(
    device::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    perimeter::Topologies.Perimeter2D,
    localelems::AbstractVector{Int},
)
    nlocalelems = length(localelems)
    if nlocalelems > 0
        (nperimeter, _, _, nlevels, _) =
            DataLayouts.universal_size(perimeter_data)
        nitems = nlevels * nperimeter * nlocalelems
        nthreads, nblocks = _configure_threadblock(nitems)
        args = (
            perimeter_data,
            data,
            local_geometry,
            perimeter,
            localelems,
            Val(nlocalelems),
        )
        auto_launch!(
            dss_untransform_kernel!,
            args;
            threads_s = (nthreads),
            blocks_s = (nblocks),
        )
    end
    return nothing
end

function dss_untransform_kernel!(
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    local_geometry::Union{DataLayouts.IJFH, DataLayouts.VIJFH},
    perimeter::Topologies.Perimeter2D,
    localelems::AbstractVector{Int},
    ::Val{nlocalelems},
) where {nlocalelems}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    (nperimeter, _, _, nlevels, _) = DataLayouts.universal_size(perimeter_data)
    CI = CartesianIndex
    if gidx ≤ nlevels * nperimeter * nlocalelems
        sizet = (nlevels, nperimeter, nlocalelems)
        (level, p, localelemno) = cart_ind(sizet, gidx).I
        elem = localelems[localelemno]
        ip, jp = perimeter[p]

        loc = CI(ip, jp, 1, level, elem)
        data[loc] = Topologies.dss_untransform(
            eltype(data),
            perimeter_data[CI(p, 1, 1, level, elem)],
            local_geometry[loc],
        )
    end
    return nothing
end

# TODO: Function stubs, code to be implemented, needed only for distributed GPU runs
function Topologies.dss_local_ghost!(
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    perimeter::Topologies.Perimeter2D,
    topology::Topologies.AbstractTopology,
)
    nghostvertices = length(topology.ghost_vertex_offset) - 1
    if nghostvertices > 0
        (nlevels, nperimeter, nfid, nelems) =
            DataLayouts.farray_size(perimeter_data)
        max_threads = 256
        nitems = nlevels * nfid * nghostvertices
        nthreads, nblocks = _configure_threadblock(nitems)
        args = (
            perimeter_data,
            topology.ghost_vertices,
            topology.ghost_vertex_offset,
            perimeter,
        )
        auto_launch!(
            dss_local_ghost_kernel!,
            args;
            threads_s = (nthreads),
            blocks_s = (nblocks),
        )
    end
    return nothing
end

function dss_local_ghost_kernel!(
    perimeter_data::DataLayouts.VIFH,
    ghost_vertices,
    ghost_vertex_offset,
    perimeter::Topologies.Perimeter2D,
)
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    pperimeter_data = parent(perimeter_data)
    FT = eltype(pperimeter_data)
    (nlevels, nperimeter, nfidx, _) = DataLayouts.farray_size(perimeter_data)
    nghostvertices = length(ghost_vertex_offset) - 1
    if gidx ≤ nlevels * nfidx * nghostvertices
        sizev = (nlevels, nfidx, nghostvertices)
        (level, fidx, vertexid) = cart_ind(sizev, gidx).I
        sum_data = FT(0)
        st, en =
            ghost_vertex_offset[vertexid], ghost_vertex_offset[vertexid + 1]
        for idx in st:(en - 1)
            isghost, lidx, vert = ghost_vertices[idx]
            if !isghost
                ip = perimeter_vertex_node_index(vert)
                sum_data += pperimeter_data[level, ip, fidx, lidx]
            end
        end
        for idx in st:(en - 1)
            isghost, lidx, vert = ghost_vertices[idx]
            if !isghost
                ip = perimeter_vertex_node_index(vert)
                pperimeter_data[level, ip, fidx, lidx] = sum_data
            end
        end
    end
    return nothing
end

function Topologies.fill_send_buffer!(
    ::ClimaComms.CUDADevice,
    dss_buffer::Topologies.DSSBuffer;
    synchronize = true,
)
    (; perimeter_data, send_buf_idx, send_data) = dss_buffer
    (nlevels, nperimeter, nfid, nelems) =
        DataLayouts.farray_size(perimeter_data)
    nsend = size(send_buf_idx, 1)
    if nsend > 0
        nitems = nsend * nlevels * nfid
        nthreads, nblocks = _configure_threadblock(nitems)
        args = (send_data, send_buf_idx, perimeter_data, Val(nsend))
        auto_launch!(
            fill_send_buffer_kernel!,
            args;
            threads_s = (nthreads),
            blocks_s = (nblocks),
        )
        if synchronize
            CUDA.synchronize(; blocking = true) # CUDA MPI uses a separate stream. This will synchronize across streams
        end
    end
    return nothing
end

function fill_send_buffer_kernel!(
    send_data::AbstractArray{FT, 1},
    send_buf_idx::AbstractArray{I, 2},
    perimeter_data::AbstractData,
    ::Val{nsend},
) where {FT <: AbstractFloat, I <: Int, nsend}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    (nlevels, _, nfid, nelems) = DataLayouts.farray_size(perimeter_data)
    pperimeter_data = parent(perimeter_data)
    #sizet = (nsend, nlevels, nfid)
    sizet = (nlevels, nfid, nsend)
    #if gidx ≤ nsend * nlevels * nfid
    if gidx ≤ nlevels * nfid * nsend
        #(isend, level, fidx) = cart_ind(sizet, gidx).I
        (level, fidx, isend) = cart_ind(sizet, gidx).I
        lidx = send_buf_idx[isend, 1]
        ip = send_buf_idx[isend, 2]
        idx = level + ((fidx - 1) + (isend - 1) * nfid) * nlevels
        send_data[idx] = pperimeter_data[level, ip, fidx, lidx]
    end
    return nothing
end

function Topologies.load_from_recv_buffer!(
    ::ClimaComms.CUDADevice,
    dss_buffer::Topologies.DSSBuffer,
)
    (; perimeter_data, recv_buf_idx, recv_data) = dss_buffer
    (nlevels, nperimeter, nfid, nelems) =
        DataLayouts.farray_size(perimeter_data)
    nrecv = size(recv_buf_idx, 1)
    if nrecv > 0
        nitems = nrecv * nlevels * nfid
        nthreads, nblocks = _configure_threadblock(nitems)
        args = (perimeter_data, recv_data, recv_buf_idx, Val(nrecv))
        auto_launch!(
            load_from_recv_buffer_kernel!,
            args;
            threads_s = (nthreads),
            blocks_s = (nblocks),
        )
    end
    return nothing
end

function load_from_recv_buffer_kernel!(
    perimeter_data::AbstractData,
    recv_data::AbstractArray{FT, 1},
    recv_buf_idx::AbstractArray{I, 2},
    ::Val{nrecv},
) where {FT <: AbstractFloat, I <: Int, nrecv}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    pperimeter_data = parent(perimeter_data)
    (nlevels, _, nfid, nelems) = DataLayouts.farray_size(perimeter_data)
    #sizet = (nrecv, nlevels, nfid)
    sizet = (nlevels, nfid, nrecv)
    #if gidx ≤ nrecv * nlevels * nfid
    if gidx ≤ nlevels * nfid * nrecv
        #(irecv, level, fidx) = cart_ind(sizet, gidx).I
        (level, fidx, irecv) = cart_ind(sizet, gidx).I
        lidx = recv_buf_idx[irecv, 1]
        ip = recv_buf_idx[irecv, 2]
        idx = level + ((fidx - 1) + (irecv - 1) * nfid) * nlevels
        CUDA.@atomic pperimeter_data[level, ip, fidx, lidx] += recv_data[idx]
    end
    return nothing
end


function Topologies.dss_ghost!(
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    perimeter::Topologies.Perimeter2D,
    topology::Topologies.Topology2D,
)
    nghostvertices = length(topology.ghost_vertex_offset) - 1
    if nghostvertices > 0
        (nlevels, _, nfidx, _) = DataLayouts.farray_size(perimeter_data)
        nitems = nlevels * nfidx * nghostvertices
        nthreads, nblocks = _configure_threadblock(nitems)
        args = (
            perimeter_data,
            topology.ghost_vertices,
            topology.ghost_vertex_offset,
            topology.repr_ghost_vertex,
            perimeter,
        )
        auto_launch!(
            dss_ghost_kernel!,
            args;
            threads_s = (nthreads),
            blocks_s = (nblocks),
        )
    end
    return nothing
end

function dss_ghost_kernel!(
    perimeter_data::AbstractData,
    ghost_vertices,
    ghost_vertex_offset,
    repr_ghost_vertex,
    perimeter::Topologies.Perimeter2D,
)
    pperimeter_data = parent(perimeter_data)
    FT = eltype(pperimeter_data)
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    (nlevels, _, nfidx, _) = DataLayouts.farray_size(perimeter_data)
    nghostvertices = length(ghost_vertex_offset) - 1

    if gidx ≤ nlevels * nfidx * nghostvertices
        (level, fidx, ghostvertexidx) =
            cart_ind((nlevels, nfidx, nghostvertices), gidx).I
        idxresult, lvertresult = repr_ghost_vertex[ghostvertexidx]
        ipresult = perimeter_vertex_node_index(lvertresult)
        result = pperimeter_data[level, ipresult, fidx, idxresult]
        st, en = ghost_vertex_offset[ghostvertexidx],
        ghost_vertex_offset[ghostvertexidx + 1]
        for vertexidx in st:(en - 1)
            isghost, eidx, lvert = ghost_vertices[vertexidx]
            if !isghost
                ip = perimeter_vertex_node_index(lvert)
                pperimeter_data[level, ip, fidx, eidx] = result
            end
        end
    end
    return nothing
end
