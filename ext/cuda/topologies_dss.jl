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
        data_idx = linear_ind(sized, (level, ip, jp, elem))
        pperimeter_data.arrays[fidx][level, p, elem] =
            pdata.arrays[fidx][data_idx]
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
        pdata.arrays[fidx][data_idx] =
            pperimeter_data.arrays[fidx][level, p, elem]
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
    perimeter::Topologies.Perimeter2D{Nq},
) where {Nq}
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
            sum_data += pperimeter_data.arrays[fidx][level, ip, lidx]
        end
        for idx in st:(en - 1)
            (lidx, vert) = local_vertices[idx]
            ip = perimeter_vertex_node_index(vert)
            pperimeter_data.arrays[fidx][level, ip, lidx] = sum_data
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
                pperimeter_data.arrays[fidx][level, ip1, lidx1] +
                pperimeter_data.arrays[fidx][level, ip2, lidx2]
            pperimeter_data.arrays[fidx][level, ip1, lidx1] = val
            pperimeter_data.arrays[fidx][level, ip2, lidx2] = val
        end
    end

    return nothing
end

function Topologies.dss_transform!(
    device::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂ξ∂x::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂x∂ξ::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    weight::DataLayouts.IJFH,
    perimeter::Topologies.Perimeter2D,
    scalarfidx::AbstractVector{Int},
    covariant12fidx::AbstractVector{Int},
    contravariant12fidx::AbstractVector{Int},
    covariant123fidx::AbstractVector{Int},
    contravariant123fidx::AbstractVector{Int},
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
        (nlevels, nperimeter, _, _) = DataLayouts.array_size(perimeter_data)
        nitems = nlevels * nperimeter * nlocalelems
        nthreads, nblocks = _configure_threadblock(nitems)
        args = (
            perimeter_data,
            pdata,
            p∂ξ∂x,
            p∂x∂ξ,
            Val(nmetric),
            pweight,
            perimeter,
            scalarfidx,
            covariant12fidx,
            contravariant12fidx,
            covariant123fidx,
            contravariant123fidx,
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
    pdata::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    p∂ξ∂x::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    p∂x∂ξ::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    ::Val{nmetric},
    pweight::AbstractArray{FT, 4},
    perimeter::Topologies.Perimeter2D{Nq},
    scalarfidx::AbstractVector{Int},
    covariant12fidx::AbstractVector{Int},
    contravariant12fidx::AbstractVector{Int},
    covariant123fidx::AbstractVector{Int},
    contravariant123fidx::AbstractVector{Int},
    localelems::AbstractVector{Int},
    ::Val{nlocalelems},
) where {FT <: AbstractFloat, Nq, nmetric, nlocalelems}
    pperimeter_data = parent(perimeter_data)
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    (nlevels, nperimeter, nfid, nelems) =
        DataLayouts.farray_size(perimeter_data)
    if gidx ≤ nlevels * nperimeter * nlocalelems
        sizet = (nlevels, nperimeter, nlocalelems)
        sizet_data = (nlevels, Nq, Nq, nfid, nelems)
        sizet_wt = (Nq, Nq, nelems)
        sizet_metric = (nlevels, Nq, Nq, nmetric, nelems)

        (level, p, localelemno) = cart_ind(sizet, gidx).I
        elem = localelems[localelemno]
        (ip, jp) = perimeter[p]

        weight = pweight[linear_ind(sizet_wt, (ip, jp, 1, elem))]
        for fidx in scalarfidx
            data_idx = linear_ind(sizet_data, (level, ip, jp, fidx, elem))
            pperimeter_data[level, p, fidx, elem] = pdata[data_idx] * weight
        end
        for fidx in covariant12fidx
            data_idx = linear_ind(sizet_data, (level, ip, jp, elem))
            (idx11, idx12, idx21, idx22) = (1,2,3,4)
                # Topologies._get_idx_metric(sizet_metric, (level, ip, jp, elem))
            pperimeter_data[level, p, fidx, elem] =
                (
                    p∂ξ∂x.arrays[idx11][data_idx] * pdata.arrays[fidx][data_idx] +
                    p∂ξ∂x.arrays[idx12][data_idx] * pdata.arrays[fidx+1][data_idx]
                ) * weight
            pperimeter_data[level, p, fidx + 1, elem] =
                (
                    p∂ξ∂x.arrays[idx21][data_idx] * pdata.arrays[fidx][data_idx] +
                    p∂ξ∂x.arrays[idx22][data_idx] * pdata.arrays[fidx+1][data_idx]
                ) * weight
        end
        for fidx in contravariant12fidx
            data_idx = linear_ind(sizet_data, (level, ip, jp, elem))
            (idx11, idx12, idx21, idx22) = (1,2,3,4)
                # Topologies._get_idx_metric(sizet_metric, (level, ip, jp, elem))
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
        for fidx in covariant123fidx
            data_idx1 =
                Topologies.linear_ind(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = Topologies.linear_ind(
                sizet_data,
                (level, ip, jp, fidx + 1, elem),
            )
            data_idx3 = Topologies.linear_ind(
                sizet_data,
                (level, ip, jp, fidx + 2, elem),
            )

            (idx11, idx12, idx13, idx21, idx22, idx23, idx31, idx32, idx33) =
                Topologies._get_idx_metric_3d(
                    sizet_metric,
                    (level, ip, jp, elem),
                )

            # Covariant to physical transformation
            pperimeter_data[level, p, fidx, elem] =
                (
                    p∂ξ∂x[idx11] * pdata[data_idx1] +
                    p∂ξ∂x[idx12] * pdata[data_idx2] +
                    p∂ξ∂x[idx13] * pdata[data_idx3]
                ) * weight
            pperimeter_data[level, p, fidx + 1, elem] =
                (
                    p∂ξ∂x[idx21] * pdata[data_idx1] +
                    p∂ξ∂x[idx22] * pdata[data_idx2] +
                    p∂ξ∂x[idx23] * pdata[data_idx3]
                ) * weight
            pperimeter_data[level, p, fidx + 2, elem] =
                (
                    p∂ξ∂x[idx31] * pdata[data_idx1] +
                    p∂ξ∂x[idx32] * pdata[data_idx2] +
                    p∂ξ∂x[idx33] * pdata[data_idx3]
                ) * weight
        end

        for fidx in contravariant123fidx
            data_idx1 =
                Topologies.linear_ind(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = Topologies.linear_ind(
                sizet_data,
                (level, ip, jp, fidx + 1, elem),
            )
            data_idx3 = Topologies.linear_ind(
                sizet_data,
                (level, ip, jp, fidx + 2, elem),
            )
            (idx11, idx12, idx13, idx21, idx22, idx23, idx31, idx32, idx33) =
                Topologies._get_idx_metric_3d(
                    sizet_metric,
                    (level, ip, jp, elem),
                )
            # Contravariant to physical transformation
            pperimeter_data[level, p, fidx, elem] =
                (
                    p∂x∂ξ[idx11] * pdata[data_idx1] +
                    p∂x∂ξ[idx21] * pdata[data_idx2] +
                    p∂x∂ξ[idx31] * pdata[data_idx3]
                ) * weight
            pperimeter_data[level, p, fidx + 1, elem] =
                (
                    p∂x∂ξ[idx12] * pdata[data_idx1] +
                    p∂x∂ξ[idx22] * pdata[data_idx2] +
                    p∂x∂ξ[idx32] * pdata[data_idx3]
                ) * weight
            pperimeter_data[level, p, fidx + 2, elem] =
                (
                    p∂x∂ξ[idx13] * pdata[data_idx1] +
                    p∂x∂ξ[idx23] * pdata[data_idx2] +
                    p∂x∂ξ[idx33] * pdata[data_idx3]
                ) * weight
        end
    end
    return nothing
end

function Topologies.dss_untransform!(
    device::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIFH,
    data::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂ξ∂x::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    ∂x∂ξ::Union{DataLayouts.VIJFH, DataLayouts.IJFH},
    perimeter::Topologies.Perimeter2D{Nq},
    scalarfidx::AbstractVector{Int},
    covariant12fidx::AbstractVector{Int},
    contravariant12fidx::AbstractVector{Int},
    covariant123fidx::AbstractVector{Int},
    contravariant123fidx::AbstractVector{Int},
    localelems::AbstractVector{Int},
) where {Nq}
    nlocalelems = length(localelems)
    if nlocalelems > 0
        pdata = parent(data)
        p∂x∂ξ = parent(∂x∂ξ)
        p∂ξ∂x = parent(∂ξ∂x)
        nmetric = cld(length(p∂ξ∂x), prod(size(∂ξ∂x)))
        (nlevels, nperimeter, _, _) = DataLayouts.array_size(perimeter_data)
        nitems = nlevels * nperimeter * nlocalelems
        nthreads, nblocks = _configure_threadblock(nitems)
        args = (
            perimeter_data,
            pdata,
            p∂ξ∂x,
            p∂x∂ξ,
            nmetric,
            perimeter,
            scalarfidx,
            covariant12fidx,
            contravariant12fidx,
            covariant123fidx,
            contravariant123fidx,
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
    pdata::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    p∂ξ∂x::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    p∂x∂ξ::Union{AbstractArray{FT, 4}, AbstractArray{FT, 5}},
    nmetric::Int,
    perimeter::Topologies.Perimeter2D{Nq},
    scalarfidx::AbstractVector{Int},
    covariant12fidx::AbstractVector{Int},
    contravariant12fidx::AbstractVector{Int},
    covariant123fidx::AbstractVector{Int},
    contravariant123fidx::AbstractVector{Int},
    localelems::AbstractVector{Int},
    ::Val{nlocalelems},
) where {FT <: AbstractFloat, Nq, nlocalelems}
    gidx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    (nlevels, nperimeter, nfid, nelems) =
        DataLayouts.farray_size(perimeter_data)
    pperimeter_data = parent(perimeter_data)
    if gidx ≤ nlevels * nperimeter * nlocalelems
        sizet = (nlevels, nperimeter, nlocalelems)
        sizet_data = (nlevels, Nq, Nq, nfid, nelems)
        sizet_wt = (Nq, Nq, 1, nelems)
        sizet_metric = (nlevels, Nq, Nq, nmetric, nelems)

        (level, p, localelemno) = cart_ind(sizet, gidx).I
        elem = localelems[localelemno]
        ip, jp = perimeter[p]
        for fidx in scalarfidx
            data_idx = linear_ind(sizet_data, (level, ip, jp, fidx, elem))
            pdata[data_idx] = pperimeter_data[level, p, fidx, elem]
        end
        for fidx in covariant12fidx
            data_idx1 = linear_ind(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = linear_ind(sizet_data, (level, ip, jp, fidx + 1, elem))
            (idx11, idx12, idx21, idx22) =
                Topologies._get_idx_metric(sizet_metric, (level, ip, jp, elem))
            pdata[data_idx1] =
                p∂x∂ξ[idx11] * pperimeter_data[level, p, fidx, elem] +
                p∂x∂ξ[idx12] * pperimeter_data[level, p, fidx + 1, elem]
            pdata[data_idx2] =
                p∂x∂ξ[idx21] * pperimeter_data[level, p, fidx, elem] +
                p∂x∂ξ[idx22] * pperimeter_data[level, p, fidx + 1, elem]
        end
        for fidx in contravariant12fidx
            data_idx1 = linear_ind(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = linear_ind(sizet_data, (level, ip, jp, fidx + 1, elem))
            (idx11, idx12, idx21, idx22) =
                Topologies._get_idx_metric(sizet_metric, (level, ip, jp, elem))
            pdata[data_idx1] =
                p∂ξ∂x[idx11] * pperimeter_data[level, p, fidx, elem] +
                p∂ξ∂x[idx21] * pperimeter_data[level, p, fidx + 1, elem]
            pdata[data_idx2] =
                p∂ξ∂x[idx12] * pperimeter_data[level, p, fidx, elem] +
                p∂ξ∂x[idx22] * pperimeter_data[level, p, fidx + 1, elem]
        end
        for fidx in covariant123fidx
            data_idx1 = linear_ind(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = linear_ind(sizet_data, (level, ip, jp, fidx + 1, elem))
            data_idx3 = linear_ind(sizet_data, (level, ip, jp, fidx + 2, elem))
            (idx11, idx12, idx13, idx21, idx22, idx23, idx31, idx32, idx33) =
                Topologies._get_idx_metric_3d(
                    sizet_metric,
                    (level, ip, jp, elem),
                )
            pdata[data_idx1] =
                p∂x∂ξ[idx11] * pperimeter_data[level, p, fidx, elem] +
                p∂x∂ξ[idx12] * pperimeter_data[level, p, fidx + 1, elem] +
                p∂x∂ξ[idx13] * pperimeter_data[level, p, fidx + 2, elem]
            pdata[data_idx2] =
                p∂x∂ξ[idx21] * pperimeter_data[level, p, fidx, elem] +
                p∂x∂ξ[idx22] * pperimeter_data[level, p, fidx + 1, elem] +
                p∂x∂ξ[idx23] * pperimeter_data[level, p, fidx + 2, elem]
            pdata[data_idx3] =
                p∂x∂ξ[idx31] * pperimeter_data[level, p, fidx, elem] +
                p∂x∂ξ[idx32] * pperimeter_data[level, p, fidx + 1, elem] +
                p∂x∂ξ[idx33] * pperimeter_data[level, p, fidx + 2, elem]
        end
        for fidx in contravariant123fidx
            data_idx1 = linear_ind(sizet_data, (level, ip, jp, fidx, elem))
            data_idx2 = linear_ind(sizet_data, (level, ip, jp, fidx + 1, elem))
            data_idx3 = linear_ind(sizet_data, (level, ip, jp, fidx + 2, elem))
            (idx11, idx12, idx13, idx21, idx22, idx23, idx31, idx32, idx33) =
                Topologies._get_idx_metric_3d(
                    sizet_metric,
                    (level, ip, jp, elem),
                )
            pdata[data_idx1] =
                p∂ξ∂x[idx11] * pperimeter_data[level, p, fidx, elem] +
                p∂ξ∂x[idx21] * pperimeter_data[level, p, fidx + 1, elem] +
                p∂ξ∂x[idx31] * pperimeter_data[level, p, fidx + 2, elem]
            pdata[data_idx2] =
                p∂ξ∂x[idx12] * pperimeter_data[level, p, fidx, elem] +
                p∂ξ∂x[idx22] * pperimeter_data[level, p, fidx + 1, elem]
            p∂ξ∂x[idx32] * pperimeter_data[level, p, fidx + 2, elem]
            pdata[data_idx3] =
                p∂ξ∂x[idx13] * pperimeter_data[level, p, fidx, elem] +
                p∂ξ∂x[idx23] * pperimeter_data[level, p, fidx + 1, elem] +
                p∂ξ∂x[idx33] * pperimeter_data[level, p, fidx + 2, elem]
        end
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
    perimeter::Topologies.Perimeter2D{Nq},
) where {Nq}
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
        CUDA.@atomic pperimeter_data.arrays[fidx][level, ip, lidx] +=
            recv_data[idx]
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
    perimeter::Topologies.Perimeter2D{Nq},
) where {Nq}
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
