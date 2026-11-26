import ClimaCore: DataLayouts, Topologies

# Generic block-size cap; its callers are the DG volume and remapping kernels.
_max_threads_cuda() = 256

# The DSS perimeter kernels have their own cap (CLIMA_DSS_MAX_THREADS
# overrides it), so tuning DSS does not reconfigure the other kernels.
function dss_config(nitems)
    config = linear_partition(nitems, DSS_MAX_THREADS[])
    return (; threads_s = config.threads, blocks_s = config.blocks)
end

function Topologies.dss_transform!(
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIJHWithF,
    data::DataLayouts.VIJHWithF,
    perimeter::Topologies.Perimeter2D,
    local_geometry::DataLayouts.VIJHWithF,
    dss_weights::DataLayouts.VIJHWithF,
    localelems,
)
    Nv = DataLayouts.nlevels(perimeter_data)
    nitems = Nv * length(perimeter) * length(localelems)
    iszero(nitems) && return nothing
    auto_launch!(
        (perimeter_data, data, local_geometry, dss_weights, localelems);
        dss_config(nitems)...,
    ) do perimeter_data, data, local_geometry, dss_weights, localelems
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= nitems
            (v, p, elem_index) =
                CartesianIndices((Nv, length(perimeter), length(localelems)))[gidx].I
            (i, j) = perimeter[p]
            h = localelems[elem_index]
            # dss_weights only vary in the horizontal, so their level index is 1
            perimeter_data[v, p, 1, h] = Topologies.dss_transform(
                data[v, i, j, h],
                local_geometry[v, i, j, h],
                dss_weights[1, i, j, h],
            )
        end
        nothing
    end
end

function Topologies.dss_untransform!(
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIJHWithF,
    data::DataLayouts.VIJHWithF,
    local_geometry::DataLayouts.VIJHWithF,
    perimeter::Topologies.Perimeter2D,
    localelems,
)
    Nv = DataLayouts.nlevels(perimeter_data)
    nitems = Nv * length(perimeter) * length(localelems)
    iszero(nitems) && return nothing
    auto_launch!(
        (perimeter_data, data, local_geometry, localelems);
        dss_config(nitems)...,
    ) do perimeter_data, data, local_geometry, localelems
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= nitems
            (v, p, elem_index) =
                CartesianIndices((Nv, length(perimeter), length(localelems)))[gidx].I
            (i, j) = perimeter[p]
            h = localelems[elem_index]
            data[v, i, j, h] = Topologies.dss_untransform(
                eltype(data),
                perimeter_data[v, p, 1, h],
                local_geometry[v, i, j, h],
            )
        end
        nothing
    end
end

function Topologies.dss_load_perimeter_data!(
    ::ClimaComms.CUDADevice,
    (; perimeter_data)::Topologies.DSSBuffer,
    data::DataLayouts.VIJHWithF,
    perimeter::Topologies.Perimeter2D,
)
    nitems = length(perimeter_data)
    auto_launch!((perimeter_data, data); dss_config(nitems)...) do perimeter_data, data
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= nitems
            (v, p, _, h) = CartesianIndices(perimeter_data)[gidx].I
            (i, j) = perimeter[p]
            perimeter_data[v, p, 1, h] = data[v, i, j, h]
        end
        nothing
    end
end

function Topologies.dss_unload_perimeter_data!(
    ::ClimaComms.CUDADevice,
    data::DataLayouts.VIJHWithF,
    (; perimeter_data)::Topologies.DSSBuffer,
    perimeter::Topologies.Perimeter2D,
)
    nitems = length(perimeter_data)
    auto_launch!((data, perimeter_data); dss_config(nitems)...) do data, perimeter_data
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= nitems
            (v, p, _, h) = CartesianIndices(perimeter_data)[gidx].I
            (i, j) = perimeter[p]
            data[v, i, j, h] = perimeter_data[v, p, 1, h]
        end
        nothing
    end
end

function Topologies.dss_local!(
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIJHWithF,
    perimeter::Topologies.Perimeter2D,
    (; local_vertices, local_vertex_offset, interior_faces)::Topologies.Topology2D,
)
    Nv = DataLayouts.nlevels(perimeter_data)
    nlocalvertices = length(local_vertex_offset) - 1
    nlocalfaces = length(interior_faces)
    nitems = Nv * (nlocalfaces + nlocalvertices)
    iszero(nitems) && return nothing
    auto_launch!(
        (perimeter_data, local_vertices, local_vertex_offset, interior_faces);
        dss_config(nitems)...,
    ) do perimeter_data, local_vertices, local_vertex_offset, interior_faces
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= Nv * nlocalvertices
            (v, vertex_index) = CartesianIndices((Nv, nlocalvertices))[gidx].I
            first_offset = local_vertex_offset[vertex_index]
            last_offset = local_vertex_offset[vertex_index + 1] - 1
            # Accumulate in a loop: sum with a closure would box v (also assigned in the
            # other branch), and sum's empty-collection error path cannot be compiled in
            # a GPU kernel. Every vertex has a local_vertices entry, so the loop is
            # nonempty.
            (h, vert) = local_vertices[first_offset]
            p = Topologies.perimeter_vertex_node_index(vert)
            sum_data = perimeter_data[v, p, 1, h]
            for offset in (first_offset + 1):last_offset
                (h, vert) = local_vertices[offset]
                p = Topologies.perimeter_vertex_node_index(vert)
                sum_data += perimeter_data[v, p, 1, h]
            end
            for offset in first_offset:last_offset
                (h, vert) = local_vertices[offset]
                p = Topologies.perimeter_vertex_node_index(vert)
                perimeter_data[v, p, 1, h] = sum_data
            end
        elseif gidx <= nitems
            (v, face_index) =
                CartesianIndices((Nv, nlocalfaces))[gidx - Nv * nlocalvertices].I
            (h1, face1, h2, face2, reversed) = interior_faces[face_index]
            nfacedof = length(perimeter) ÷ 4 - 1
            pr1 = Topologies.perimeter_face_indices(face1, nfacedof, false)
            pr2 = Topologies.perimeter_face_indices(face2, nfacedof, reversed)
            for (p1, p2) in zip(pr1, pr2)
                sum_data = perimeter_data[v, p1, 1, h1] + perimeter_data[v, p2, 1, h2]
                perimeter_data[v, p1, 1, h1] = sum_data
                perimeter_data[v, p2, 1, h2] = sum_data
            end
        end
        nothing
    end
end

function Topologies.dss_local_ghost!(
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIJHWithF,
    perimeter::Topologies.Perimeter2D,
    (; ghost_vertices, ghost_vertex_offset)::Topologies.Topology2D,
)
    Nv = DataLayouts.nlevels(perimeter_data)
    nghostvertices = length(ghost_vertex_offset) - 1
    nitems = Nv * nghostvertices
    iszero(nitems) && return nothing
    auto_launch!(
        (perimeter_data, ghost_vertices, ghost_vertex_offset);
        dss_config(nitems)...,
    ) do perimeter_data, ghost_vertices, ghost_vertex_offset
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= nitems
            (v, vertex_index) = CartesianIndices((Nv, nghostvertices))[gidx].I
            first_offset = ghost_vertex_offset[vertex_index]
            last_offset = ghost_vertex_offset[vertex_index + 1] - 1
            # Accumulate in a loop instead of calling sum with a closure, since
            # the empty-collection error path of sum cannot be compiled in a
            # GPU kernel.
            sum_data = zero(eltype(perimeter_data))
            for offset in first_offset:last_offset
                (isghost, h, vert) = ghost_vertices[offset]
                isghost && continue
                p = Topologies.perimeter_vertex_node_index(vert)
                sum_data += perimeter_data[v, p, 1, h]
            end
            for offset in first_offset:last_offset
                (isghost, h, vert) = ghost_vertices[offset]
                isghost && continue
                p = Topologies.perimeter_vertex_node_index(vert)
                perimeter_data[v, p, 1, h] = sum_data
            end
        end
        nothing
    end
end

function Topologies.dss_ghost!(
    ::ClimaComms.CUDADevice,
    perimeter_data::DataLayouts.VIJHWithF,
    perimeter::Topologies.Perimeter2D,
    (; ghost_vertices, ghost_vertex_offset, repr_ghost_vertex)::Topologies.Topology2D,
)
    Nv = DataLayouts.nlevels(perimeter_data)
    nghostvertices = length(ghost_vertex_offset) - 1
    nitems = Nv * nghostvertices
    iszero(nitems) && return nothing
    auto_launch!(
        (perimeter_data, ghost_vertices, ghost_vertex_offset, repr_ghost_vertex);
        dss_config(nitems)...,
    ) do perimeter_data, ghost_vertices, ghost_vertex_offset, repr_ghost_vertex
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= nitems
            (v, vertex_index) = CartesianIndices((Nv, nghostvertices))[gidx].I
            h_result, vert_result = repr_ghost_vertex[vertex_index]
            p_result = Topologies.perimeter_vertex_node_index(vert_result)
            result = perimeter_data[v, p_result, 1, h_result]
            first_offset = ghost_vertex_offset[vertex_index]
            last_offset = ghost_vertex_offset[vertex_index + 1] - 1
            for offset in first_offset:last_offset
                (isghost, h, vert) = ghost_vertices[offset]
                isghost && continue
                p = Topologies.perimeter_vertex_node_index(vert)
                perimeter_data[v, p, 1, h] = result
            end
        end
        nothing
    end
end

# Enqueues the copy without synchronizing; the caller synchronizes once after
# filling all send buffers, before MPI reads them on its own stream.
function Topologies.fill_send_buffer!(
    ::ClimaComms.CUDADevice,
    (; perimeter_data, send_buf_idx, send_data, buf_offset)::Topologies.DSSBuffer,
)
    Nv = DataLayouts.nlevels(perimeter_data)
    Nf = DataLayouts.ncomponents(perimeter_data)
    nsend = size(send_buf_idx, 1)
    nitems = Nv * nsend
    iszero(nitems) && return nothing
    auto_launch!(
        (perimeter_data, send_data, send_buf_idx, buf_offset);
        dss_config(nitems)...,
    ) do perimeter_data, send_data, send_buf_idx, buf_offset
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= nitems
            (v, send_index) = CartesianIndices((Nv, nsend))[gidx].I
            # Avoid indexing with a colon, which would allocate in the kernel.
            (h, p) = (send_buf_idx[send_index, 1], send_buf_idx[send_index, 2])
            item = perimeter_data[v, p, 1, h]
            buffer_index = buf_offset[send_index] + (v - 1) * Nf + 1
            DataLayouts.set_struct!(send_data, item, buffer_index, Val(1))
        end
        nothing
    end
end

# One thread per level and receiving perimeter node adds that node's receive rows
# in ascending order, the order of the CPU loop, so the result is deterministic
# and matches the CPU bitwise even where several processes send to one node.
function Topologies.load_from_recv_buffer!(
    ::ClimaComms.CUDADevice,
    (;
        perimeter_data,
        recv_data,
        buf_offset,
        recv_target_idx,
        recv_target_offset,
        recv_target_rows,
    )::Topologies.DSSBuffer,
)
    Nv = DataLayouts.nlevels(perimeter_data)
    Nf = DataLayouts.ncomponents(perimeter_data)
    ntargets = size(recv_target_idx, 1)
    nitems = Nv * ntargets
    iszero(nitems) && return nothing
    auto_launch!(
        (
            perimeter_data,
            recv_data,
            buf_offset,
            recv_target_idx,
            recv_target_offset,
            recv_target_rows,
        );
        dss_config(nitems)...,
    ) do perimeter_data,
    recv_data,
    buf_offset,
    recv_target_idx,
    recv_target_offset,
    recv_target_rows
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= nitems
            T = eltype(perimeter_data)
            (v, target) = CartesianIndices((Nv, ntargets))[gidx].I
            (h, p) = (recv_target_idx[target, 1], recv_target_idx[target, 2])
            sum_data = perimeter_data[v, p, 1, h]
            for n in recv_target_offset[target]:(recv_target_offset[target + 1] - 1)
                buffer_index = buf_offset[recv_target_rows[n]] + (v - 1) * Nf + 1
                sum_data += DataLayouts.get_struct(recv_data, T, buffer_index, Val(1))
            end
            perimeter_data[v, p, 1, h] = sum_data
        end
        nothing
    end
end

function Topologies.dss_1d!(
    ::ClimaComms.CUDADevice,
    data::DataLayouts.VIJHWithF,
    topology::Topologies.IntervalTopology,
    local_geometry = nothing,
    dss_weights = nothing,
)
    Nv = DataLayouts.nlevels(data)
    Ni = DataLayouts.nquadpoints(data)
    Nh = DataLayouts.nelems(data)
    nfaces = Topologies.isperiodic(topology) ? Nh : Nh - 1
    nitems = Nv * nfaces
    iszero(nitems) && return nothing
    auto_launch!(
        (Base.broadcastable(data), local_geometry, dss_weights);
        dss_config(nitems)...,
    ) do data, local_geometry, dss_weights
        gidx = DataLayouts.thread_rank(ThisKernel())
        @inbounds if gidx <= nitems
            T = eltype(data)
            (v, h) = CartesianIndices((Nv, nfaces))[gidx].I
            I1 = CartesianIndex(v, Ni, 1, h)
            I2 = CartesianIndex(v, 1, 1, h == Nh ? 1 : h + 1)
            sum_data =
                Topologies.dss_transform(data, local_geometry, dss_weights, I1) +
                Topologies.dss_transform(data, local_geometry, dss_weights, I2)
            data[I1] = Topologies.dss_untransform(T, sum_data, local_geometry, I1)
            data[I2] = Topologies.dss_untransform(T, sum_data, local_geometry, I2)
        end
        nothing
    end
end
