import ..Topologies
using ..RecursiveApply

@inline function dss_transform(arg, local_geometry)
    RecursiveApply.rmap(arg) do x
        Base.@_inline_meta
        dss_transform(x, local_geometry)
    end
end
@inline dss_transform(arg::Number, local_geometry) = arg
@inline dss_transform(
    arg::Geometry.AxisTensor{T, N, <:Tuple{Vararg{Geometry.CartesianAxis}}},
    local_geometry::Geometry.LocalGeometry,
) where {T, N} = arg
@inline dss_transform(
    arg::Geometry.CartesianVector,
    local_geometry::Geometry.LocalGeometry,
) where {T, N} = arg
@inline dss_transform(
    arg::Geometry.AxisTensor{T, N, <:Tuple{Vararg{Geometry.LocalAxis}}},
    local_geometry::Geometry.LocalGeometry,
) where {T, N} = arg
@inline dss_transform(
    arg::Geometry.LocalVector,
    local_geometry::Geometry.LocalGeometry,
) where {T, N} = arg

@inline function dss_transform(
    arg::Geometry.AxisVector,
    local_geometry::Geometry.LocalGeometry,
)
    ax = axes(local_geometry.∂x∂ξ, 1)
    axfrom = axes(arg, 1)
    # TODO: make this consistent for 2D / 3D
    # 2D domain axis (1,2) horizontal curl
    if ax isa Geometry.UVAxis && (
        axfrom isa Geometry.Covariant3Axis ||
        axfrom isa Geometry.Contravariant3Axis
    )
        return arg
    end
    # 2D domain axis (1,3) curl
    if ax isa Geometry.UWAxis && (
        axfrom isa Geometry.Covariant2Axis ||
        axfrom isa Geometry.Contravariant2Axis
    )
        return arg
    end
    Geometry.transform(ax, arg, local_geometry)
end

@inline function dss_untransform(refarg, targ, local_geometry)
    RecursiveApply.rmap(refarg, targ) do rx, tx
        Base.@_inline_meta
        dss_untransform(rx, tx, local_geometry)
    end
end
@inline dss_untransform(refarg::Number, targ, local_geometry) = targ
@inline dss_untransform(
    refarg::T,
    targ::T,
    local_geometry::Geometry.LocalGeometry,
) where {T <: Geometry.AxisTensor} = targ
@inline function dss_untransform(
    refarg::Geometry.AxisTensor,
    targ::Geometry.AxisTensor,
    local_geometry::Geometry.LocalGeometry,
)
    ax = axes(refarg, 1)
    Geometry.transform(ax, targ, local_geometry)
end

function dss_1d!(
    dest,
    src,
    local_geometry_data,
    htopology::Topologies.AbstractTopology,
    Nq::Int,
    Nv::Int = 1,
)
    idx1 = CartesianIndex(1, 1, 1, 1, 1)
    idx2 = CartesianIndex(Nq, 1, 1, 1, 1)
    for (elem1, face1, elem2, face2, reversed) in
        Topologies.interior_faces(htopology)
        for level in 1:Nv
            @assert face1 == 1 && face2 == 2 && !reversed
            local_geometry_slab1 = slab(local_geometry_data, level, elem1)
            src_slab1 = slab(src, level, elem1)
            local_geometry_slab2 = slab(local_geometry_data, level, elem2)
            src_slab2 = slab(src, level, elem2)
            val =
                dss_transform(src_slab1[idx1], local_geometry_slab1[idx1]) ⊞
                dss_transform(src_slab2[idx2], local_geometry_slab2[idx2])

            dest_slab1 = slab(dest, level, elem1)
            dest_slab2 = slab(dest, level, elem2)

            dest_slab1[idx1] = dss_untransform(
                dest_slab1[idx1],
                val,
                local_geometry_slab1[idx1],
            )
            dest_slab2[idx2] = dss_untransform(
                dest_slab2[idx2],
                val,
                local_geometry_slab2[idx2],
            )
        end
    end
    return dest
end

function dss_2d!(
    dest,
    src,
    local_geometry_data,
    ghost_geometry_data,
    topology::Topologies.AbstractTopology,
    Nq::Int,
    Nv::Int = 1,
    comms_ctx = nothing,
)
    @assert ghost_geometry_data === nothing && comms_ctx === nothing

    # TODO: generalize to extruded domains by returning a cartesian index?
    # iterate over the interior faces for each element of the mesh
    for (elem1, face1, elem2, face2, reversed) in
        Topologies.interior_faces(topology)
        for level in 1:Nv
            # iterate over non-vertex nodes
            src_slab1 = slab(src, level, elem1)
            src_slab2 = slab(src, level, elem2)
            local_geometry_slab1 = slab(local_geometry_data, level, elem1)
            local_geometry_slab2 = slab(local_geometry_data, level, elem2)
            dest_slab1 = slab(dest, level, elem1)
            dest_slab2 = slab(dest, level, elem2)
            for q in 2:(Nq - 1)
                i1, j1 = Topologies.face_node_index(face1, Nq, q, false)
                i2, j2 = Topologies.face_node_index(face2, Nq, q, reversed)
                val =
                    dss_transform(
                        src_slab1[i1, j1],
                        local_geometry_slab1[i1, j1],
                    ) ⊞ dss_transform(
                        src_slab2[i2, j2],
                        local_geometry_slab2[i2, j2],
                    )
                dest_slab1[i1, j1] = dss_untransform(
                    dest_slab1[i1, j1],
                    val,
                    local_geometry_slab1[i1, j1],
                )
                dest_slab2[i2, j2] = dss_untransform(
                    dest_slab2[i2, j2],
                    val,
                    local_geometry_slab2[i2, j2],
                )
            end
        end
    end

    # iterate over all vertices
    for vertex in Topologies.vertices(topology)
        # for each level
        for level in 1:Nv
            # gather: compute sum over shared vertices
            sum_data = mapreduce(⊞, vertex) do (elem, vertex_num)
                i, j = Topologies.vertex_node_index(vertex_num, Nq)
                local_geometry_slab = slab(local_geometry_data, level, elem)
                src_slab = slab(src, level, elem)
                dss_transform(src_slab[i, j], local_geometry_slab[i, j])
            end

            # scatter: assign sum to shared vertices
            for (elem, vertex_num) in vertex
                dest_slab = slab(dest, level, elem)
                i, j = Topologies.vertex_node_index(vertex_num, Nq)
                local_geometry_slab = slab(local_geometry_data, level, elem)
                dest_slab[i, j] = dss_untransform(
                    dest_slab[i, j],
                    sum_data,
                    local_geometry_slab[i, j],
                )
            end
        end
    end
    return dest
end

function dss_2d!(
    dest,
    src,
    local_geometry_data,
    ghost_geometry_data,
    topology::Topologies.DistributedTopology2D,
    Nq::Int,
    Nv::Int = 1,
    comms_ctx = nothing,
)
    if Topologies.nneighbors(topology) > 0
        @assert comms_ctx !== nothing
    end
    pid = ClimaComms.mypid(comms_ctx)
    elt = eltype(dest)

    # prepare send buffers from send elements
    for nbr_idx in 1:Topologies.nneighbors(topology)
        sbuf = ClimaComms.send_stage(ClimaComms.neighbors(comms_ctx)[nbr_idx])
        for level in 1:Nv
            senum = 1
            for sendelem in topology.send_elems[nbr_idx]
                sidx = Topologies.localelemindex(topology, sendelem)
                src_slab = slab(src, level, sidx)
                dest_slab =
                    DataLayouts.IJF{elt, Nq}(view(sbuf, level, :, :, :, senum))
                copyto!(dest_slab, src_slab)
                senum += 1
            end
        end
    end

    # start the communication
    ClimaComms.start(comms_ctx)

    # DSS over the interior faces and vertices
    # TODO: generalize to extruded domains by returning a cartesian index?
    # iterate over the interior faces for each element of the mesh
    for (e1, face1, e2, face2, reversed) in Topologies.interior_faces(topology)
        for level in 1:Nv
            # iterate over non-vertex nodes
            e1idx = Topologies.localelemindex(topology, e1)
            e2idx = Topologies.localelemindex(topology, e2)
            src_slab1 = slab(src, level, e1idx)
            src_slab2 = slab(src, level, e2idx)
            local_geometry_slab1 = slab(local_geometry_data, level, e1idx)
            local_geometry_slab2 = slab(local_geometry_data, level, e2idx)
            dest_slab1 = slab(dest, level, e1idx)
            dest_slab2 = slab(dest, level, e2idx)
            for q in 2:(Nq - 1)
                i1, j1 = Topologies.face_node_index(face1, Nq, q, false)
                i2, j2 = Topologies.face_node_index(face2, Nq, q, reversed)
                val =
                    dss_transform(
                        src_slab1[i1, j1],
                        local_geometry_slab1[i1, j1],
                    ) ⊞ dss_transform(
                        src_slab2[i2, j2],
                        local_geometry_slab2[i2, j2],
                    )
                dest_slab1[i1, j1] = dss_untransform(
                    dest_slab1[i1, j1],
                    val,
                    local_geometry_slab1[i1, j1],
                )
                dest_slab2[i2, j2] = dss_untransform(
                    dest_slab2[i2, j2],
                    val,
                    local_geometry_slab2[i2, j2],
                )
            end
        end
    end

    # progress communication
    ClimaComms.progress(comms_ctx)

    # iterate over interior vertices
    for vertex in Topologies.interior_vertices(topology)
        # for each level
        for level in 1:Nv
            # gather: compute sum over shared vertices
            sum_data = mapreduce(⊞, vertex) do (elem, vertex_num)
                e = Topologies.localelemindex(topology, elem)
                i, j = Topologies.vertex_node_index(vertex_num, Nq)
                local_geometry_slab = slab(local_geometry_data, level, e)
                src_slab = slab(src, level, e)
                dss_transform(src_slab[i, j], local_geometry_slab[i, j])
            end

            # scatter: assign sum to shared vertices
            for (elem, vertex_num) in vertex
                e = Topologies.localelemindex(topology, elem)
                dest_slab = slab(dest, level, e)
                i, j = Topologies.vertex_node_index(vertex_num, Nq)
                local_geometry_slab = slab(local_geometry_data, level, e)
                dest_slab[i, j] = dss_untransform(
                    dest_slab[i, j],
                    sum_data,
                    local_geometry_slab[i, j],
                )
            end
        end
    end

    # complete communication
    ClimaComms.finish(comms_ctx)

    # DSS over ghost faces; this is set up such that `elem1` is local
    # and `elem2` is remote
    #  - potentially change to:
    #   (localelemnumber, face1, nbr_idx, gei, face2)
    for (e1, face1, e2, face2, reversed) in Topologies.ghost_faces(topology)

        # this is the local index for global order element 1
        lei = topology.localorderindex[e1]

        # find the index of the neighbor in global order that owns element 2
        elem2 = topology.elemorder[e2]
        opid = topology.elempid[e2]
        nbr_idx =
            findfirst(npid -> npid == opid, Topologies.neighbors(topology))

        # get the receive buffer for this neighbor
        rbuf = ClimaComms.recv_stage(ClimaComms.neighbors(comms_ctx)[nbr_idx])

        for level in 1:Nv
            src_slab1 = slab(src, level, lei)
            local_geometry_slab1 = slab(local_geometry_data, level, lei)

            # find the index of this element in the ghost elements for
            # this neighbor and use it to find the slab we want
            gei = findfirst(ge -> ge == elem2, topology.ghost_elems[nbr_idx])
            src_slab2 =
                DataLayouts.IJF{elt, Nq}(view(rbuf, level, :, :, :, gei))

            # unlike the ghost elements, the geometry data is not organized
            # by neighbor so we must compute the correct index into it
            geomei = gei
            for n in 1:(nbr_idx - 1)
                geomei += length(topology.ghost_elems[n])
            end
            ghost_geometry_slab2 = slab(ghost_geometry_data, level, geomei)

            dest_slab = slab(dest, level, lei)
            for q in 2:(Nq - 1)
                i1, j1 = Topologies.face_node_index(face1, Nq, q, false)
                i2, j2 = Topologies.face_node_index(face2, Nq, q, reversed)
                val =
                    dss_transform(
                        src_slab1[i1, j1],
                        local_geometry_slab1[i1, j1],
                    ) ⊞ dss_transform(
                        src_slab2[i2, j2],
                        ghost_geometry_slab2[i2, j2],
                    )

                dest_slab[i1, j1] = dss_untransform(
                    dest_slab[i1, j1],
                    val,
                    local_geometry_slab1[i1, j1],
                )
            end
        end
    end

    # DSS over ghost vertices
    # potentially change to an iterator over (nbr_idx, gei, vert) or (0, lei, vert)
    for verts in Topologies.ghost_vertices(topology)
        for level in 1:Nv
            # gather: compute sum over shared vertices
            sum_data = mapreduce(⊞, verts) do (e, vertex_num)
                i, j = Topologies.vertex_node_index(vertex_num, Nq)
                opid = topology.elempid[e]
                if opid == pid
                    lei = topology.localorderindex[e]
                    src_slab = slab(src, level, lei)
                    geometry_slab = slab(local_geometry_data, level, lei)
                else
                    elem = topology.elemorder[e]
                    nbr_idx = findfirst(
                        npid -> npid == opid,
                        Topologies.neighbors(topology),
                    )
                    rbuf = ClimaComms.recv_stage(
                        ClimaComms.neighbors(comms_ctx)[nbr_idx],
                    )
                    gei = findfirst(
                        ge -> ge == elem,
                        topology.ghost_elems[nbr_idx],
                    )
                    src_slab = DataLayouts.IJF{elt, Nq}(
                        view(rbuf, level, :, :, :, gei),
                    )
                    geomei = gei
                    for n in 1:(nbr_idx - 1)
                        geomei += length(topology.ghost_elems[n])
                    end
                    geometry_slab = slab(ghost_geometry_data, level, geomei)
                end
                dss_transform(src_slab[i, j], geometry_slab[i, j])
            end

            # scatter: assign sum to shared vertices
            for (e, vertex_num) in verts
                if topology.elempid[e] == pid
                    lei = topology.localorderindex[e]
                    dest_slab = slab(dest, level, lei)
                    i, j = Topologies.vertex_node_index(vertex_num, Nq)
                    local_geometry_slab = slab(local_geometry_data, level, lei)
                    dest_slab[i, j] = dss_untransform(
                        dest_slab[i, j],
                        sum_data,
                        local_geometry_slab[i, j],
                    )
                end
            end
        end
    end

    return dest
end

"""
    horizontal_dss!(dest, src, topology, Nq)

Apply horizontal direct stiffness summation (DSS) to `src`, storing the result in `dest`.
"""
function horizontal_dss!(dest, src, space::AbstractSpace, comms_ctx = nothing)
    if space isa ExtrudedFiniteDifferenceSpace
        Nv = nlevels(space)
        hspace = space.horizontal_space
    else
        Nv = 1
        hspace = space
    end
    htopology = hspace.topology
    Nq = Quadratures.degrees_of_freedom(hspace.quadrature_style)::Int
    if hspace isa SpectralElementSpace1D
        dss_1d!(dest, src, local_geometry_data(space), htopology, Nq, Nv)
    elseif hspace isa SpectralElementSpace2D
        dss_2d!(
            dest,
            src,
            local_geometry_data(space),
            ghost_geometry_data(space),
            htopology,
            Nq,
            Nv,
            comms_ctx,
        )
    end
end

horizontal_dss!(data, space::AbstractSpace, comms_ctx = nothing) =
    horizontal_dss!(data, data, space::AbstractSpace, comms_ctx)

weighted_dss!(dest, src, space::AbstractSpace) = horizontal_dss!(
    dest,
    Base.Broadcast.broadcasted(⊠, src, space.dss_weights),
    space,
    comms_ctx,
)

function weighted_dss!(dest, src, space::AbstractSpace, comms_ctx)
    weighted_src = Base.Broadcast.broadcasted(⊠, src, space.dss_weights)
    horizontal_dss!(dest, weighted_src, space, comms_ctx)
end

function weighted_dss!(
    dest,
    src,
    space::ExtrudedFiniteDifferenceSpace,
    comms_ctx = nothing,
)
    weighted_src =
        Base.Broadcast.broadcasted(⊠, src, space.horizontal_space.dss_weights)
    horizontal_dss!(dest, weighted_src, space, comms_ctx)
end
weighted_dss!(data, space::AbstractSpace, comms_ctx = nothing) =
    weighted_dss!(data, data, space, comms_ctx)
