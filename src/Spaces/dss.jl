import ..Topologies
using ..RecursiveApply

function dss_1d!(
    dest,
    src,
    htopology::Topologies.AbstractTopology,
    Nq::Integer,
    Nv::Integer = 1,
)
    idx1 = CartesianIndex(1, 1, 1, 1, 1)
    idx2 = CartesianIndex(Nq, 1, 1, 1, 1)
    for (elem1, face1, elem2, face2, reversed) in
        Topologies.interior_faces(htopology)
        for level in 1:Nv
            @assert face1 == 1 && face2 == 2 && !reversed
            src_slab1 = slab(src, level, elem1)
            src_slab2 = slab(src, level, elem2)
            val = src_slab1[idx1] ⊞ src_slab2[idx2]
            dest_slab1 = slab(dest, level, elem1)
            dest_slab2 = slab(dest, level, elem2)
            dest_slab1[idx1] = dest_slab2[idx2] = val
        end
    end
    return dest
end

"""
    horizontal_dss!(dest, src, topology, Nq)

Apply horizontal direct stiffness summation (DSS) to `src`, storing the result in `dest`.
"""
function dss_2d!(
    dest,
    src,
    topology::Topologies.AbstractTopology,
    Nq::Integer,
    Nv::Integer = 1,
)
    @assert Nv == 1 # need to make this work more generally

    # TODO: generalize to extruded domains by returning a cartesian index?
    # iterate over the interior faces for each element of the mesh
    for (elem1, face1, elem2, face2, reversed) in
        Topologies.interior_faces(topology)
        # iterate over non-vertex nodes
        src_slab1 = slab(src, elem1)
        src_slab2 = slab(src, elem2)
        dest_slab1 = slab(dest, elem1)
        dest_slab2 = slab(dest, elem2)
        for q in 2:(Nq - 1)
            i1, j1 = Topologies.face_node_index(face1, Nq, q, false)
            i2, j2 = Topologies.face_node_index(face2, Nq, q, reversed)
            value = src_slab1[i1, j1] ⊞ src_slab2[i2, j2]
            dest_slab1[i1, j1] = dest_slab2[i2, j2] = value
        end
    end

    # iterate over all vertices
    for vertex in Topologies.vertices(topology)
        # gather: compute sum over shared vertices
        sum_data = mapreduce(⊞, vertex) do (elem, vertex_num)
            src_slab = slab(src, elem)
            i, j = Topologies.vertex_node_index(vertex_num, Nq)
            src_slab[i, j]
        end

        # scatter: assign sum to shared vertices
        for (elem, vertex_num) in vertex
            dest_slab = slab(dest, elem)
            i, j = Topologies.vertex_node_index(vertex_num, Nq)
            dest_slab[i, j] = sum_data
        end
    end
    return dest
end

function horizontal_dss!(dest, src, space::AbstractSpace)
    if space isa ExtrudedFiniteDifferenceSpace
        Nv = nlevels(space)
        hspace = space.horizontal_space
    else
        Nv = 1
        hspace = space
    end
    htopology = hspace.topology
    Nq = Quadratures.degrees_of_freedom(hspace.quadrature_style)
    if hspace isa SpectralElementSpace1D
        dss_1d!(dest, src, htopology, Nq, Nv)
    elseif hspace isa SpectralElementSpace2D
        dss_2d!(dest, src, htopology, Nq, Nv)
    end
end

horizontal_dss!(data, space::AbstractSpace) =
    horizontal_dss!(data, data, space::AbstractSpace)

weighted_dss!(dest, src, space::AbstractSpace) = horizontal_dss!(
    dest,
    Base.Broadcast.broadcasted(⊠, src, space.dss_weights),
    space,
)

weighted_dss!(dest, src, space::ExtrudedFiniteDifferenceSpace) =
    horizontal_dss!(
        dest,
        Base.Broadcast.broadcasted(⊠, src, space.horizontal_space.dss_weights),
        space,
    )

weighted_dss!(data, space::AbstractSpace) = weighted_dss!(data, data, space)
