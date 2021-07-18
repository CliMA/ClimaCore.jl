import ..Topologies
using ..RecursiveApply

"""
    horizontal_dss!(dest, src, topology, Nq)

Apply horizontal direct stiffness summation (DSS) to `src`, storing the result in `dest`.
"""
function horizontal_dss!(
    dest,
    src,
    topology::Topologies.AbstractTopology,
    Nq::Integer,
)

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
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)
    horizontal_dss!(dest, src, space.topology, Nq)
end

horizontal_dss!(data, space::AbstractSpace) =
    horizontal_dss!(data, data, space::AbstractSpace)

weighted_dss!(dest, src, space::AbstractSpace) = horizontal_dss!(
    dest,
    Base.Broadcast.broadcasted(⊠, src, space.dss_weights),
    space,
)

weighted_dss!(data, space::AbstractSpace) = weighted_dss!(data, data, space)
