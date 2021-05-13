import ..Topologies
using ..RecursiveOperators

"""
    horizontal_dss!(data, topology, Nq)

Apply direct stiffness summation (DSS) to `data` horizontally.
"""
function horizontal_dss!(
    data,
    topology::Topologies.AbstractTopology,
    Nq::Integer,
)

    # iterate over the interior faces for each element of the mesh
    for (elem1, face1, elem2, face2, reversed) in
        Topologies.interior_faces(topology)
        # iterate over non-vertex nodes
        slab1 = slab(data, elem1)
        slab2 = slab(data, elem2)
        for q in 2:(Nq - 1)
            i1, j1 = Topologies.face_node_index(face1, Nq, q, false)
            i2, j2 = Topologies.face_node_index(face2, Nq, q, reversed)
            value = slab1[i1, j1] ⊞ slab2[i2, j2]
            slab1[i1, j1] = slab2[i2, j2] = value
        end
    end

    # iterate over all vertices
    for vertex in Topologies.vertices(topology)
        # gather: compute sum over shared vertices
        sum_data = mapreduce(⊞, vertex) do (elem, vertex_num)
            data_slab = slab(data, elem)
            i, j = Topologies.vertex_node_index(vertex_num, Nq)
            data_slab[i, j]
        end

        # scatter: assign sum to shared vertices
        for (elem, vertex_num) in vertex
            data_slab = slab(data, elem)
            i, j = Topologies.vertex_node_index(vertex_num, Nq)
            data_slab[i, j] = sum_data
        end
    end
    return data
end

function horizontal_dss!(data, mesh::AbstractMesh)
    Nq = Quadratures.degrees_of_freedom(mesh.quadrature_style)
    horizontal_dss!(data, mesh.topology, Nq)
end
