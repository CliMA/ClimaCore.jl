import ..Topologies
using ..RecursiveApply


function dss_transform(arg, local_geometry)
    RecursiveApply.rmap(arg) do x
        dss_transform(x, local_geometry)
    end
end
dss_transform(arg::Number, local_geometry) = arg
dss_transform(
    arg::Geometry.AxisTensor{T, N, <:Tuple{Vararg{Geometry.CartesianAxis}}},
    local_geometry::Geometry.LocalGeometry,
) where {T, N} = arg
dss_transform(
    arg::Geometry.CartesianVector,
    local_geometry::Geometry.LocalGeometry,
) where {T, N} = arg
dss_transform(
    arg::Geometry.AxisTensor{T, N, <:Tuple{Vararg{Geometry.LocalAxis}}},
    local_geometry::Geometry.LocalGeometry,
) where {T, N} = arg
dss_transform(
    arg::Geometry.LocalVector,
    local_geometry::Geometry.LocalGeometry,
) where {T, N} = arg

function dss_transform(
    arg::Geometry.AxisVector,
    local_geometry::Geometry.LocalGeometry,
)
    ax = axes(local_geometry.∂x∂ξ, 1)
    axfrom = axes(arg, 1)
    if ax isa Geometry.Cartesian12Axis && (
        axfrom isa Geometry.Covariant3Axis ||
        axfrom isa Geometry.Contravariant3Axis
    )
        return arg
    end
    Geometry.transform(ax, arg, local_geometry)
end

function dss_untransform(refarg, targ, local_geometry)
    RecursiveApply.rmap(refarg, targ) do rx, tx
        dss_untransform(rx, tx, local_geometry)
    end
end
dss_untransform(refarg::Number, targ, local_geometry) = targ
dss_untransform(
    refarg::T,
    targ::T,
    local_geometry::Geometry.LocalGeometry,
) where {T <: Geometry.AxisTensor} = targ
function dss_untransform(
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
    Nq::Integer,
    Nv::Integer = 1,
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


"""
    horizontal_dss!(dest, src, topology, Nq)

Apply horizontal direct stiffness summation (DSS) to `src`, storing the result in `dest`.
"""
function dss_2d!(
    dest,
    src,
    local_geometry_data,
    topology::Topologies.AbstractTopology,
    Nq::Integer,
    Nv::Integer = 1,
)
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
        dss_1d!(dest, src, local_geometry_data(space), htopology, Nq, Nv)
    elseif hspace isa SpectralElementSpace2D
        dss_2d!(dest, src, local_geometry_data(space), htopology, Nq, Nv)
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
