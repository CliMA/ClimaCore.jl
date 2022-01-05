"""
    TerrainWarpedIntervalTopology(topology::Topologies.IntervalTopology)

This is used to represent an interval topology that has been modified by terrain warping.
"""
struct TerrainWarpedIntervalTopology{T <: Topologies.IntervalTopology} <:
       Topologies.AbstractIntervalTopology
    topology::T
end

Topologies.domain(warped_topology::TerrainWarpedIntervalTopology) =
    Topologies.domain(warped_topology.topology)

Topologies.boundaries(warped_topology::TerrainWarpedIntervalTopology) =
    Topologies.boundaries(warped_topology.topology)

Topologies.nlocalelems(warped_topology::TerrainWarpedIntervalTopology) =
    Topologies.nlocalelems(warped_topology.topology)

function adapt_space!(
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    fZ::Fields.FaceExtrudedFiniteDifferenceField,
)
    @assert Spaces.FaceExtrudedFiniteDifferenceSpace(space) === axes(fZ)
    # take the horizontal gradient of Z
    grad = Operators.Gradient()
    f∇Z = grad.(fZ)
    If2c = Operators.InterpolateF2C()
    cZ = If2c.(fZ)
    c∇Z = If2c.(f∇Z)

    Ni, Nj, Nk, Nv, Nh = size(space.center_local_geometry)
    for h in 1:Nh, j in 1:Nj, i in 1:Ni
        face_column = column(space.face_local_geometry, i, j, h)
        center_column = column(space.center_local_geometry, i, j, h)
        fZ_column = column(Fields.field_values(fZ), i, j, h)
        f∇Z_column = column(Fields.field_values(f∇Z), i, j, h)
        cZ_column = column(Fields.field_values(cZ), i, j, h)
        c∇Z_column = column(Fields.field_values(c∇Z), i, j, h)

        I = (1, 3)

        # update face metrics
        for v in 1:(Nv + 1)
            local_geom = face_column[v]
            coord = Geometry.XZPoint(local_geom.coordinates.x, fZ_column[v])
            Δz =
                v == 1 ? 2 * (cZ_column[v] - fZ_column[v]) :
                v == Nv + 1 ? 2 * (fZ_column[v] - cZ_column[v - 1]) :
                (cZ_column[v] - cZ_column[v - 1])
            ∂x∂ξ = reconstruct_metric(local_geom.∂x∂ξ, f∇Z_column[v], Δz)
            W = local_geom.WJ / local_geom.J
            J = det(Geometry.components(∂x∂ξ))
            face_column[v] = Geometry.LocalGeometry(coord, J, W * J, ∂x∂ξ)
        end

        # update center metrics
        for v in 1:Nv
            local_geom = center_column[v]
            coord = Geometry.XZPoint(local_geom.coordinates.x, cZ_column[v])
            Δz = fZ_column[v + 1] - fZ_column[v]
            ∂x∂ξ = reconstruct_metric(local_geom.∂x∂ξ, c∇Z_column[v], Δz)
            W = local_geom.WJ / local_geom.J
            J = det(Geometry.components(∂x∂ξ))
            center_column[v] = Geometry.LocalGeometry(coord, J, W * J, ∂x∂ξ)
        end
    end
    return Spaces.ExtrudedFiniteDifferenceSpace(
        space.staggering,
        space.horizontal_space,
        TerrainWarpedIntervalTopology(space.vertical_topology),
        space.global_geometry,
        space.center_local_geometry,
        space.face_local_geometry,
    )
    return space
end


function adapt_space!(
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    z_surface::Fields.Field,
    adaption::TerrainAdaption,
)
    @assert space.horizontal_space === axes(z_surface)
    z_top = space.vertical_topology.mesh.domain.coord_max
    # construct Z levels at all faces
    coords =
        Fields.coordinate_field(Spaces.FaceExtrudedFiniteDifferenceSpace(space))
    fZ = Fields.Field(
        adapt.(
            Fields.field_values(coords.z),
            Fields.field_values(z_surface),
            z_top.z,
            Ref(adaption),
        ),
        axes(coords),
    )
    adapt_space!(space, fZ)
end

function Spaces.ExtrudedFiniteDifferenceSpace(
    horizontal_space::Spaces.AbstractSpace,
    vertical_space::Spaces.FiniteDifferenceSpace,
    z_surface::Fields.Field,
    adaption::TerrainAdaption,
)

    space =
        Spaces.ExtrudedFiniteDifferenceSpace(horizontal_space, vertical_space)
    return adapt_space!(space, z_surface, adaption)
end


function reconstruct_metric(
    ∂x∂ξ::Geometry.Axis2Tensor{
        T,
        Tuple{Geometry.UWAxis, Geometry.Covariant13Axis},
    },
    ∇z::Geometry.Covariant1Vector,
    Δz::Real,
) where {T}
    Geometry.AxisTensor(
        axes(∂x∂ξ),
        @SMatrix [
            Geometry.components(∂x∂ξ)[1, 1] 0
            Geometry.components(∇z)[1] Δz
        ]
    )
end
function reconstruct_metric(
    ∂x∂ξ::Geometry.Axis2Tensor{
        T,
        Tuple{Geometry.UVWAxis, Geometry.Covariant123Axis},
    },
    ∇z::Geometry.Covariant12Vector,
    Δz::Real,
) where {T}
    Geometry.AxisTensor(
        axes(∂x∂ξ),
        @SMatrix [
            Geometry.components(∂x∂ξ)[1, 1] Geometry.components(∂x∂ξ)[1, 2] 0
            Geometry.components(∂x∂ξ)[2, 1] Geometry.components(∂x∂ξ)[2, 2] 0
            Geometry.components(∇z)[1] Geometry.components(∇z)[2] Δz
        ]
    )
end
