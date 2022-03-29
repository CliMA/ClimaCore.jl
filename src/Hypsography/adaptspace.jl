"""
    TerrainWarpedIntervalTopology(topology::Topologies.IntervalTopology)

Used to represent an interval topology that has been modified by a
terrain warping from a signed distance to reference surface field.
"""
struct TerrainWarpedIntervalTopology{T <: Topologies.IntervalTopology} <:
       Topologies.AbstractIntervalTopology
    topology::T
end

Topologies.domain(warped_topology::TerrainWarpedIntervalTopology) =
    Topologies.domain(warped_topology.topology)

Topologies.mesh(warped_topology::TerrainWarpedIntervalTopology) =
    Topologies.mesh(warped_topology.topology)

Topologies.boundaries(warped_topology::TerrainWarpedIntervalTopology) =
    Topologies.boundaries(warped_topology.topology)

Topologies.nlocalelems(warped_topology::TerrainWarpedIntervalTopology) =
    Topologies.nlocalelems(warped_topology.topology)


function reconstruct_metric(
    ∂x∂ξ::Geometry.Axis2Tensor{
        T,
        Tuple{Geometry.UWAxis, Geometry.Covariant13Axis},
    },
    ∇z::Geometry.Covariant1Vector,
    Δz::Real,
) where {T}
    v∂x∂ξ = Geometry.components(∂x∂ξ)
    v∇z = Geometry.components(∇z)
    Geometry.AxisTensor(axes(∂x∂ξ), @SMatrix [
        v∂x∂ξ[1, 1] 0
        v∇z[1] Δz
    ])
end

function reconstruct_metric(
    ∂x∂ξ::Geometry.Axis2Tensor{
        T,
        Tuple{Geometry.UVWAxis, Geometry.Covariant123Axis},
    },
    ∇z::Geometry.Covariant12Vector,
    Δz::Real,
) where {T}
    v∂x∂ξ = Geometry.components(∂x∂ξ)
    v∇z = Geometry.components(∇z)
    Geometry.AxisTensor(
        axes(∂x∂ξ),
        @SMatrix [
            v∂x∂ξ[1, 1] v∂x∂ξ[1, 2] 0
            v∂x∂ξ[2, 1] v∂x∂ξ[2, 2] 0
            v∇z[1] v∇z[2] Δz
        ]
    )
end

function adapt_space!(
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    fZ::Fields.FaceExtrudedFiniteDifferenceField,
)
    @assert Spaces.FaceExtrudedFiniteDifferenceSpace(space) === axes(fZ)

    # Take the horizontal gradient for the Z surface field
    # for computing updated ∂x∂ξ₃₁, ∂x∂ξ₃₂ terms
    grad = Operators.Gradient()
    If2c = Operators.InterpolateF2C()

    # DSS the horizontal gradient of Z surface field to force
    # deriv continuity along horizontal element boundaries
    f∇Z = grad.(fZ)
    Spaces.weighted_dss!(f∇Z)

    # Interpolate horizontal gradient surface field to centers
    # used to compute ∂x∂ξ₃₃ (Δz) metric term
    cZ = If2c.(fZ)

    # DSS the interpolated horizontal gradients as well
    c∇Z = If2c.(f∇Z)
    Spaces.weighted_dss!(c∇Z)

    Ni, Nj, _, Nv, Nh = size(space.center_local_geometry)
    for h in 1:Nh, j in 1:Nj, i in 1:Ni
        face_column = column(space.face_local_geometry, i, j, h)
        fZ_column = column(Fields.field_values(fZ), i, j, h)
        f∇Z_column = column(Fields.field_values(f∇Z), i, j, h)

        center_column = column(space.center_local_geometry, i, j, h)
        cZ_column = column(Fields.field_values(cZ), i, j, h)
        c∇Z_column = column(Fields.field_values(c∇Z), i, j, h)

        # update face metrics
        for v in 1:(Nv + 1)
            local_geom = face_column[v]
            coord =
                if typeof(space) <: Spaces.FaceExtrudedFiniteDifferenceSpace2D
                    c1 = Geometry.components(local_geom.coordinates)[1]
                    typeof(local_geom.coordinates)(c1, fZ_column[v])
                else
                    c1 = Geometry.components(local_geom.coordinates)[1]
                    c2 = Geometry.components(local_geom.coordinates)[2]
                    typeof(local_geom.coordinates)(c1, c2, fZ_column[v])
                end
            Δz = if v == 1
                # if this is the domain min face level compute the metric
                # extrapolating from the bottom face level of the domain
                2 * (cZ_column[v] - fZ_column[v])
            elseif v == Nv + 1
                # if this is the domain max face level compute the metric
                # extrapolating from the top face level of the domain
                2 * fZ_column[v] - cZ_column[v - 1]
            else
                cZ_column[v] - cZ_column[v - 1]
            end
            ∂x∂ξ = reconstruct_metric(local_geom.∂x∂ξ, f∇Z_column[v], Δz)
            W = local_geom.WJ / local_geom.J
            J = det(Geometry.components(∂x∂ξ))
            face_column[v] = Geometry.LocalGeometry(coord, J, W * J, ∂x∂ξ)
        end

        # update center metrics
        for v in 1:Nv
            local_geom = center_column[v]
            coord =
                if typeof(space) <: Spaces.FaceExtrudedFiniteDifferenceSpace2D
                    c1 = Geometry.components(local_geom.coordinates)[1]
                    typeof(local_geom.coordinates)(c1, cZ_column[v])
                else
                    c1 = Geometry.components(local_geom.coordinates)[1]
                    c2 = Geometry.components(local_geom.coordinates)[2]
                    typeof(local_geom.coordinates)(c1, c2, cZ_column[v])
                end
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
        space.center_ghost_geometry,
        space.face_ghost_geometry,
    )
end

"""
    adapt_space!(space, adaption, reference_distance)

Inplace updates local_geometry coordinate and metric terms given a surface elevation
`reference_distance` Field (ex. mean sea-level). Vertical level coordinates are adapted
to the surface elevation warped space given a `TerrainAdaption` method.

Returns the adapted `ExtrudedFiniteDifferenceSpace`.
"""
function adapt_space!(
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    adaption::TerrainAdaption,
    reference_distance::Fields.Field,
)
    # construct Z levels at all faces
    if axes(reference_distance) !== space.horizontal_space
        error(
            "`reference_distance` space is not the same instance as the `ExtrudedFiniteDifferenceSpace` horizontal space",
        )
    end
    coords =
        Fields.coordinate_field(Spaces.FaceExtrudedFiniteDifferenceSpace(space))
    vertical_ref =
        if typeof(space) <: Spaces.FaceExtrudedFiniteDifferenceSpace2D
            Geometry.component.(coords, 2)
        else
            Geometry.component.(coords, 3)
        end
    # TODO Generalise 178 so that it works always with the last coordinate
    vertical_domain = Topologies.domain(space.vertical_topology)
    domain_max = Geometry.component(vertical_domain.coord_max, 1)
    f∇Z = Fields.Field(
        adapt.(
            Ref(adaption),
            Fields.field_values(vertical_ref),
            Fields.field_values(reference_distance),
            domain_max,
        ),
        axes(coords),
    )
    return adapt_space!(space, f∇Z)
end

function Spaces.ExtrudedFiniteDifferenceSpace(
    horizontal_space::Spaces.AbstractSpace,
    vertical_space::Spaces.FiniteDifferenceSpace,
    adaption::TerrainAdaption,
    reference_distance::Fields.Field,
)
    space =
        Spaces.ExtrudedFiniteDifferenceSpace(horizontal_space, vertical_space)
    return adapt_space!(space, adaption, reference_distance)
end
