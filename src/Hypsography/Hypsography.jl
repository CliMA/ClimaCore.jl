module Hypsography

import ..slab, ..column
import ..Geometry, ..Domains, ..Topologies, ..Spaces, ..Fields, ..Operators
import ..Spaces: ExtrudedFiniteDifferenceSpace, HypsographyAdaption, Flat

using StaticArrays, LinearAlgebra

"""
    LinearAdaption(surface::Field)

Locate the levels by linear interpolation between the surface field and the top
of the domain, using the method of [GalChen1975](@cite).
"""
struct LinearAdaption{F <: Union{Fields.Field, Nothing}} <: HypsographyAdaption
    # Union can be removed once deprecation removed.
    surface::F
end

# deprecated in 0.10.12
LinearAdaption() = LinearAdaption(nothing)

@deprecate(
    ExtrudedFiniteDifferenceSpace(
        horizontal_space::Spaces.AbstractSpace,
        vertical_space::Spaces.FiniteDifferenceSpace,
        adaption::LinearAdaption{Nothing},
        z_surface::Fields.Field,
    ),
    ExtrudedFiniteDifferenceSpace(
        horizontal_space,
        vertical_space,
        LinearAdaption(z_surface),
    ),
    false
)

# linear coordinates
function ExtrudedFiniteDifferenceSpace(
    horizontal_space::Spaces.AbstractSpace,
    vertical_space::Spaces.FiniteDifferenceSpace,
    adaption::HypsographyAdaption,
)
    if adaption isa LinearAdaption
        if isnothing(adaption.surface)
            error("LinearAdaption requires a Field argument")
        end
        if axes(adaption.surface) !== horizontal_space
            error("Terrain must be defined on the horizontal space")
        end
    end

    # construct initial flat space, then mutate
    space = Spaces.ExtrudedFiniteDifferenceSpace(
        horizontal_space,
        vertical_space,
        Flat(),
    )
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(space)
    coord_type = eltype(Spaces.coordinates_data(face_space))
    z_ref = Spaces.coordinates_data(face_space).z

    vertical_domain = Topologies.domain(space.vertical_topology)
    z_top = vertical_domain.coord_max.z
    
    grad = Operators.Gradient()
    wdiv = Operators.WeakDivergence()

    if adaption isa LinearAdaption
        z_surface = adaption.surface
        FT = eltype(z_surface)
        @show extrema(z_surface)
        κ_smooth = eltype(z_surface)(1e8)
        dt = eltype(z_surface)(1e-1)
        # ∂ϕ∂t = ∂²ϕ/∂x² |> Operators are the same as those described in the spectral mesh. 
        for iter = 1:20000
           χzₛ = wdiv.(grad.(z_surface))
           Spaces.weighted_dss!(χzₛ)
           z_surface .+= κ_smooth .* dt .* χzₛ
           Spaces.weighted_dss!(z_surface)
           @. z_surface = ifelse(z_surface < FT(0), FT(0), z_surface)
        end
        Spaces.weighted_dss!(z_surface)
        z_surface = Fields.field_values(z_surface)
        @show extrema(z_surface)
        fZ_data = @. z_ref + (1 - z_ref / z_top) * z_surface
        fZ = Fields.Field(fZ_data, face_space)
    end

    # Take the horizontal gradient for the Z surface field
    # for computing updated ∂x∂ξ₃₁, ∂x∂ξ₃₂ terms
    If2c = Operators.InterpolateF2C()

    # DSS the horizontal gradient of Z surface field to force
    # deriv continuity along horizontal element boundaries
    f∇Z = grad.(fZ)
    #Spaces.weighted_dss!(f∇Z)

    # Interpolate horizontal gradient surface field to centers
    # used to compute ∂x∂ξ₃₃ (Δz) metric term
    cZ = If2c.(fZ)

    # DSS the interpolated horizontal gradients as well
    c∇Z = If2c.(f∇Z)
    #Spaces.weighted_dss!(c∇Z)

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
            coord = if coord_type <: Geometry.Abstract2DPoint
                c1 = Geometry.components(local_geom.coordinates)[1]
                coord_type(c1, fZ_column[v])
            elseif coord_type <: Geometry.Abstract3DPoint
                c1 = Geometry.components(local_geom.coordinates)[1]
                c2 = Geometry.components(local_geom.coordinates)[2]
                coord_type(c1, c2, fZ_column[v])
            end
            Δz = if v == 1
                # if this is the domain min face level compute the metric
                # extrapolating from the bottom face level of the domain
                2 * (cZ_column[v] - fZ_column[v])
            elseif v == Nv + 1
                # if this is the domain max face level compute the metric
                # extrapolating from the top face level of the domain
                2 * (fZ_column[v] - cZ_column[v - 1])
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
            coord = if coord_type <: Geometry.Abstract2DPoint
                c1 = Geometry.components(local_geom.coordinates)[1]
                coord_type(c1, cZ_column[v])
            elseif coord_type <: Geometry.Abstract3DPoint
                c1 = Geometry.components(local_geom.coordinates)[1]
                c2 = Geometry.components(local_geom.coordinates)[2]
                coord_type(c1, c2, cZ_column[v])
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
        space.vertical_topology,
        adaption,
        space.global_geometry,
        space.center_local_geometry,
        space.face_local_geometry,
        space.center_ghost_geometry,
        space.face_ghost_geometry,
    )
end



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

end
