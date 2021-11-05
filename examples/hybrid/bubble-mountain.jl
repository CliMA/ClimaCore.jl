push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators
using ClimaCore.Geometry




function warp_agnesi_peak(coord; z_top = 1000.0, a = 1 / 2)

    h = 8 * a^3 / (x_in^2 + 4 * a^2)
    x, z = x_in, z_in + h
    return x, z
end


# set up function space
function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 10,
    velem = 50,
    npoly = 4;
    stretch = Meshes.NoStretching(),
    topography_file = nothing,
)

    # build vertical mesh information with stretching in [0, H]
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = velem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)




    # build horizontal mesh information
    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1])..Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)


    # todo do we seperate hv_center_space & hv_face_space
    # construct hv center/face spaces, recompute metric terms

    hv_face_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_face_space)
end

# set up rhs!
space = hvspace_2D((-500, 500), (0, 1000))


Z = map(Fields.coordinate_field(space)) do coord
    z_top = 1000.0
    a = 1 / 2
    z_ref = coord.z
    z_s = 8 * a^3 / (coord.x^2 + 4 * a^2)
    return z_ref + (1 - z_ref / z_top) * z_s
end

function vertical_warp!(
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    Z::Fields.FaceExtrudedFiniteDifferenceField,
)
    @assert space === axes(Z)
    # take the horizontal gradient of Z
    grad = Operators.Gradient()
    ∇Z = grad.(Z)
    If2c = Operators.InterpolateF2C()
    cZ = If2c.(Z)
    c∇Z = If2c.(∇Z)

    Ni, Nj, Nk, Nv, Nh = size(space.center_local_geometry)
    for h in 1:Nh, j in 1:Nj, i in 1:Ni
        face_column = ClimaCore.column(space.face_local_geometry, i, j, h)
        center_column = ClimaCore.column(space.center_local_geometry, i, j, h)
        Z_column = ClimaCore.column(Fields.field_values(Z), i, j, h)
        ∇Z_column = ClimaCore.column(Fields.field_values(∇Z), i, j, h)
        cZ_column = ClimaCore.column(Fields.field_values(cZ), i, j, h)
        c∇Z_column = ClimaCore.column(Fields.field_values(c∇Z), i, j, h)

        I = (1, 3)

        # update face metrics
        for v in 1:(Nv + 1)
            local_geom = face_column[v]
            coord = Geometry.XZPoint(local_geom.coordinates.x, Z_column[v])
            Δz =
                v == 1 ? 2 * (cZ_column[v] - Z_column[v]) :
                v == Nv + 1 ? 2 * (Z_column[v] - cZ_column[v - 1]) :
                (cZ_column[v] - cZ_column[v - 1])
            ∂x∂ξ = reconstruct_metric(local_geom.∂x∂ξ, ∇Z_column[v], Δz)
            W = local_geom.WJ / local_geom.J
            J = det(Geometry.components(∂x∂ξ))
            face_column[v] = Geometry.LocalGeometry(coord, J, W * J, ∂x∂ξ)
        end

        # update center metrics
        for v in 1:Nv
            local_geom = center_column[v]
            coord = Geometry.XZPoint(local_geom.coordinates.x, cZ_column[v])
            Δz = Z_column[v + 1] - cZ_column[v]
            ∂x∂ξ = reconstruct_metric(local_geom.∂x∂ξ, c∇Z_column[v], Δz)
            W = local_geom.WJ / local_geom.J
            J = det(Geometry.components(∂x∂ξ))
            center_column[v] = Geometry.LocalGeometry(coord, J, W * J, ∂x∂ξ)
        end
    end
    return space
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

vertical_warp!(space, Z)
