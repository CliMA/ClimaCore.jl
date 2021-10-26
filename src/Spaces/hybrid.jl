#####
##### Hybrid mesh
#####

struct ExtrudedFiniteDifferenceSpace{
    S <: Staggering,
    H <: AbstractSpace,
    T <: Topologies.IntervalTopology,
    G,
} <: AbstractSpace
    staggering::S
    horizontal_space::H
    vertical_topology::T
    center_local_geometry::G
    face_local_geometry::G
end

const CenterExtrudedFiniteDifferenceSpace =
    ExtrudedFiniteDifferenceSpace{CellCenter}

const FaceExtrudedFiniteDifferenceSpace =
    ExtrudedFiniteDifferenceSpace{CellFace}

function ExtrudedFiniteDifferenceSpace{S}(
    space::ExtrudedFiniteDifferenceSpace,
) where {S <: Staggering}
    ExtrudedFiniteDifferenceSpace(
        S(),
        space.horizontal_space,
        space.vertical_topology,
        space.center_local_geometry,
        space.face_local_geometry,
    )
end

local_geometry_data(space::CenterExtrudedFiniteDifferenceSpace) =
    space.center_local_geometry

local_geometry_data(space::FaceExtrudedFiniteDifferenceSpace) =
    space.face_local_geometry

function ExtrudedFiniteDifferenceSpace(
    horizontal_space::H,
    vertical_space::V,
) where {H <: AbstractSpace, V <: FiniteDifferenceSpace}
    staggering = vertical_space.staggering
    vertical_topology = vertical_space.topology
    center_local_geometry =
        product_geometry.(
            horizontal_space.local_geometry,
            vertical_space.center_local_geometry,
        )
    face_local_geometry =
        product_geometry.(
            horizontal_space.local_geometry,
            vertical_space.face_local_geometry,
        )
    return ExtrudedFiniteDifferenceSpace(
        staggering,
        horizontal_space,
        vertical_topology,
        center_local_geometry,
        face_local_geometry,
    )
    return nothing
end



function ExtrudedFiniteDifferenceSpace(
    horizontal_space::H,
    vertical_mesh::V,
    # stretching function is (0,1)->（0,1)
    stretching_function,
    topography,
) where {H <: AbstractSpace, V}
    #todo how to get FT
    FT = Float64
    nhelems = Topologies.nlocalelems(horizontal_space.topology)
    nvelems = length(vertical_mesh.faces) - 1
    nvlevels = 2nvelems + 1

    quadrature_style = horizontal_space.quadrature_style
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)
    Ht = Geometry.component(vertical_mesh.domain.coord_max, 1)


    topo_coord_x1, topo_coord_x3 = zeros(Nq), zeros(Nq)

    # loop each column
    for helem in 1:nhelems
        # odd layer indicates cell faces, even layer indeicates cell centers

        # topo_coord_x1 .= reshape(
        #     parent(slab(horizontal_space.local_geometry.coordinates, helem)),
        #     Nq,
        # )
        # topo_coord_x3 .= Hb .+ topography[helem]

        topo_coord_x1 .= topography[helem][:, 1]
        topo_coord_x3 .= topography[helem][:, 2]
        for i in 1:Nq
            for vlevel in 1:nvlevels


                ξ1 = quad_points[i]
                ξ3 = (vlevel - 1) / 2.0

                # the map is 
                ξ_x_map =
                    (ξ1, ξ3) -> begin
                        xb1 = Geometry.high_order_interpolate(topo_coord_x1, ξ1)
                        xb3 = Geometry.high_order_interpolate(topo_coord_x3, ξ1)

                        x1 = xb1
                        x3 =
                            xb3 + (Ht - xb3) * stretching_function(ξ3 / nvelems)
                        return x1, x3
                    end
                ξ = SVector(ξ1, ξ3)

                ∂x∂ξ = ForwardDiff.jacobian(ξ) do ξ
                    local x
                    x = ξ_x_map(ξ[1], ξ[2])
                    SVector(Geometry.component(x, 1), Geometry.component(x, 2))
                end

                J = det(∂x∂ξ)
                ∂ξ∂x = inv(∂x∂ξ)
                WJ = J * quad_weights[i]


                # compute metric terms at the cell center
                if vlevel % 2 == 0
                    center_local_geometry = 0

                    # compute metric terms at the cell face
                else

                    face_local_geometry = 0

                end
            end


        end

        # compute metric terms at the top (cell top face)

    end




    return nothing
end


quadrature_style(space::ExtrudedFiniteDifferenceSpace) =
    space.horizontal_space.quadrature_style

topology(space::ExtrudedFiniteDifferenceSpace) = space.horizontal_space.topology

slab(space::ExtrudedFiniteDifferenceSpace, v, h) =
    slab(space.horizontal_space, v, h)

column(space::ExtrudedFiniteDifferenceSpace, i, j, h) = FiniteDifferenceSpace(
    space.staggering,
    space.vertical_topology,
    column(space.center_local_geometry, i, j, h),
    column(space.face_local_geometry, i, j, h),
)

nlevels(space::CenterExtrudedFiniteDifferenceSpace) =
    size(space.center_local_geometry, 4)

nlevels(space::FaceExtrudedFiniteDifferenceSpace) =
    size(space.face_local_geometry, 4)

left_boundary_name(space::ExtrudedFiniteDifferenceSpace) =
    propertynames(space.vertical_topology.boundaries)[1]

right_boundary_name(space::ExtrudedFiniteDifferenceSpace) =
    propertynames(space.vertical_topology.boundaries)[2]

function blockmat(
    a::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Cartesian1Axis, Geometry.Covariant1Axis},
        SMatrix{1, 1, FT, 1},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Cartesian3Axis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    Geometry.AxisTensor(
        (Geometry.Cartesian13Axis(), Geometry.Covariant13Axis()),
        SMatrix{2, 2}(A[1, 1], zero(FT), zero(FT), B[1, 1]),
    )
end

function blockmat(
    a::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Cartesian12Axis, Geometry.Covariant12Axis},
        SMatrix{2, 2, FT, 4},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Cartesian3Axis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    Geometry.AxisTensor(
        (Geometry.Cartesian123Axis(), Geometry.Covariant123Axis()),
        SMatrix{3, 3}(
            A[1, 1],
            A[1, 2],
            zero(FT),
            A[2, 1],
            A[2, 2],
            zero(FT),
            zero(FT),
            zero(FT),
            B[1, 1],
        ),
    )
end

function product_geometry(
    horizontal_local_geometry::Geometry.LocalGeometry,
    vertical_local_geometry::Geometry.LocalGeometry,
)
    coordinates = Geometry.product_coordinates(
        horizontal_local_geometry.coordinates,
        vertical_local_geometry.coordinates,
    )
    J = horizontal_local_geometry.J * vertical_local_geometry.J
    WJ = horizontal_local_geometry.WJ * vertical_local_geometry.WJ
    ∂x∂ξ =
        blockmat(horizontal_local_geometry.∂x∂ξ, vertical_local_geometry.∂x∂ξ)
    ∂ξ∂x = inv(∂x∂ξ)
    return Geometry.LocalGeometry(coordinates, J, WJ, ∂x∂ξ, ∂ξ∂x)
end

function eachslabindex(cspace::CenterExtrudedFiniteDifferenceSpace)
    h_iter = eachslabindex(cspace.horizontal_space)
    Nv = size(cspace.center_local_geometry, 4)
    return Iterators.product(1:Nv, h_iter)
end
function eachslabindex(fspace::FaceExtrudedFiniteDifferenceSpace)
    h_iter = eachslabindex(fspace.horizontal_space)
    Nv = size(fspace.face_local_geometry, 4)
    return Iterators.product(1:Nv, h_iter)
end
