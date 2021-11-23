#####
##### Hybrid mesh
#####

struct ExtrudedFiniteDifferenceSpace{
    S <: Staggering,
    H <: AbstractSpace,
    T <: Topologies.IntervalTopology,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
} <: AbstractSpace
    staggering::S
    horizontal_space::H
    vertical_topology::T
    global_geometry::GG
    center_local_geometry::LG
    face_local_geometry::LG
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
        space.global_geometry,
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
    global_geometry = horizontal_space.global_geometry
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
        global_geometry,
        center_local_geometry,
        face_local_geometry,
    )
    return nothing
end

quadrature_style(space::ExtrudedFiniteDifferenceSpace) =
    space.horizontal_space.quadrature_style

topology(space::ExtrudedFiniteDifferenceSpace) = space.horizontal_space.topology

function slab(space::ExtrudedFiniteDifferenceSpace, v, h)
    SpectralElementSpaceSlab(
        space.horizontal_space.quadrature_style,
        slab(local_geometry_data(space), v, h),
    )
end

column(space::ExtrudedFiniteDifferenceSpace, i, j, h) = FiniteDifferenceSpace(
    space.staggering,
    space.vertical_topology,
    Geometry.CartesianGlobalGeometry(),
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
        Tuple{Geometry.UAxis, Geometry.Covariant1Axis},
        SMatrix{1, 1, FT, 1},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    Geometry.AxisTensor(
        (Geometry.UWAxis(), Geometry.Covariant13Axis()),
        SMatrix{2, 2}(A[1, 1], zero(FT), zero(FT), B[1, 1]),
    )
end

function blockmat(
    a::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.UVAxis, Geometry.Covariant12Axis},
        SMatrix{2, 2, FT, 4},
    },
    b::Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.WAxis, Geometry.Covariant3Axis},
        SMatrix{1, 1, FT, 1},
    },
) where {FT}
    A = Geometry.components(a)
    B = Geometry.components(b)
    Geometry.AxisTensor(
        (Geometry.UVWAxis(), Geometry.Covariant123Axis()),
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
    return Geometry.LocalGeometry(coordinates, J, WJ, ∂x∂ξ)
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
