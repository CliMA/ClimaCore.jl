#####
##### Hybrid mesh
#####

abstract type HypsographyAdaption end

"""
    Flat()

No surface hypsography.
"""
struct Flat <: HypsographyAdaption end


struct ExtrudedFiniteDifferenceSpace{
    S <: Staggering,
    H <: AbstractSpace,
    T <: Topologies.AbstractIntervalTopology,
    A <: HypsographyAdaption,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
    LGG,
} <: AbstractSpace
    staggering::S
    horizontal_space::H
    vertical_topology::T
    hypsography::A
    global_geometry::GG
    center_local_geometry::LG
    face_local_geometry::LG
    center_ghost_geometry::LGG
    face_ghost_geometry::LGG
end

function issubspace(
    hspace::AbstractSpectralElementSpace,
    extruded_space::ExtrudedFiniteDifferenceSpace,
)
    if hspace === extruded_space.horizontal_space
        return true
    end
    # TODO: improve level handling
    return Spaces.topology(hspace) ===
           Spaces.topology(extruded_space.horizontal_space) &&
           quadrature_style(hspace) ===
           quadrature_style(extruded_space.horizontal_space)
end


Adapt.adapt_structure(to, space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace(
        space.staggering,
        Adapt.adapt(to, space.horizontal_space),
        Adapt.adapt(to, space.vertical_topology),
        Adapt.adapt(to, space.hypsography),
        Adapt.adapt(to, space.global_geometry),
        Adapt.adapt(to, space.center_local_geometry),
        Adapt.adapt(to, space.face_local_geometry),
        Adapt.adapt(to, space.center_ghost_geometry),
        Adapt.adapt(to, space.face_ghost_geometry),
    )


const CenterExtrudedFiniteDifferenceSpace =
    ExtrudedFiniteDifferenceSpace{CellCenter}

const FaceExtrudedFiniteDifferenceSpace =
    ExtrudedFiniteDifferenceSpace{CellFace}

const FaceExtrudedFiniteDifferenceSpace2D =
    ExtrudedFiniteDifferenceSpace{CellFace, <:SpectralElementSpace1D}
const FaceExtrudedFiniteDifferenceSpace3D =
    ExtrudedFiniteDifferenceSpace{CellFace, <:SpectralElementSpace2D}
const CenterExtrudedFiniteDifferenceSpace2D =
    ExtrudedFiniteDifferenceSpace{CellCenter, <:SpectralElementSpace1D}
const CenterExtrudedFiniteDifferenceSpace3D =
    ExtrudedFiniteDifferenceSpace{CellCenter, <:SpectralElementSpace2D}


function ExtrudedFiniteDifferenceSpace{S}(
    space::ExtrudedFiniteDifferenceSpace,
) where {S <: Staggering}
    ExtrudedFiniteDifferenceSpace(
        S(),
        space.horizontal_space,
        space.vertical_topology,
        space.hypsography,
        space.global_geometry,
        space.center_local_geometry,
        space.face_local_geometry,
        space.center_ghost_geometry,
        space.face_ghost_geometry,
    )
end
function Base.show(io::IO, space::ExtrudedFiniteDifferenceSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(
        io,
        space isa CenterExtrudedFiniteDifferenceSpace ?
        "CenterExtrudedFiniteDifferenceSpace" :
        "FaceExtrudedFiniteDifferenceSpace",
        ":",
    )
    println(iio, " "^(indent + 2), space.horizontal_space)
    print(iio, " "^(indent + 2), space.vertical_topology)
end
local_geometry_data(space::CenterExtrudedFiniteDifferenceSpace) =
    space.center_local_geometry

local_geometry_data(space::FaceExtrudedFiniteDifferenceSpace) =
    space.face_local_geometry

# TODO: will need to be defined for distributed
ghost_geometry_data(space::CenterExtrudedFiniteDifferenceSpace) =
    space.center_ghost_geometry
ghost_geometry_data(space::FaceExtrudedFiniteDifferenceSpace) =
    space.face_ghost_geometry
function ExtrudedFiniteDifferenceSpace(
    horizontal_space::H,
    vertical_space::V,
    hypsography::Flat = Flat(),
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

    if horizontal_space isa SpectralElementSpace2D
        center_ghost_geometry =
            product_geometry.(
                horizontal_space.ghost_geometry,
                vertical_space.center_local_geometry,
            )
        face_ghost_geometry =
            product_geometry.(
                horizontal_space.ghost_geometry,
                vertical_space.face_local_geometry,
            )
    else
        center_ghost_geometry = nothing
        face_ghost_geometry = nothing
    end
    return ExtrudedFiniteDifferenceSpace(
        staggering,
        horizontal_space,
        vertical_topology,
        hypsography,
        global_geometry,
        center_local_geometry,
        face_local_geometry,
        center_ghost_geometry,
        face_ghost_geometry,
    )
end

quadrature_style(space::ExtrudedFiniteDifferenceSpace) =
    space.horizontal_space.quadrature_style

topology(space::ExtrudedFiniteDifferenceSpace) = space.horizontal_space.topology
ClimaComms.device(space::ExtrudedFiniteDifferenceSpace) =
    ClimaComms.device(topology(space))
vertical_topology(space::ExtrudedFiniteDifferenceSpace) =
    space.vertical_topology

Base.@propagate_inbounds function slab(
    space::ExtrudedFiniteDifferenceSpace,
    v,
    h,
)
    SpectralElementSpaceSlab(
        space.horizontal_space.quadrature_style,
        slab(local_geometry_data(space), v, h),
    )
end

Base.@propagate_inbounds function column(
    space::ExtrudedFiniteDifferenceSpace,
    i,
    j,
    h,
)
    FiniteDifferenceSpace(
        space.staggering,
        space.vertical_topology,
        Geometry.CartesianGlobalGeometry(),
        column(space.center_local_geometry, i, j, h),
        column(space.face_local_geometry, i, j, h),
    )
end

Base.@propagate_inbounds function column(
    space::ExtrudedFiniteDifferenceSpace,
    i,
    h,
)
    FiniteDifferenceSpace(
        space.staggering,
        space.vertical_topology,
        Geometry.CartesianGlobalGeometry(),
        column(space.center_local_geometry, i, h),
        column(space.face_local_geometry, i, h),
    )
end


Base.@propagate_inbounds function level(
    space::CenterExtrudedFiniteDifferenceSpace,
    v::Integer,
)
    horizontal_space = space.horizontal_space
    if horizontal_space isa SpectralElementSpace1D
        SpectralElementSpace1D(
            horizontal_space.topology,
            horizontal_space.quadrature_style,
            horizontal_space.global_geometry,
            level(space.center_local_geometry, v),
            horizontal_space.dss_weights,
        )
    elseif horizontal_space isa SpectralElementSpace2D
        SpectralElementSpace2D(
            horizontal_space.topology,
            horizontal_space.quadrature_style,
            horizontal_space.global_geometry,
            level(space.center_local_geometry, v),
            level(space.center_ghost_geometry, v),
            horizontal_space.local_dss_weights,
            horizontal_space.ghost_dss_weights,
            horizontal_space.internal_surface_geometry,
            horizontal_space.boundary_surface_geometries,
        )
    else
        error("Unsupported horizontal space")
    end
end
Base.@propagate_inbounds function level(
    space::FaceExtrudedFiniteDifferenceSpace,
    v::PlusHalf,
)
    horizontal_space = space.horizontal_space
    if horizontal_space isa SpectralElementSpace1D
        @inbounds SpectralElementSpace1D(
            horizontal_space.topology,
            horizontal_space.quadrature_style,
            horizontal_space.global_geometry,
            level(space.face_local_geometry, v.i + 1),
            horizontal_space.dss_weights,
        )
    elseif horizontal_space isa SpectralElementSpace2D
        @inbounds SpectralElementSpace2D(
            horizontal_space.topology,
            horizontal_space.quadrature_style,
            horizontal_space.global_geometry,
            level(space.face_local_geometry, v.i + 1),
            level(space.face_ghost_geometry, v.i + 1),
            horizontal_space.local_dss_weights,
            horizontal_space.ghost_dss_weights,
            horizontal_space.internal_surface_geometry,
            horizontal_space.boundary_surface_geometries,
        )
    else
        error("Unsupported horizontal space")
    end
end

nlevels(space::CenterExtrudedFiniteDifferenceSpace) =
    size(space.center_local_geometry, 4)

nlevels(space::FaceExtrudedFiniteDifferenceSpace) =
    size(space.face_local_geometry, 4)

function left_boundary_name(space::ExtrudedFiniteDifferenceSpace)
    boundaries = Topologies.boundaries(space.vertical_topology)
    propertynames(boundaries)[1]
end
function right_boundary_name(space::ExtrudedFiniteDifferenceSpace)
    boundaries = Topologies.boundaries(space.vertical_topology)
    propertynames(boundaries)[2]
end
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
        Tuple{Geometry.VAxis, Geometry.Covariant2Axis},
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
        (Geometry.VWAxis(), Geometry.Covariant23Axis()),
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
            A[2, 1],
            zero(FT),
            A[1, 2],
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
