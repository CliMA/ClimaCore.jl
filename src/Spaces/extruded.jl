#####
##### Hybrid mesh
#####

abstract type HypsographyAdaption end

"""
    Flat()

No surface hypsography.
"""
struct Flat <: HypsographyAdaption end


mutable struct CenterExtrudedFiniteDifferenceSpace{
    H <: AbstractSpace,
    T <: Topologies.AbstractIntervalTopology,
    A <: HypsographyAdaption,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
    LGG,
} <: AbstractSpace
    horizontal_space::H
    vertical_topology::T
    hypsography::A
    global_geometry::GG
    center_local_geometry::LG
    face_local_geometry::LG
    center_ghost_geometry::LGG
    face_ghost_geometry::LGG
end

face_space(space::CenterExtrudedFiniteDifferenceSpace) = 
    FaceSpace(space)
center_space(space::CenterExtrudedFiniteDifferenceSpace) = 
    space


const FaceExtrudedFiniteDifferenceSpace =
    FaceSpace{<:CenterExtrudedFiniteDifferenceSpace}

const ExtrudedFiniteDifferenceSpace = Union{
    CenterExtrudedFiniteDifferenceSpace,
    FaceExtrudedFiniteDifferenceSpace,
}

function issubspace(
    hspace::AbstractSpectralElementSpace,
    extruded_space::ExtrudedFiniteDifferenceSpace,
)
    if hspace === extruded_Spaces.horizontal_space(space)
        return true
    end
    # TODO: improve level handling
    return Spaces.topology(hspace) ===
           Spaces.topology(Spaces.horizontal_space(extrued_space)) &&
           quadrature_style(hspace) ===
           quadrature_style(Spaces.horizontal_space(extrued_space))
end


Adapt.adapt_structure(to, space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace(
        space.staggering,
        Adapt.adapt(to, Spaces.horizontal_space(space)),
        Adapt.adapt(to, space.vertical_topology),
        Adapt.adapt(to, space.hypsography),
        Adapt.adapt(to, space.global_geometry),
        Adapt.adapt(to, space.center_local_geometry),
        Adapt.adapt(to, space.face_local_geometry),
        Adapt.adapt(to, space.center_ghost_geometry),
        Adapt.adapt(to, space.face_ghost_geometry),
    )



const CenterExtrudedFiniteDifferenceSpace2D =
    CenterExtrudedFiniteDifferenceSpace{<:SpectralElementSpace1D}
const CenterExtrudedFiniteDifferenceSpace3D =
    CenterExtrudedFiniteDifferenceSpace{<:SpectralElementSpace2D}
const FaceExtrudedFiniteDifferenceSpace2D =
    FaceSpace{<:CenterExtrudedFiniteDifferenceSpace2D}
const FaceExtrudedFiniteDifferenceSpace3D =
    FaceSpace{<:CenterExtrudedFiniteDifferenceSpace3D}

CenterExtrudedFiniteDifferenceSpace(
    space::CenterExtrudedFiniteDifferenceSpace,
) = space
CenterExtrudedFiniteDifferenceSpace(space::FaceExtrudedFiniteDifferenceSpace) =
    space.center_space
FaceExtrudedFiniteDifferenceSpace(space::CenterExtrudedFiniteDifferenceSpace) =
    FaceSpace(space)
FaceExtrudedFiniteDifferenceSpace(space::FaceExtrudedFiniteDifferenceSpace) =
    space

function Base.show(io::IO, space::CenterExtrudedFiniteDifferenceSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(
        io,
        space isa CenterExtrudedFiniteDifferenceSpace ?
        "CenterExtrudedFiniteDifferenceSpace" :
        "FaceExtrudedFiniteDifferenceSpace",
        ":",
    )
    print(iio, " "^(indent + 2), "context: ")
    Topologies.print_context(iio, space.horizontal_space.topology.context)
    println(iio)
    println(iio, " "^(indent + 2), "horizontal:")
    println(
        iio,
        " "^(indent + 4),
        "mesh: ",
        space.horizontal_space.topology.mesh,
    )
    println(
        iio,
        " "^(indent + 4),
        "quadrature: ",
        space.horizontal_space.quadrature_style,
    )
    println(iio, " "^(indent + 2), "vertical:")
    print(iio, " "^(indent + 4), "mesh: ", space.vertical_topology.mesh)
end
local_geometry_data(space::CenterExtrudedFiniteDifferenceSpace) =
    space.center_local_geometry

local_geometry_data(space::FaceExtrudedFiniteDifferenceSpace) =
    center_space(space).face_local_geometry

# TODO: will need to be defined for distributed
ghost_geometry_data(space::CenterExtrudedFiniteDifferenceSpace) =
    space.center_ghost_geometry
ghost_geometry_data(space::FaceExtrudedFiniteDifferenceSpace) =
    space.center_space.face_ghost_geometry


ExtrudedFiniteDifferenceSpace(
    horizontal_space::AbstractSpace,
    vertical_space::CenterFiniteDifferenceSpace,
    hypsography::Flat = Flat(),
) = CenterExtrudedFiniteDifferenceSpace(
    horizontal_space,
    vertical_space,
    hypsography,
)
ExtrudedFiniteDifferenceSpace(
    horizontal_space::AbstractSpace,
    vertical_space::FaceFiniteDifferenceSpace,
    hypsography::Flat = Flat(),
) = FaceSpace(
    CenterExtrudedFiniteDifferenceSpace(
        horizontal_space,
        center_space(vertical_space),
        hypsography,
    ),
)

@memoize WeakValueDict function CenterExtrudedFiniteDifferenceSpace(
    horizontal_space::AbstractSpace,
    vertical_space::CenterFiniteDifferenceSpace,
    hypsography::Flat = Flat(),
)
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
    return CenterExtrudedFiniteDifferenceSpace(
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

quadrature_style(space::CenterExtrudedFiniteDifferenceSpace) =
    space.horizontal_space.quadrature_style
quadrature_style(space::FaceExtrudedFiniteDifferenceSpace) =
    quadrature_style(space.center_space)

topology(space::CenterExtrudedFiniteDifferenceSpace) =
    space.horizontal_space.topology
topology(space::FaceExtrudedFiniteDifferenceSpace) =
    topology(space.center_space)

topology(space::ExtrudedFiniteDifferenceSpace) = topology(horizontal_space(space))
ClimaComms.device(space::ExtrudedFiniteDifferenceSpace) =
    ClimaComms.device(topology(space))
vertical_topology(space::CenterExtrudedFiniteDifferenceSpace) =
    space.vertical_topology
vertical_topology(space::FaceExtrudedFiniteDifferenceSpace) =
    vertical_topology(space.center_space)

Base.@propagate_inbounds function slab(
    space::ExtrudedFiniteDifferenceSpace,
    v,
    h,
)
    SpectralElementSpaceSlab(
        Spaces.horizontal_space(space).quadrature_style,
        slab(local_geometry_data(space), v, h),
    )
end

"""
    ColumnIndex(ij,h)

An index into a column of a field. This can be used as an argument to `getindex`
of a `Field`, to return a field on that column.

# Example
```julia
colidx = ColumnIndex((1,1),1)
field[colidx]
```
"""
struct ColumnIndex{N}
    ij::NTuple{N, Int}
    h::Int
end



struct CenterColumnSpace{S<:CenterExtrudedFiniteDifferenceSpace, C<:ColumnIndex} <: AbstractFiniteDifferenceSpace
    space::S
    column::C
end

const FaceColumnSpace = FaceSpace{<:CenterColumnSpace}

const ColumnSpace = Union{
    CenterColumnSpace,
    FaceColumnSpace,
}

face_space(space::CenterColumnSpace) = 
    FaceSpace(space)
center_space(space::CenterColumnSpace) = 
    space

full_space(colspace::CenterColumnSpace) = colspace.space
full_space(colspace::FaceColumnSpace) = FaceSpace(center_space(colspace).space)

column(
    space::CenterExtrudedFiniteDifferenceSpace,
    colidx::ColumnIndex
) =
    CenterColumnSpace(space, colidx)

column(
    space::FaceExtrudedFiniteDifferenceSpace,
    colidx::ColumnIndex
) =
    FaceSpace(CenterColumnSpace(center_space(space), colidx))
    

vertical_topology(space::ColumnSpace) = vertical_topology(full_space(space))

function local_geometry_data(
    columnspace::ColumnSpace,
)
    column(local_geometry_data(full_space(columnspace)), center_space(columnspace).column)
end



ClimaComms.device(columnspace::ColumnSpace) =
    ClimaComms.device(full_space(columnspace))
ClimaComms.context(columnspace::ColumnSpace) =
    ClimaComms.context(full_space(columnspace))


# TODO: deprecate these
column(
    space::ExtrudedFiniteDifferenceSpace,
    i,
    j,
    h,
) = 
    column(space, ColumnIndex((i, j), h))


struct LevelSpace{S, L} <: AbstractSpace
    space::S
    level::L
end

level(space::CenterExtrudedFiniteDifferenceSpace, v::Integer) =
    LevelSpace(space, v)
level(space::FaceExtrudedFiniteDifferenceSpace, v::PlusHalf) =
    LevelSpace(space, v)

function local_geometry_data(
    levelspace::LevelSpace{<:CenterExtrudedFiniteDifferenceSpace, <:Integer},
)
    level(local_geometry_data(levelspace.space), levelspace.level)
end
function local_geometry_data(
    levelspace::LevelSpace{<:FaceExtrudedFiniteDifferenceSpace, <:PlusHalf},
)
    level(local_geometry_data(levelspace.space), levelspace.level + half)
end

function column(levelspace::LevelSpace, args...)
    local_geometry = column(local_geometry_data(levelspace), args...)
    PointSpace(local_geometry)
end


nlevels(space::CenterExtrudedFiniteDifferenceSpace) =
    size(space.center_local_geometry, 4)

nlevels(space::FaceExtrudedFiniteDifferenceSpace) =
    size(center_space(space).face_local_geometry, 4)

function left_boundary_name(space::ExtrudedFiniteDifferenceSpace)
    boundaries = Topologies.boundaries(Spaces.vertical_topology(space))
    propertynames(boundaries)[1]
end
function right_boundary_name(space::ExtrudedFiniteDifferenceSpace)
    boundaries = Topologies.boundaries(Spaces.vertical_topology(space))
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
    h_iter = eachslabindex(cSpaces.horizontal_space(space))
    Nv = size(cspace.center_local_geometry, 4)
    return Iterators.product(1:Nv, h_iter)
end
function eachslabindex(fspace::FaceExtrudedFiniteDifferenceSpace)
    h_iter = eachslabindex(fSpaces.horizontal_space(space))
    Nv = size(fspace.face_local_geometry, 4)
    return Iterators.product(1:Nv, h_iter)
end
