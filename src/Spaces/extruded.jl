#####
##### Hybrid mesh
#####

abstract type HypsographyAdaption end

"""
    Flat()

No surface hypsography.
"""
struct Flat <: HypsographyAdaption end


"""
    ExtrudedFiniteDifferenceGrid(
        horizontal_space::AbstractSpace,
        vertical_space::FiniteDifferenceSpace,
        hypsography::HypsographyAdaption = Flat(),
    )

Construct an `ExtrudedFiniteDifferenceGrid` from the horizontal and vertical spaces.
"""
mutable struct ExtrudedFiniteDifferenceGrid{
    H <: AbstractGrid,
    T <: Topologies.AbstractIntervalTopology,
    A <: HypsographyAdaption,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
} <: AbstractGrid
    horizontal_grid::H
    vertical_topology::T # should we cache the vertical grid?
    hypsography::A
    global_geometry::GG
    center_local_geometry::LG
    face_local_geometry::LG
end

@memoize WeakValueDict function ExtrudedFiniteDifferenceGrid(
    horizontal_grid::Union{SpectralElementGrid1D, SpectralElementGrid2D},
    vertical_grid::FiniteDifferenceGrid,
    hypsography::Flat = Flat(),
)
    vertical_topology = vertical_grid.topology
    global_geometry = horizontal_grid.global_geometry
    center_local_geometry =
        product_geometry.(
            horizontal_grid.local_geometry,
            vertical_grid.center_local_geometry,
        )
    face_local_geometry =
        product_geometry.(
            horizontal_grid.local_geometry,
            vertical_grid.face_local_geometry,
        )

    return ExtrudedFiniteDifferenceGrid(
        horizontal_grid,
        vertical_topology,
        hypsography,
        global_geometry,
        center_local_geometry,
        face_local_geometry,
    )
end




struct ExtrudedFiniteDifferenceSpace{
    S <: Staggering,
    G <: ExtrudedFiniteDifferenceGrid,
} <: AbstractSpace
    staggering::S
    grid::G
end

const FaceExtrudedFiniteDifferenceSpace =
    ExtrudedFiniteDifferenceSpace{CellFace}
const CenterExtrudedFiniteDifferenceSpace =
    ExtrudedFiniteDifferenceSpace{CellCenter}


ExtrudedFiniteDifferenceSpace{S}(
    grid::ExtrudedFiniteDifferenceGrid,
) where {S <: Staggering} = ExtrudedFiniteDifferenceSpace(S(), grid)
ExtrudedFiniteDifferenceSpace{S}(
    space::ExtrudedFiniteDifferenceSpace,
) where {S <: Staggering} = ExtrudedFiniteDifferenceSpace{S}(space.grid)


function ExtrudedFiniteDifferenceSpace(
    horizontal_space::AbstractSpace,
    vertical_space::FiniteDifferenceSpace{S},
    hypsography::HypsographyAdaption = Flat(),
) where {S <: Staggering}
    grid = ExtrudedFiniteDifferenceGrid(
        horizontal_space.grid,
        vertical_space.grid,
        hypsography,
    )
    return ExtrudedFiniteDifferenceSpace{S}(grid)
end

local_dss_weights(grid::ExtrudedFiniteDifferenceGrid) =
    local_dss_weights(grid.horizontal_grid)
local_dss_weights(space::ExtrudedFiniteDifferenceSpace) =
    local_dss_weights(space.grid)

face_space(space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace{CellFace}(space)
center_space(space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace{CellCenter}(space)




ExtrudedFiniteDifferenceSpace{S}(
    horizontal_space::AbstractSpace,
    vertical_space::FiniteDifferenceSpace,
    hypsography::HypsographyAdaption = Flat(),
) where {S <: Staggering} = ExtrudedFiniteDifferenceSpace{S}(
    ExtrudedFiniteDifferenceGrid(horizontal_space, vertical_space, hypsography),
)

#=
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
=#

Adapt.adapt_structure(to, space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace(
        space.staggering,
        Adapt.adapt(to, Spaces.horizontal_space(space)),
        Adapt.adapt(to, space.vertical_topology),
        Adapt.adapt(to, space.hypsography),
        Adapt.adapt(to, space.global_geometry),
        Adapt.adapt(to, space.center_local_geometry),
        Adapt.adapt(to, space.face_local_geometry),
    )



const CenterExtrudedFiniteDifferenceSpace2D =
    CenterExtrudedFiniteDifferenceSpace{<:SpectralElementSpace1D}
const CenterExtrudedFiniteDifferenceSpace3D =
    CenterExtrudedFiniteDifferenceSpace{<:SpectralElementSpace2D}
const FaceExtrudedFiniteDifferenceSpace2D =
    FaceExtrudedFiniteDifferenceSpace{<:SpectralElementSpace1D}
const FaceExtrudedFiniteDifferenceSpace3D =
    FaceExtrudedFiniteDifferenceSpace{<:SpectralElementSpace2D}

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


local_geometry_data(::CellCenter, grid::ExtrudedFiniteDifferenceGrid) =
    grid.center_local_geometry
local_geometry_data(::CellFace, grid::ExtrudedFiniteDifferenceGrid) =
    grid.face_local_geometry
local_geometry_data(space::ExtrudedFiniteDifferenceSpace) =
    local_geometry_data(space.staggering, space.grid)

quadrature_style(grid::ExtrudedFiniteDifferenceGrid) =
    quadrature_style(grid.horizontal_grid)
quadrature_style(space::ExtrudedFiniteDifferenceSpace) =
    quadrature_style(space.grid)
topology(grid::ExtrudedFiniteDifferenceGrid) = topology(grid.horizontal_grid)
topology(space::ExtrudedFiniteDifferenceSpace) = topology(space.grid)


ClimaComms.device(grid::ExtrudedFiniteDifferenceGrid) =
    ClimaComms.device(topology(grid))
ClimaComms.context(grid::ExtrudedFiniteDifferenceGrid) =
    ClimaComms.context(topology(grid))

ClimaComms.device(space::ExtrudedFiniteDifferenceSpace) =
    ClimaComms.device(space.grid)
ClimaComms.context(space::ExtrudedFiniteDifferenceSpace) =
    ClimaComms.context(space.grid)

horizontal_space(space::ExtrudedFiniteDifferenceSpace) =
    AbstractSpectralElementSpace(space.grid.horizontal_grid)

vertical_topology(grid::ExtrudedFiniteDifferenceGrid) = grid.vertical_topology
vertical_topology(space::ExtrudedFiniteDifferenceSpace) =
    vertical_topology(space.grid)

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


struct ColumnGrid{G <: ExtrudedFiniteDifferenceGrid, C <: ColumnIndex} <:
       AbstractFiniteDifferenceGrid
    full_grid::G
    colidx::C
end


column(grid::ExtrudedFiniteDifferenceGrid, colidx::ColumnIndex) =
    ColumnGrid(grid, colidx)


function column(space::ExtrudedFiniteDifferenceSpace, colidx::ColumnIndex)
    column_grid = column(space.grid, colidx)
    FiniteDifferenceSpace(space.staggering, column_grid)
end

vertical_topology(colgrid::ColumnGrid) = colgrid.full_grid.vertical_topology

local_geometry_data(staggering::Staggering, colgrid::ColumnGrid) =
    column(local_geometry_data(staggering, colgrid.full_grid), colgrid.colidx)

topology(colgrid::ColumnGrid) = vertical_topology(colgrid.full_grid)

ClimaComms.device(colgrid::ColumnGrid) = ClimaComms.device(colgrid.full_grid)
ClimaComms.context(colgrid::ColumnGrid) = ClimaComms.context(colgrid.full_grid)


# TODO: deprecate these
column(space::ExtrudedFiniteDifferenceSpace, i, j, h) =
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


nlevels(space::ExtrudedFiniteDifferenceSpace) =
    size(local_geometry_data(space), 4)

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
