

struct ExtrudedFiniteDifferenceSpace{
    G <: Grids.ExtrudedFiniteDifferenceGrid,
    S <: Staggering,
} <: AbstractSpace
    grid::G
    staggering::S
end

space(staggering::Staggering, grid::Grids.ExtrudedFiniteDifferenceGrid) =
    ExtrudedFiniteDifferenceSpace(staggering, grid)

const FaceExtrudedFiniteDifferenceSpace{G} =
    ExtrudedFiniteDifferenceSpace{G,CellFace}
const CenterExtrudedFiniteDifferenceSpace{G} =
    ExtrudedFiniteDifferenceSpace{G,CellCenter}

#=
ExtrudedFiniteDifferenceSpace{S}(
    grid::Grids.ExtrudedFiniteDifferenceGrid,
) where {S <: Staggering} = ExtrudedFiniteDifferenceSpace(S(), grid)
ExtrudedFiniteDifferenceSpace{S}(
    space::ExtrudedFiniteDifferenceSpace,
) where {S <: Staggering} = ExtrudedFiniteDifferenceSpace{S}(space.grid)
=#

function ExtrudedFiniteDifferenceSpace(
    horizontal_space::AbstractSpace,
    vertical_space::FiniteDifferenceSpace,
    hypsography::Grids.HypsographyAdaption = Grids.Flat(),
)
    grid = Grids.ExtrudedFiniteDifferenceGrid(
        horizontal_space.grid,
        vertical_space.grid,
        hypsography,
    )
    return ExtrudedFiniteDifferenceSpace(grid, vertical_space.staggering)
end

local_dss_weights(space::ExtrudedFiniteDifferenceSpace) =
    local_dss_weights(grid(space))

staggering(space::ExtrudedFiniteDifferenceSpace) = space.staggering

space(space::ExtrudedFiniteDifferenceSpace, staggering::Staggering) =
    ExtrudedFiniteDifferenceSpace(grid(space), staggering)

#=

ExtrudedFiniteDifferenceSpace{S}(
    horizontal_space::AbstractSpace,
    vertical_space::FiniteDifferenceSpace,
    hypsography::HypsographyAdaption = Flat(),
) where {S <: Staggering} = ExtrudedFiniteDifferenceSpace{S}(
    Grids.ExtrudedFiniteDifferenceGrid(horizontal_space, vertical_space, hypsography),
)
=#

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
        Adapt.adapt(to, space.grid),
        space.staggering,
    )

#=

const CenterExtrudedFiniteDifferenceSpace2D =
    CenterExtrudedFiniteDifferenceSpace{<:SpectralElementSpace1D}
const CenterExtrudedFiniteDifferenceSpace3D =
    CenterExtrudedFiniteDifferenceSpace{<:SpectralElementSpace2D}
const FaceExtrudedFiniteDifferenceSpace2D =
    FaceExtrudedFiniteDifferenceSpace{<:SpectralElementSpace1D}
const FaceExtrudedFiniteDifferenceSpace3D =
    FaceExtrudedFiniteDifferenceSpace{<:SpectralElementSpace2D}
=#
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

quadrature_style(space::ExtrudedFiniteDifferenceSpace) =
    quadrature_style(space.grid)
topology(space::ExtrudedFiniteDifferenceSpace) = topology(space.grid)


horizontal_space(full_space::ExtrudedFiniteDifferenceSpace) =
    space(full_space.grid.horizontal_grid, nothing)

vertical_topology(space::ExtrudedFiniteDifferenceSpace) =
    vertical_topology(space.grid)



function column(space::ExtrudedFiniteDifferenceSpace, colidx::Grids.ColumnIndex)
    column_grid = column(space.grid, colidx)
    FiniteDifferenceSpace(column_grid, space.staggering)
end





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
