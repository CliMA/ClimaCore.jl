

struct ExtrudedFiniteDifferenceSpace{
    G <: Grids.ExtrudedFiniteDifferenceGrid,
    S <: Staggering,
} <: AbstractSpace
    grid::G
    staggering::S
end

space(grid::Grids.ExtrudedFiniteDifferenceGrid, staggering::Staggering) =
    ExtrudedFiniteDifferenceSpace(grid, staggering)

const FaceExtrudedFiniteDifferenceSpace{G} =
    ExtrudedFiniteDifferenceSpace{G, CellFace}
const CenterExtrudedFiniteDifferenceSpace{G} =
    ExtrudedFiniteDifferenceSpace{G, CellCenter}

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

FaceExtrudedFiniteDifferenceSpace(grid::Grids.ExtrudedFiniteDifferenceGrid) =
    ExtrudedFiniteDifferenceSpace(grid, CellFace())
CenterExtrudedFiniteDifferenceSpace(grid::Grids.ExtrudedFiniteDifferenceGrid) =
    ExtrudedFiniteDifferenceSpace(grid, CellCenter())
FaceExtrudedFiniteDifferenceSpace(space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace(space.grid, CellFace())
CenterExtrudedFiniteDifferenceSpace(space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace(space.grid, CellCenter())


local_dss_weights(space::ExtrudedFiniteDifferenceSpace) =
    local_dss_weights(grid(space))

staggering(space::ExtrudedFiniteDifferenceSpace) = space.staggering
grid(space::ExtrudedFiniteDifferenceSpace) = space.grid
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
    ExtrudedFiniteDifferenceSpace(Adapt.adapt(to, space.grid), space.staggering)

const ExtrudedFiniteDifferenceSpace2D = ExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedFiniteDifferenceGrid{<:Grids.SpectralElementGrid1D},
}
const ExtrudedFiniteDifferenceSpace3D = ExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedFiniteDifferenceGrid{<:Grids.SpectralElementGrid1D},
}

const CenterExtrudedFiniteDifferenceSpace2D =
    CenterExtrudedFiniteDifferenceSpace{
        <:Grids.ExtrudedFiniteDifferenceGrid{<:Grids.SpectralElementGrid1D},
    }
const CenterExtrudedFiniteDifferenceSpace3D =
    CenterExtrudedFiniteDifferenceSpace{
        <:Grids.ExtrudedFiniteDifferenceGrid{<:Grids.SpectralElementGrid2D},
    }
const FaceExtrudedFiniteDifferenceSpace2D = FaceExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedFiniteDifferenceGrid{<:Grids.SpectralElementGrid1D},
}
const FaceExtrudedFiniteDifferenceSpace3D = FaceExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedFiniteDifferenceGrid{<:Grids.SpectralElementGrid2D},
}

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
    hspace = Spaces.horizontal_space(space)
    Topologies.print_context(iio, Spaces.topology(hspace).context)
    println(iio)
    println(iio, " "^(indent + 2), "horizontal:")
    println(iio, " "^(indent + 4), "mesh: ", Spaces.topology(hspace).mesh)
    println(iio, " "^(indent + 4), "quadrature: ", quadrature_style(hspace))
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
    column(space, Grids.ColumnIndex((i, j), h))


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

function eachslabindex(cspace::CenterExtrudedFiniteDifferenceSpace)
    h_iter = eachslabindex(Spaces.horizontal_space(cspace))
    Nv = size(cspace.center_local_geometry, 4)
    return Iterators.product(1:Nv, h_iter)
end
function eachslabindex(fspace::FaceExtrudedFiniteDifferenceSpace)
    h_iter = eachslabindex(Spaces.horizontal_space(fspace))
    Nv = size(fspace.face_local_geometry, 4)
    return Iterators.product(1:Nv, h_iter)
end
