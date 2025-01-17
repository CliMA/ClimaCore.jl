
"""
    ExtrudedFiniteDifferenceSpace(grid, staggering)

    ExtrudedFiniteDifferenceSpace(
        horizontal_space::AbstractSpace,
        vertical_space::FiniteDifferenceSpace,
        hypsography::Grids.HypsographyAdaption = Grids.Flat();
        deep::Bool = false,
    )

An extruded finite-difference space,
where the extruded direction is _staggered_,
containing grid information at either
 - cell centers (where `staggering` is [`Grids.CellCenter`](@ref)) or
 - cell faces (where `staggering` is [`Grids.CellFace`](@ref))
"""
struct ExtrudedFiniteDifferenceSpace{
    G <: Grids.AbstractExtrudedFiniteDifferenceGrid,
    S <: Staggering,
} <: AbstractSpace
    grid::G
    staggering::S
end

local_geometry_type(::Type{ExtrudedFiniteDifferenceSpace{G, S}}) where {G, S} =
    local_geometry_type(G)

space(grid::Grids.ExtrudedFiniteDifferenceGrid, staggering::Staggering) =
    ExtrudedFiniteDifferenceSpace(grid, staggering)

const FaceExtrudedFiniteDifferenceSpace{G} =
    ExtrudedFiniteDifferenceSpace{G, CellFace}
const CenterExtrudedFiniteDifferenceSpace{G} =
    ExtrudedFiniteDifferenceSpace{G, CellCenter}

"""
    face_space(space::ExtrudedFiniteDifferenceSpace)

Return face-centered space corresponding to `space`.

If `space` is already face-centered, return itself.
"""
function face_space(space::ExtrudedFiniteDifferenceSpace)
    return ExtrudedFiniteDifferenceSpace(grid(space), CellFace())
end

"""
    center_space(space::ExtrudedFiniteDifferenceSpace)

Return center-centered space corresponding to `space`.

If `space` is already center-centered, return itself.
"""
function center_space(space::ExtrudedFiniteDifferenceSpace)
    return ExtrudedFiniteDifferenceSpace(grid(space), CellCenter())
end


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
    hypsography::Grids.HypsographyAdaption = Grids.Flat();
    deep = false,
)
    grid_space = Grids.ExtrudedFiniteDifferenceGrid(
        grid(horizontal_space),
        grid(vertical_space),
        hypsography;
        deep,
    )
    return ExtrudedFiniteDifferenceSpace(grid_space, vertical_space.staggering)
end

FaceExtrudedFiniteDifferenceSpace(grid::Grids.ExtrudedFiniteDifferenceGrid) =
    ExtrudedFiniteDifferenceSpace(grid, CellFace())
CenterExtrudedFiniteDifferenceSpace(grid::Grids.ExtrudedFiniteDifferenceGrid) =
    ExtrudedFiniteDifferenceSpace(grid, CellCenter())
FaceExtrudedFiniteDifferenceSpace(space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace(grid(space), CellFace())
CenterExtrudedFiniteDifferenceSpace(space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace(grid(space), CellCenter())

staggering(space::ExtrudedFiniteDifferenceSpace) = getfield(space, :staggering)
grid(space::ExtrudedFiniteDifferenceSpace) = getfield(space, :grid)
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

function issubspace(
    hspace::AbstractSpectralElementSpace,
    extruded_space::ExtrudedFiniteDifferenceSpace,
)
    return grid(hspace) === grid(extruded_space).horizontal_grid
end
function issubspace(
    level_space::SpectralElementSpace2D{<:Grids.LevelGrid},
    extruded_space::ExtrudedFiniteDifferenceSpace,
)
    return grid(level_space).full_grid === grid(extruded_space)
end
function issubspace(
    level_space::SpectralElementSpace1D{<:Grids.LevelGrid},
    extruded_space::ExtrudedFiniteDifferenceSpace,
)
    return grid(level_space).full_grid === grid(extruded_space)
end


Adapt.adapt_structure(to, space::ExtrudedFiniteDifferenceSpace) =
    ExtrudedFiniteDifferenceSpace(
        Adapt.adapt(to, grid(space)),
        staggering(space),
    )

const ExtrudedFiniteDifferenceSpace2D = ExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedFiniteDifferenceGrid{<:Grids.SpectralElementGrid1D},
}
const ExtrudedFiniteDifferenceSpace3D = ExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedFiniteDifferenceGrid{<:Grids.SpectralElementGrid2D},
}
const ExtrudedSpectralElementSpace2D =
    ExtrudedFiniteDifferenceSpace{<:Grids.ExtrudedSpectralElementGrid2D}
const ExtrudedSpectralElementSpace3D =
    ExtrudedFiniteDifferenceSpace{<:Grids.ExtrudedSpectralElementGrid3D}

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
    print(iio, " "^(indent + 4), "mesh: ", vertical_topology(space).mesh)
end

quadrature_style(space::ExtrudedFiniteDifferenceSpace) =
    quadrature_style(grid(space))
topology(space::ExtrudedFiniteDifferenceSpace) = topology(grid(space))


horizontal_space(full_space::ExtrudedFiniteDifferenceSpace) =
    space(grid(full_space).horizontal_grid, nothing)

vertical_topology(space::ExtrudedFiniteDifferenceSpace) =
    vertical_topology(grid(space))



function column(space::ExtrudedFiniteDifferenceSpace, colidx::Grids.ColumnIndex)
    column_grid = column(grid(space), colidx)
    FiniteDifferenceSpace(column_grid, space.staggering)
end





Base.@propagate_inbounds function slab(
    space::ExtrudedFiniteDifferenceSpace,
    v,
    h,
)
    SpectralElementSpaceSlab(
        Spaces.quadrature_style(space),
        slab(local_geometry_data(space), v, h),
    )
end



# TODO: deprecate these
column(space::ExtrudedFiniteDifferenceSpace, i, j, h) =
    column(space, Grids.ColumnIndex((i, j), h))
column(space::ExtrudedFiniteDifferenceSpace, i, h) =
    column(space, Grids.ColumnIndex((i,), h))

level(space::CenterExtrudedFiniteDifferenceSpace2D, v::Integer) =
    SpectralElementSpace1D(level(grid(space), v))
level(space::FaceExtrudedFiniteDifferenceSpace2D, v::PlusHalf) =
    SpectralElementSpace1D(level(grid(space), v))
level(space::CenterExtrudedFiniteDifferenceSpace3D, v::Integer) =
    SpectralElementSpace2D(level(grid(space), v))
level(space::FaceExtrudedFiniteDifferenceSpace3D, v::PlusHalf) =
    SpectralElementSpace2D(level(grid(space), v))


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
    center_local_geometry =
        local_geometry_data(grid(cspace), Grids.CellCenter())
    Nv = size(center_local_geometry, 4)
    return Iterators.product(1:Nv, h_iter)
end
function eachslabindex(fspace::FaceExtrudedFiniteDifferenceSpace)
    h_iter = eachslabindex(Spaces.horizontal_space(fspace))
    face_local_geometry = local_geometry_data(grid(fspace), Grids.CellFace())
    Nv = size(face_local_geometry, 4)
    return Iterators.product(1:Nv, h_iter)
end


## aliases
const ExtrudedRectilinearSpectralElementSpace3D = ExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedRectilinearSpectralElementGrid3D,
}
const ExtrudedCubedSphereSpectralElementSpace3D = ExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedCubedSphereSpectralElementGrid3D,
}
