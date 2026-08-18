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

FiniteDifferenceSpace(space::ExtrudedFiniteDifferenceSpace) =
    FiniteDifferenceSpace(
        grid(space).vertical_grid,
        staggering(space),
    )

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
    Topologies.print_context(iio, ClimaComms.context(space))
    if has_horizontal(space)
        hspace = horizontal_space(space)
        hmesh = topology(hspace).mesh
        Topologies.print_context(iio, topology(hspace).context)
        println(iio)
        println(iio, " "^(indent + 2), "horizontal:")
        println(iio, " "^(indent + 4), "mesh: ", hmesh)
        println(
            iio,
            " "^(indent + 4),
            "node_horizontal_length_scale: ",
            node_horizontal_length_scale(hspace),
        )
        println(
            iio,
            " "^(indent + 4),
            "element_horizontal_length_scale: ",
            Meshes.element_horizontal_length_scale(hmesh),
        )
        println(iio, " "^(indent + 4), "quadrature: ", quadrature_style(hspace))
    end
    if has_vertical(space)
        println(iio, " "^(indent + 2), "vertical:")
        print(iio, " "^(indent + 4), "mesh: ", vertical_topology(space).mesh)
    end
end

quadrature_style(space::ExtrudedFiniteDifferenceSpace) =
    quadrature_style(grid(space))
topology(space::ExtrudedFiniteDifferenceSpace) = topology(grid(space))


horizontal_space(full_space::ExtrudedFiniteDifferenceSpace) =
    space(grid(full_space).horizontal_grid, nothing)

vertical_topology(space::ExtrudedFiniteDifferenceSpace) =
    vertical_topology(grid(space))

issubspace(subspace::AbstractSpectralElementSpace, space::ExtrudedFiniteDifferenceSpace) =
    grid(subspace) === grid(space).horizontal_grid ||
    (grid(subspace) isa Grids.LevelGrid && grid(subspace).full_grid === grid(space))
issubspace(subspace::FiniteDifferenceSpace, space::ExtrudedFiniteDifferenceSpace) =
    grid(subspace) === grid(space).vertical_grid ||
    (grid(subspace) isa Grids.ColumnGrid && grid(subspace).full_grid === grid(space))

# This must also cover device-side spaces, which appear in place of their host
# counterparts inside GPU kernels (see reconstruct_placeholder_space). Without a
# method that matches, `level` falls back to the generic identity method, which
# silently returns the extruded space itself. Device-side grids do not store
# their horizontal grid, so the aliases above cannot tell how many horizontal
# dimensions such a space spans; the directions spanned by the coordinates of
# its local geometry, which host and device spaces share, stand in for that.
Base.@propagate_inbounds level(space::ExtrudedFiniteDifferenceSpace, v) =
    _level_space(
        eltype(local_geometry_data(space)),
        level(grid(space), staggered_level_index(space, v)),
    )
_level_space(::Type{<:Geometry.LocalGeometry{(1, 2, 3)}}, level_grid) =
    SpectralElementSpace2D(level_grid)
_level_space(
    ::Type{
        <:Union{Geometry.LocalGeometry{(1, 3)}, Geometry.LocalGeometry{(2, 3)}},
    },
    level_grid,
) = SpectralElementSpace1D(level_grid)

Base.@propagate_inbounds slab(space::ExtrudedFiniteDifferenceSpace, v, h) =
    SpectralElementSpaceSlab(
        quadrature_style(space),
        slab(local_geometry_data(space), integer_level_index(space, v), h),
    )

Base.@propagate_inbounds column(space::ExtrudedFiniteDifferenceSpace, indices...) =
    FiniteDifferenceSpace(column(grid(space), indices...), space.staggering)

nlevels(space::ExtrudedFiniteDifferenceSpace) =
    size(local_geometry_data(space), 1)

function left_boundary_name(space::ExtrudedFiniteDifferenceSpace)
    boundaries = Topologies.boundaries(vertical_topology(space))
    propertynames(boundaries)[1]
end
function right_boundary_name(space::ExtrudedFiniteDifferenceSpace)
    boundaries = Topologies.boundaries(vertical_topology(space))
    propertynames(boundaries)[2]
end

function eachslabindex(cspace::CenterExtrudedFiniteDifferenceSpace)
    h_iter = eachslabindex(horizontal_space(cspace))
    center_local_geometry =
        local_geometry_data(grid(cspace), Grids.CellCenter())
    Nv = size(center_local_geometry, 1)
    return Iterators.product(1:Nv, h_iter)
end
function eachslabindex(fspace::FaceExtrudedFiniteDifferenceSpace)
    h_iter = eachslabindex(horizontal_space(fspace))
    face_local_geometry = local_geometry_data(grid(fspace), Grids.CellFace())
    Nv = size(face_local_geometry, 1)
    return Iterators.product(1:Nv, h_iter)
end


## aliases
const ExtrudedRectilinearSpectralElementSpace3D = ExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedRectilinearSpectralElementGrid3D,
}
const ExtrudedCubedSphereSpectralElementSpace3D = ExtrudedFiniteDifferenceSpace{
    <:Grids.ExtrudedCubedSphereSpectralElementGrid3D,
}
