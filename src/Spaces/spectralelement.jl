abstract type AbstractSpectralElementSpace <: AbstractSpace end

Topologies.nlocalelems(space::AbstractSpectralElementSpace) =
    Topologies.nlocalelems(Spaces.topology(space))



quadrature_style(space::AbstractSpectralElementSpace) =
    quadrature_style(grid(space))
local_dss_weights(space::AbstractSpectralElementSpace) =
    local_dss_weights(grid(space))

horizontal_space(space::AbstractSpectralElementSpace) = space
nlevels(space::AbstractSpectralElementSpace) = 1

eachslabindex(space::AbstractSpectralElementSpace) =
    1:Topologies.nlocalelems(Spaces.topology(space))

staggering(space::AbstractSpectralElementSpace) = nothing

function Base.show(io::IO, space::AbstractSpectralElementSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, nameof(typeof(space)), ":")
    if hasfield(typeof(grid(space)), :topology)
        # some reduced spaces (like slab space) do not have topology
        print(iio, " "^(indent + 2), "context: ")
        Topologies.print_context(iio, Spaces.topology(grid(space)).context)
        println(iio)
        println(
            iio,
            " "^(indent + 2),
            "mesh: ",
            Spaces.topology(grid(space)).mesh,
        )
    end
    print(
        iio,
        " "^(indent + 2),
        "quadrature: ",
        Spaces.quadrature_style(grid(space)),
    )
end



# 1D
"""
    SpectralElementSpace1D(grid::SpectralElementGrid1D)
"""
struct SpectralElementSpace1D{G} <: AbstractSpectralElementSpace
    grid::G
end
space(grid::Grids.SpectralElementGrid1D, ::Nothing) =
    SpectralElementSpace1D(grid)
space(grid::Grids.LevelGrid{<:Grids.ExtrudedSpectralElementGrid2D}, ::Nothing) =
    SpectralElementSpace1D(grid)
grid(space::Spaces.SpectralElementSpace1D) = getfield(space, :grid)

function SpectralElementSpace1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle,
)
    grid = Grids.SpectralElementGrid1D(topology, quadrature_style)
    SpectralElementSpace1D(grid)
end


@inline function Base.getproperty(space::SpectralElementSpace1D, name::Symbol)
    if name == :topology
        Base.depwarn(
            "`space.topology` is deprecated, use `Spaces.topology(space)` instead",
            :getproperty,
        )
        return topology(space)
    elseif name == :quadrature_style
        Base.depwarn(
            "`space.quadrature_style` is deprecated, use `Spaces.quadrature_style(space)` instead",
            :getproperty,
        )
        return quadrature_style(space)
    elseif name == :global_geometry
        Base.depwarn(
            "`space.global_geometry` is deprecated, use `Spaces.global_geometry(space)` instead",
            :getproperty,
        )
        return global_geometry(space)
    elseif name == :local_geometry
        Base.depwarn(
            "`space.local_geometry` is deprecated, use `Spaces.local_geometry_data(space)` instead",
            :getproperty,
        )
        return local_geometry_data(space)
    elseif name == :local_dss_weights
        Base.depwarn(
            "`space.local_dss_weights` is deprecated, use `Spaces.local_dss_weights(space)` instead",
            :getproperty,
        )
        return local_dss_weights(space)
    end
    return getfield(space, name)
end

# 2D
"""
    SpectralElementSpace2D(grid::SpectralElementGrid1D)
"""
struct SpectralElementSpace2D{G} <: AbstractSpectralElementSpace
    grid::G
end
space(grid::Grids.SpectralElementGrid2D, ::Nothing) =
    SpectralElementSpace2D(grid)
space(grid::Grids.LevelGrid{<:Grids.ExtrudedSpectralElementGrid3D}, ::Nothing) =
    SpectralElementSpace2D(grid)

grid(space::Spaces.SpectralElementSpace2D) = getfield(space, :grid)

function SpectralElementSpace2D(
    topology::Topologies.Topology2D,
    quadrature_style::Quadratures.QuadratureStyle;
    kwargs...,
)
    grid = Grids.SpectralElementGrid2D(topology, quadrature_style; kwargs...)
    SpectralElementSpace2D(grid)
end

@inline function Base.getproperty(space::SpectralElementSpace2D, name::Symbol)
    if name == :topology
        Base.depwarn(
            "`space.topology` is deprecated, use `Spaces.topology(space)` instead",
            :getproperty,
        )
        return topology(space)
    elseif name == :quadrature_style
        Base.depwarn(
            "`space.quadrature_style` is deprecated, use `Spaces.quadrature_style(space)` instead",
            :getproperty,
        )
        return quadrature_style(space)
    elseif name == :global_geometry
        Base.depwarn(
            "`space.global_geometry` is deprecated, use `Spaces.global_geometry(space)` instead",
            :getproperty,
        )
        return global_geometry(space)
    elseif name == :local_geometry
        Base.depwarn(
            "`space.local_geometry` is deprecated, use `Spaces.local_geometry_data(space)` instead",
            :getproperty,
        )
        return local_geometry_data(space)
    elseif name == :ghost_geometry
        Base.depwarn(
            "`space.ghost_geometry` is deprecated, use `nothing` instead",
            :getproperty,
        )
        return nothing
    elseif name == :local_dss_weights
        Base.depwarn(
            "`space.local_dss_weights` is deprecated, use `Spaces.local_dss_weights(space)` instead",
            :getproperty,
        )
        return local_dss_weights(space)
    elseif name == :ghost_dss_weights
        Base.depwarn(
            "`space.ghost_dss_weights` is deprecated, use `nothing` instead",
            :getproperty,
        )
        return nothing
    elseif name == :internal_surface_geometry
        Base.depwarn(
            "`space.internal_surface_geometry` is deprecated, use `Spaces.grid(space).internal_surface_geometry` instead",
            :getproperty,
        )
        return grid(space).internal_surface_geometry
    elseif name == :boundary_surface_geometries
        Base.depwarn(
            "`space.boundary_surface_geometries` is deprecated, use `Spaces.grid(space).boundary_surface_geometries` instead",
            :getproperty,
        )
        return grid(space).boundary_surface_geometries
    end
    return getfield(space, name)
end


Adapt.adapt_structure(to, space::SpectralElementSpace2D) =
    SpectralElementSpace2D(Adapt.adapt(to, grid(space)))


function issubspace(
    hspace::SpectralElementSpace2D{<:Grids.SpectralElementGrid2D},
    level_space::SpectralElementSpace2D{<:Grids.LevelGrid},
)
    return grid(hspace) === grid(level_space).full_grid.horizontal_grid
end


"""
    SpectralElementSpaceSlab <: AbstractSpace

A view into a `SpectralElementSpace2D` for a single slab.
"""
struct SpectralElementSpaceSlab{Q, G} <: AbstractSpectralElementSpace
    quadrature_style::Q
    local_geometry::G
end

const SpectralElementSpaceSlab1D =
    SpectralElementSpaceSlab{Q, DL} where {Q, DL <: DataLayouts.DataSlab1D}

const SpectralElementSpaceSlab2D =
    SpectralElementSpaceSlab{Q, DL} where {Q, DL <: DataLayouts.DataSlab2D}

nlevels(space::SpectralElementSpaceSlab1D) = 1
nlevels(space::SpectralElementSpaceSlab2D) = 1


"""
    Spaces.node_horizontal_length_scale(space::AbstractSpectralElementSpace)

The approximate length scale of the distance between nodes. This is defined as the
length scale of the mesh (see [`Meshes.element_horizontal_length_scale`](@ref)), divided by the
number of unique quadrature points along each dimension.
"""
function node_horizontal_length_scale(space::AbstractSpectralElementSpace)
    quad = quadrature_style(space)
    Nu = Quadratures.unique_degrees_of_freedom(quad)
    return Meshes.element_horizontal_length_scale(space.topology.mesh) / Nu
end





Base.@propagate_inbounds function slab(
    space::AbstractSpectralElementSpace,
    v,
    h,
)
    SpectralElementSpaceSlab(
        quadrature_style(space),
        slab(space.local_geometry, v, h),
    )
end
Base.@propagate_inbounds slab(space::AbstractSpectralElementSpace, h) =
    @inbounds slab(space, 1, h)

Base.@propagate_inbounds function column(space::SpectralElementSpace1D, i, h)
    local_geometry = column(local_geometry_data(space), i, h)
    PointSpace(ClimaComms.context(space), local_geometry)
end
Base.@propagate_inbounds column(space::SpectralElementSpace1D, i, j, h) =
    column(space, i, h)

Base.@propagate_inbounds function column(space::SpectralElementSpace2D, i, j, h)
    local_geometry = column(local_geometry_data(space), i, j, h)
    PointSpace(ClimaComms.context(space), local_geometry)
end

function all_nodes(space::SpectralElementSpace2D)
    Nq = Quadratures.degrees_of_freedom(quadrature_style(space))
    nelem = Topologies.nlocalelems(topology(space))
    Iterators.product(Iterators.product(1:Nq, 1:Nq), 1:nelem)
end

"""
    unique_nodes(space::SpectralElementSpace2D)

An iterator over the unique nodes of `space`. Each node is represented by the
first `((i,j), e)` triple.

This function is experimental, and may change in future.
"""
unique_nodes(space::SpectralElementSpace2D) =
    unique_nodes(space, quadrature_style(space))

unique_nodes(space::SpectralElementSpace2D, quad::Quadratures.QuadratureStyle) =
    UniqueNodeIterator(space)
unique_nodes(space::SpectralElementSpace2D, ::Quadratures.GL) = all_nodes(space)

struct UniqueNodeIterator{S}
    space::S
end

Base.eltype(iter::UniqueNodeIterator{<:SpectralElementSpace2D}) =
    Tuple{Tuple{Int, Int}, Int}

function Base.length(iter::UniqueNodeIterator{<:SpectralElementSpace2D})
    space = iter.space
    topology = Spaces.topology(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style(space))

    nelem = Topologies.nlocalelems(topology)
    nvert = length(Topologies.local_vertices(topology))
    nface_interior = length(Topologies.interior_faces(topology))
    if isempty(Topologies.boundary_tags(topology))
        nface_boundary = 0
    else
        nface_boundary = sum(Topologies.boundary_tags(topology)) do tag
            length(Topologies.boundary_faces(topology, tag))
        end
    end
    return nelem * (Nq - 2)^2 +
           nvert +
           nface_interior * (Nq - 2) +
           nface_boundary * (Nq - 2)
end
Base.iterate(::UniqueNodeIterator{<:SpectralElementSpace2D}) =
    ((1, 1), 1), ((1, 1), 1)
function Base.iterate(
    iter::UniqueNodeIterator{<:SpectralElementSpace2D},
    ((i, j), e),
)
    space = iter.space
    Nq = Quadratures.degrees_of_freedom(quadrature_style(space))
    while true
        # find next node
        i += 1
        if i > Nq
            i = 1
            j += 1
        end
        if j > Nq
            j = 1
            e += 1
        end
        if e > Topologies.nlocalelems(space) # we're done
            return nothing
        end
        # check if this node has been seen
        # this assumes we don't have any shared vertices that are connected in a diagonal order,
        # e.g.
        #  1 | 3
        #  --+--
        #  4 | 2
        # we could check this by walking along the vertices as we go
        # this also doesn't deal with the case where eo == e
        if j == 1
            # face 1
            eo, _, _ = Topologies.opposing_face(Spaces.topology(space), e, 1)
            if 0 < eo < e
                continue
            end
        end
        if i == Nq
            # face 2
            eo, _, _ = Topologies.opposing_face(Spaces.topology(space), e, 2)
            if 0 < eo < e
                continue
            end
        end
        if j == Nq
            # face 3
            eo, _, _ = Topologies.opposing_face(Spaces.topology(space), e, 3)
            if 0 < eo < e
                continue
            end
        end
        if i == 1
            # face 4
            eo, _, _ = Topologies.opposing_face(Spaces.topology(space), e, 4)
            if 0 < eo < e
                continue
            end
        end
        return ((i, j), e), ((i, j), e)
    end
end

## aliases
const RectilinearSpectralElementSpace2D = SpectralElementSpace2D{
    <:Union{
        Grids.RectilinearSpectralElementGrid2D,
        Grids.LevelRectilinearSpectralElementGrid2D,
    },
}
const CubedSphereSpectralElementSpace2D = SpectralElementSpace2D{
    <:Union{
        Grids.CubedSphereSpectralElementGrid2D,
        Grids.LevelCubedSphereSpectralElementGrid2D,
    },
}
