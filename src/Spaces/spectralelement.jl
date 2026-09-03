abstract type AbstractSpectralElementSpace <: AbstractSpace end

Topologies.nlocalelems(space::AbstractSpectralElementSpace) =
    Topologies.nlocalelems(topology(space))



quadrature_style(space::AbstractSpectralElementSpace) =
    quadrature_style(grid(space))

horizontal_space(space::AbstractSpectralElementSpace) = space
nlevels(space::AbstractSpectralElementSpace) = 1

eachslabindex(space::AbstractSpectralElementSpace) =
    1:Topologies.nlocalelems(topology(space))

staggering(space::AbstractSpectralElementSpace) = nothing

function Base.show(io::IO, space::AbstractSpectralElementSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, nameof(typeof(space)), ":")
    if get_mask(space) isa DataLayouts.NoMask
        println(iio, " "^(indent + 2), "mask_enabled: false")
    else
        println(iio, " "^(indent + 2), "mask_enabled: true")
    end
    if hasfield(typeof(grid(space)), :topology)
        # some reduced spaces (like slab space) do not have topology
        print(iio, " "^(indent + 2), "context: ")
        Topologies.print_context(iio, topology(grid(space)).context)
        println(iio)
        println(
            iio,
            " "^(indent + 2),
            "mesh: ",
            topology(grid(space)).mesh,
        )
    end
    print(
        iio,
        " "^(indent + 2),
        "quadrature: ",
        quadrature_style(grid(space)),
    )
end



# 1D
"""
    SpectralElementSpace1D(grid::Grids.SpectralElementGrid1D)
    SpectralElementSpace1D(
        topology::Topologies.IntervalTopology,
        quadrature_style::Quadratures.QuadratureStyle;
        discretization = nothing,
        kwargs...
    )

A one-dimensional spectral-element space. The second form builds the
[`Grids.SpectralElementGrid1D`](@ref) and forwards `kwargs` to it;
`discretization = Grids.DG()` makes the space discontinuous across element
boundaries, and omitting it follows the quadrature. Read the choice back with
[`Spaces.discretization`](@ref).
"""
struct SpectralElementSpace1D{G} <: AbstractSpectralElementSpace
    grid::G
end
space(grid::Grids.SpectralElementGrid1D, ::Nothing) =
    SpectralElementSpace1D(grid)
space(grid::Grids.LevelGrid{<:Grids.ExtrudedSpectralElementGrid2D}, ::Nothing) =
    SpectralElementSpace1D(grid)
grid(space::SpectralElementSpace1D) = getfield(space, :grid)

local_geometry_type(::Type{SpectralElementSpace1D{G}}) where {G} =
    local_geometry_type(G)

Adapt.adapt_structure(to, space::SpectralElementSpace1D) =
    SpectralElementSpace1D(Adapt.adapt(to, grid(space)))

function SpectralElementSpace1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle;
    kwargs...,
)
    grid = Grids.SpectralElementGrid1D(topology, quadrature_style; kwargs...)
    SpectralElementSpace1D(grid)
end

# 2D
"""
    SpectralElementSpace2D(grid::Grids.SpectralElementGrid2D)
    SpectralElementSpace2D(
        topology::Topologies.Topology2D,
        quadrature_style::Quadratures.QuadratureStyle;
        discretization = nothing,
        kwargs...,
    )

A two-dimensional spectral-element space. The second form builds the
[`Grids.SpectralElementGrid2D`](@ref) and forwards `kwargs` to it
(`discretization`, `enable_bubble`, `enable_mask`, `autodiff_metric`, `VIJH`);
`discretization = Grids.DG()` makes the space discontinuous across element
boundaries, so [`Spaces.weighted_dss!`](@ref) is a no-op and inter-element
coupling goes through numerical fluxes. Omitting it follows the quadrature.
Read the choice back with [`Spaces.discretization`](@ref).

# Examples

```julia
space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())
dg_space = Spaces.SpectralElementSpace2D(
    topology,
    Quadratures.GLL{4}();
    discretization = Grids.DG(),
)
```
"""
struct SpectralElementSpace2D{G} <: AbstractSpectralElementSpace
    grid::G
end
space(grid::Grids.SpectralElementGrid2D, ::Nothing) =
    SpectralElementSpace2D(grid)
space(grid::Grids.LevelGrid{<:Grids.ExtrudedSpectralElementGrid3D}, ::Nothing) =
    SpectralElementSpace2D(grid)

local_geometry_type(::Type{SpectralElementSpace2D{G}}) where {G} =
    local_geometry_type(G)

grid(space::SpectralElementSpace2D) = getfield(space, :grid)

function SpectralElementSpace2D(
    topology::Topologies.Topology2D,
    quadrature_style::Quadratures.QuadratureStyle;
    kwargs...,
)
    grid = Grids.SpectralElementGrid2D(topology, quadrature_style; kwargs...)
    SpectralElementSpace2D(grid)
end

Adapt.adapt_structure(to, space::SpectralElementSpace2D) =
    SpectralElementSpace2D(Adapt.adapt(to, grid(space)))

"""
    SpectralElementSpaceSlab <: AbstractSpace

A view into a `SpectralElementSpace2D` for a single slab.
"""
struct SpectralElementSpaceSlab{C, Q, G} <: AbstractSpectralElementSpace
    context::C
    quadrature_style::Q
    local_geometry::G
end

local_geometry_type(::Type{SpectralElementSpaceSlab{<:Any, <:Any, G}}) where {G} =
    eltype(G) # calls eltype from DataLayouts

ClimaComms.device(space::SpectralElementSpaceSlab) = ClimaComms.device(space.context)
ClimaComms.context(space::SpectralElementSpaceSlab) = space.context

quadrature_style(space::SpectralElementSpaceSlab) = space.quadrature_style
local_geometry_data(space::SpectralElementSpaceSlab) = space.local_geometry

issubspace(space1::SpectralElementSpaceSlab, space2::SpectralElementSpaceSlab) =
    space1 == space2
issubspace(space1::AbstractSpectralElementSpace, space2::AbstractSpectralElementSpace) =
    horizontal_grid(grid(space1)) === horizontal_grid(grid(space2))

level(space::AbstractSpectralElementSpace, v) =
    isone(v) ? space : throw(ArgumentError("Space only has one level"))

Base.@propagate_inbounds slab(space::AbstractSpectralElementSpace, v, h) =
    isone(v) ? slab(space, h) : throw(ArgumentError("Space has only one level"))
Base.@propagate_inbounds slab(space::AbstractSpectralElementSpace, h) =
    SpectralElementSpaceSlab(
        ClimaComms.context(space),
        quadrature_style(space),
        slab(local_geometry_data(space), h),
    )

Base.@propagate_inbounds column(space::AbstractSpectralElementSpace, indices...) =
    PointSpace(ClimaComms.context(space), column(local_geometry_data(space), indices...))

"""
    Spaces.node_horizontal_length_scale(space::AbstractSpectralElementSpace)

The approximate length scale of the distance between nodes. This is defined as the
length scale of the mesh (see [`Meshes.element_horizontal_length_scale`](@ref)), divided by the
number of unique quadrature points along each dimension.

Returns a default length scale of 1 when no space is provided.
"""
function node_horizontal_length_scale(space::AbstractSpectralElementSpace)
    quad = quadrature_style(space)
    Nu = Quadratures.unique_degrees_of_freedom(quad)
    return Meshes.element_horizontal_length_scale(topology(space).mesh) /
           Nu
end

node_horizontal_length_scale(::Nothing) = 1

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
    space_topology = topology(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style(space))

    nelem = Topologies.nlocalelems(space_topology)
    nvert = length(Topologies.local_vertices(space_topology))
    nface_interior = length(Topologies.interior_faces(space_topology))
    if isempty(Topologies.boundary_tags(space_topology))
        nface_boundary = 0
    else
        nface_boundary = sum(Topologies.boundary_tags(space_topology)) do tag
            length(Topologies.boundary_faces(space_topology, tag))
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
            eo, _, _ = Topologies.opposing_face(topology(space), e, 1)
            if 0 < eo < e
                continue
            end
        end
        if i == Nq
            # face 2
            eo, _, _ = Topologies.opposing_face(topology(space), e, 2)
            if 0 < eo < e
                continue
            end
        end
        if j == Nq
            # face 3
            eo, _, _ = Topologies.opposing_face(topology(space), e, 3)
            if 0 < eo < e
                continue
            end
        end
        if i == 1
            # face 4
            eo, _, _ = Topologies.opposing_face(topology(space), e, 4)
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
