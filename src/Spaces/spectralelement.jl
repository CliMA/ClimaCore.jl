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
    grid = space.grid
    if hasfield(typeof(grid), :topology)
        # some reduced spaces (like slab space) do not have topology
        print(iio, " "^(indent + 2), "context: ")
        Topologies.print_context(iio, grid.topology.context)
        println(iio)
        println(iio, " "^(indent + 2), "mesh: ", grid.topology.mesh)
    end
    print(iio, " "^(indent + 2), "quadrature: ", grid.quadrature_style)
end



# 1D
struct SpectralElementSpace1D{G} <: AbstractSpectralElementSpace
    grid::G
end
space(grid::Grids.SpectralElementGrid1D, ::Nothing) =
    SpectralElementSpace1D(grid)
grid(space::Spaces.SpectralElementSpace1D) = space.grid

function SpectralElementSpace1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle,
)
    grid = Grids.SpectralElementGrid1D(topology, quadrature_style)
    SpectralElementSpace1D(grid)
end



# 2D
struct SpectralElementSpace2D{G} <: AbstractSpectralElementSpace
    grid::G
end
space(grid::Grids.SpectralElementGrid2D, ::Nothing) =
    SpectralElementSpace2D(grid)
grid(space::Spaces.SpectralElementSpace2D) = space.grid

function SpectralElementSpace2D(
    topology::Topologies.Topology2D,
    quadrature_style::Quadratures.QuadratureStyle,
)
    grid = Grids.SpectralElementGrid2D(topology, quadrature_style)
    SpectralElementSpace2D(grid)
end



const RectilinearSpectralElementSpace2D =
    SpectralElementSpace2D{<:Grids.RectilinearSpectralElementGrid2D}
const CubedSphereSpectralElementSpace2D =
    SpectralElementSpace2D{<:Grids.CubedSphereSpectralElementGrid2D}




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
    PointSpace(local_geometry)
end
Base.@propagate_inbounds column(space::SpectralElementSpace1D, i, j, h) =
    column(space, i, h)

Base.@propagate_inbounds function column(space::SpectralElementSpace2D, i, j, h)
    local_geometry = column(local_geometry_data(space), i, j, h)
    PointSpace(local_geometry)
end

function all_nodes(space::SpectralElementSpace2D)
    Nq = Quadratures.degrees_of_freedom(quadrature_style(space))
    nelem = Topologies.nlocalelems(topology(space))
    Iterators.product(Iterators.product(1:Nq, 1:Nq), 1:nelem)
end

"""
    unique_nodes(space::SpectralElementField2D)

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
