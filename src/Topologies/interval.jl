"""
    AbstractIntervalTopology <: AbstractTopology

A topology on a 1D interval domain.
"""
abstract type AbstractIntervalTopology <: AbstractTopology end


# at the moment, this only serves the purpose of putting the boundary tags into type space.
"""
    IntervalTopology(mesh::IntervalMesh)

A topology on a [`Meshes.IntervalMesh`](@ref), with the elements stored sequentially
"""
struct IntervalTopology{M <: Meshes.IntervalMesh, B} <: AbstractIntervalTopology
    mesh::M
    boundaries::B
end

function IntervalTopology(mesh::Meshes.IntervalMesh)
    if isnothing(mesh.domain.boundary_tags)
        boundaries = NamedTuple()
    else
        boundaries = NamedTuple{mesh.domain.boundary_tags}((1, 2))
    end
    IntervalTopology(mesh, boundaries)
end

function Base.show(io::IO, topology::IntervalTopology)
    print(io, "IntervalTopology on ", topology.mesh)
end

boundaries(topology::IntervalTopology) = topology.boundaries
domain(topology::IntervalTopology) = topology.mesh.domain
nlocalelems(topology::IntervalTopology) = length(topology.mesh.faces) - 1

vertex_coordinates(topology::IntervalTopology, elem) =
    (topology.mesh.faces[elem], topology.mesh.faces[elem + 1])

function opposing_face(topology::IntervalTopology, elem, face)
    n = length(topology.mesh.faces) - 1
    if face == 1
        if elem == 1
            if isempty(mesh.boundaries) # periodic
                opelem = n
            else
                return (0, 1, false)
            end
        else
            opelem = elem - 1
        end
        opface = 2
    else
        if elem == n
            if isempty(mesh.boundaries) # periodic
                opelem = 1
            else
                return (0, 2, false)
            end
        end
        opface = 1
    end
    return (opelem, opface, false)
end

function Base.length(fiter::InteriorFaceIterator{<:IntervalTopology})
    topology = fiter.topology
    if isempty(topology.boundaries)
        length(topology.mesh.faces) - 1
    else
        length(topology.mesh.faces) - 2
    end
end

function Base.iterate(fiter::InteriorFaceIterator{<:IntervalTopology}, i = 1)
    topology = fiter.topology
    periodic = isempty(topology.boundaries)
    n = length(topology.mesh.faces) - 1
    if i < n
        return (i + 1, 1, i, 2, false), i + 1
    elseif i == n && periodic
        return (1, 1, i, 2, false), i + 1
    else
        return nothing
    end
end
