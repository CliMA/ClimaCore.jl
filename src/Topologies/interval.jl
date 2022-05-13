abstract type AbstractIntervalTopology <: AbstractTopology end

"""
    IntervalTopology(mesh::IntervalMesh)

A sequential topology on an [`Meshes.IntervalMesh`](@ref).
"""
struct IntervalTopology{M <: Meshes.IntervalMesh, B} <: AbstractIntervalTopology
    mesh::M
    boundaries::B
end

function IntervalTopology(mesh::Meshes.IntervalMesh)
    if Domains.isperiodic(mesh.domain)
        boundaries = NamedTuple()
    elseif mesh.domain.boundary_names[1] == mesh.domain.boundary_names[2]
        boundaries = NamedTuple{(mesh.domain.boundary_names[1],)}(1)
    else
        boundaries = NamedTuple{mesh.domain.boundary_names}((1, 2))
    end
    IntervalTopology(mesh, boundaries)
end

isperiodic(topology::AbstractIntervalTopology) =
    Domains.isperiodic(Topologies.domain(topology))

function Base.show(io::IO, topology::AbstractIntervalTopology)
    print(io, "IntervalTopology on ", Topologies.mesh(topology))
end

function mesh(topology::AbstractIntervalTopology)
    getfield(topology, :mesh)
end

function boundaries(topology::AbstractIntervalTopology)
    getfield(topology, :boundaries)
end

function domain(topology::AbstractIntervalTopology)
    mesh = Topologies.mesh(topology)
    Meshes.domain(mesh)
end

function nlocalelems(topology::AbstractIntervalTopology)
    mesh = Topologies.mesh(topology)
    length(mesh.faces) - 1
end

function vertex_coordinates(topology::AbstractIntervalTopology, elem)
    mesh = Topologies.mesh(topology)
    (mesh.faces[elem], mesh.faces[elem + 1])
end

function opposing_face(topology::AbstractIntervalTopology, elem, face)
    mesh = Topologies.mesh(topology)
    n = length(mesh.faces) - 1
    if face == 1
        if elem == 1
            if isperiodic(topology)
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
            if isperiodic(topology)
                opelem = 1
            else
                return (0, 2, false)
            end
        end
        opface = 1
    end
    return (opelem, opface, false)
end

function Base.length(fiter::InteriorFaceIterator{<:AbstractIntervalTopology})
    topology = fiter.topology
    mesh = Topologies.mesh(topology)
    periodic = isempty(topology.boundaries)
    if periodic
        length(mesh.faces) - 1
    else
        length(mesh.faces) - 2
    end
end

function Base.iterate(
    fiter::InteriorFaceIterator{<:AbstractIntervalTopology},
    i = 1,
)
    topology = fiter.topology
    mesh = Topologies.mesh(topology)
    periodic = isempty(Topologies.boundaries(topology))
    n = length(mesh.faces) - 1
    if i < n
        return (i + 1, 1, i, 2, false), i + 1
    elseif i == n && periodic
        return (1, 1, i, 2, false), i + 1
    else
        return nothing
    end
end

function local_neighboring_elements(topology::AbstractIntervalTopology, elem)
    (opelem_1, _, _) = opposing_face(topology, elem, 1)
    (opelem_2, _, _) = opposing_face(topology, elem, 2)
    if opelem_1 == 0
        if opelem_2 == 0
            return ()
        else
            return (opelem_2,)
        end
    else
        if opelem_2 == 0
            return (opelem_1,)
        else
            return (opelem_1, opelem_2)
        end
    end
end
ghost_neighboring_elements(topology::AbstractIntervalTopology, elem) = ()
