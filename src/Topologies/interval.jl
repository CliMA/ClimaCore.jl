# at the moment, this only serves the purpose of putting the boundary tags into type space.
"""
    IntervalTopology(mesh::IntervalMesh)

A sequential topology on an [`IntervalMesh`](@ref).
"""
struct IntervalTopology{M <: Meshes.IntervalMesh, B} <: AbstractTopology
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
