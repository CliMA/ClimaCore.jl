module Fields

import ..slab, ..column
import ..DataLayouts: AbstractData, DataStyle
import ..Meshes: AbstractMesh


"""
    Field(values, mesh)

A set of `values` defined at each point of a `mesh`.
"""
struct Field{V <: AbstractData, M <: AbstractMesh}
    values::V
    mesh::M
    # add metadata/attributes?
    function Field{V, M}(values::V, mesh::M) where {V, M}
        # need to enforce that the data size matches the mesh
        # @assert support(values) === support(mesh.coordinates)
        @assert size(values) == size(mesh.coordinates)
        return new{V, M}(values, mesh)
    end
end
Field(values::V, mesh::M) where {V <: AbstractData, M <: AbstractMesh} =
    Field{V, M}(values, mesh)

function Field(array::AbstractArray{T,6}, mesh::AbstractMesh) where {T} 
    @assert size(array, 1) == size(array, 2)
    Nij, Nk = size(array, 1), size(array, 3)
    S = eltype(array)
    return Field(DataLayouts.IJKFVH{S, Nij, Nk}(array), mesh)
end

function Field(array::AbstractArray{T,4}, mesh::AbstractMesh) where {T}
    @assert size(array, 1) == size(array, 2)
    Nij = size(array, 1)
    S = eltype(array)
    return Field(DataLayouts.IJFH{S, Nij}(array), mesh)
end

"""
    field_values(field::Field) -> AbstractData

Return a reference to the `AbstractData` data object of a `field`
"""
field_values(field::Field) = getfield(field, :values)

"""
    field_values(field::Field) -> AbstractMesh

Return a reference to the `AbstractMesh` mesh object of a `field`
"""
field_mesh(field::Field) = getfield(field, :mesh)

"""
   field_like(field::Field, values) -> Field

Return a `Field` object with `values` on the same mesh structure as `field`
"""
function field_like(field::Field, values)
    return Field(values, field_mesh(field))
end

# Accessor methods
Base.propertynames(field::Field) = getfield(field, :values)

Base.getproperty(field::Field, name::Symbol) =
    Field(getproperty(field_values(field), name), field_mesh(field))

Base.eltype(field::Field) = eltype(field_values(field))

"""
    coordinate_field(mesh::AbstractMesh) -> Field

Construct a `Field` of the coordinates of the `mesh`.
"""
coordinate_field(mesh::AbstractMesh) = Field(mesh.coordinates, mesh)

"""
    coordinate_field(field::Field) -> Field

Construct a `Field` of the coordinates of the `field` mesh.
"""
coordinate_field(field::Field) = coordinates(field_mesh(field))

# Broadcasting
include("broadcast.jl")

end # module
