module Fields

import ..slab, ..column
import ..DataLayouts: AbstractData
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

Base.propertynames(field::Field) = getfield(field, :values)
values(field::Field) = getfield(field, :values)
mesh(field::Field) = getfield(field, :mesh)

Base.getproperty(field::Field, name::Symbol) =
    Field(getproperty(values(field), name), mesh(field))

# TODO: nice printing
# follow x-array like printing?
# repl: #https://earth-env-data-science.github.io/lectures/xarray/xarray.html
# html: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html

# TODO: broadcasting
end # module
