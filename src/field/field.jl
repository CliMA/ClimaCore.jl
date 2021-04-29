module Fields

import ..slab, ..column
import ..DataLayouts
import ..DataLayouts: AbstractData, DataStyle
import ..Meshes: AbstractMesh, Quadratures


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
field_values(field::Field) = getfield(field, :values)
mesh(field::Field) = getfield(field, :mesh)

Base.getproperty(field::Field, name::Symbol) =
    Field(getproperty(field_values(field), name), mesh(field))

Base.eltype(field::Field) = eltype(field_values(field))


# https://github.com/gridap/Gridap.jl/blob/master/src/Fields/DiffOperators.jl#L5
# https://github.com/gridap/Gridap.jl/blob/master/src/Fields/FieldsInterfaces.jl#L70

# TODO: nice printing
# follow x-array like printing?
# repl: #https://earth-env-data-science.github.io/lectures/xarray/xarray.html
# html: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html

# TODO: broadcasting

struct FieldStyle{DS <: DataStyle} <: Base.BroadcastStyle
    datastyle::DS
end
Base.Broadcast.BroadcastStyle(::Type{Field{V, M}}) where {V, M} =
    FieldStyle(DataStyle(V))
Base.Broadcast.BroadcastStyle(
    a::Base.Broadcast.AbstractArrayStyle{0},
    b::FieldStyle,
) = b

Base.Broadcast.broadcastable(field::Field{V, M}) where {V, M} = field

Base.axes(field::Field) = (mesh(field),)

todata(obj) = obj
todata(field::Field) = field_values(field)
function todata(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    Base.Broadcast.Broadcasted{DS}(bc.f, map(todata, bc.args))
end

function mesh(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    if bc.axes isa Nothing
        error("Call instantiate to access mesh of Broadcasted")
    end
    return bc.axes[1]
end

function Base.similar(
    bc::Base.Broadcast.Broadcasted{FieldStyle{DS}},
    ::Type{Eltype},
) where {DS, Eltype}
    return Field(similar(todata(bc), Eltype), mesh(bc))
end

function Base.copyto!(
    dest::Field{V, M},
    bc::Base.Broadcast.Broadcasted{FieldStyle{DS}},
) where {V, M, DS}
    copyto!(field_values(dest), todata(bc))
    return dest
end


"""
    coordinate_field(mesh::AbstractMesh)

Construct a `Field` of the coordinates of the mesh.
"""
coordinate_field(mesh::AbstractMesh) = Field(mesh.coordinates, mesh)
coordinate_field(field::Field) = coordinates(mesh(field))



function interpcoord(elemrange, x::Real)
    n = length(elemrange) - 1
    z = x == elemrange[end] ? n : searchsortedlast(elemrange, x) # element index
    @assert 1 <= z <= n
    lo = elemrange[z]
    hi = elemrange[z + 1]
    # Find ξ ∈ [-1,1] such that
    # x = (1-ξ)/2 * lo + (1+ξ)/2 * hi
    #   = (lo + hi) / 2 + ξ * (hi - lo) / 2
    ξ = (2x - (lo + hi)) / (hi - lo)
    return z, ξ
end

import ..Operators

import ..Meshes

function matrix_interpolate(
    field::Field,
    Q_interp::Quadratures.Uniform{Nu},
) where {Nu}
    S = eltype(field)
    fieldmesh = mesh(field)
    discretization = fieldmesh.topology.discretization
    n1 = discretization.n1
    n2 = discretization.n2

    interp_data =
        DataLayouts.IH1JH2{S, Nu}(Matrix{S}(undef, (Nu * n1, Nu * n2)))

    M = Quadratures.interpolation_matrix(
        Float64,
        Q_interp,
        fieldmesh.quadrature_style,
    )
    Operators.tensor_product!(interp_data, field_values(field), M)
    return parent(interp_data)
end
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())

include("plots.jl")


end # module
