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
    n = length(elemrange)-1
    z = x == elemrange[end] ? n : searchsortedlast(elemrange,x) # element index
    @assert 1 <= z <= n
    lo = elemrange[z]
    hi = elemrange[z+1]
    # Find ξ ∈ [-1,1] such that
    # x = (1-ξ)/2 * lo + (1+ξ)/2 * hi
    #   = (lo + hi) / 2 + ξ * (hi - lo) / 2
    ξ = (2x - (lo + hi)) / (hi - lo)
    return z, ξ
end


import UnicodePlots: heatmap
import ..Meshes

function interpolate(field::Field, r1, r2)
    fieldmesh = mesh(field)
    topology = fieldmesh.topology
    discretization = topology.discretization
    elemrange1 = discretization.range1
    elemrange2 = discretization.range2
    n1 = discretization.n1
    data = field_values(field)

    quadrature_style = fieldmesh.quadrature_style
    Nq = Meshes.Quadratures.degrees_of_freedom(quadrature_style)
    points, _ = Meshes.Quadratures.quadrature_points(Float64, quadrature_style)
    bw = Meshes.Quadratures.barycentric_weights(Float64, quadrature_style)

    FT = eltype(field)
    out = Array{FT}(undef, length(r1), length(r2))
    # assumes regular grid
    for (l2,x2) in enumerate(r2)
        z2, ξ2 = interpcoord(elemrange2, x2)

        for (l1,x1) in enumerate(r1)
            z1, ξ1 = interpcoord(elemrange1, x1)
            h = (z2-1) * n1 + z1
            data_slab = slab(data, h)
            # barycentric interpolation
            # Berrut2004 equation 4.2
            s = zero(FT)
            d = zero(FT)
            for j = 1:Nq, i = 1:Nq
                a1 = bw[i] / (ξ1 - points[i])
                a2 = bw[j] / (ξ2 - points[j])
                s += a1 * a2 * data_slab[i,j]
                d += a1 * a2
            end
            out[l1,l2] = s / d
        end
    end
    return out
end


end # module
