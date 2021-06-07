module Fields

import ..slab, ..column
import ..DataLayouts: DataLayouts, AbstractData, DataStyle
import ..Spaces: Spaces, AbstractSpace
import ..Geometry: Geometry, Cartesian12Vector
import ..RecursiveApply
import ..Topologies

import LinearAlgebra

function __init__()
    init_diffeq()
end

"""
    Field(values, space)

A set of `values` defined at each point of a `space`.
"""
struct Field{V <: AbstractData, S <: AbstractSpace}
    values::V
    space::S
    # add metadata/attributes?
    function Field{V, S}(values::V, space::S) where {V, S}
        # need to enforce that the data size matches the space
        # @assert support(values) === support(space.coordinates)
        # @assert size(values) == size(space.coordinates)
        return new{V, S}(values, space)
    end
end
Field(values::V, space::S) where {V <: AbstractData, S <: AbstractSpace} =
    Field{V, S}(values, space)

const SpectralElementField2D{V} = Field{V, <:Spaces.SpectralElementSpace2D}
const FiniteDifferenceField{V} = Field{V, <:Spaces.FiniteDifferenceSpace}


Base.propertynames(field::Field) = propertynames(getfield(field, :values))
field_values(field::Field) = getfield(field, :values)
space(field::Field) = getfield(field, :space)

# need to define twice to avoid ambiguities
Base.getproperty(field::Field, name::Symbol) =
    Field(getproperty(field_values(field), name), space(field))
Base.getproperty(field::Field, name::Integer) =
    Field(getproperty(field_values(field), name), space(field))

Base.eltype(field::Field) = eltype(field_values(field))
Base.parent(field::Field) = parent(field_values(field))

# to play nice with DifferentialEquations; may want to revisit this
# https://github.com/SciML/SciMLBase.jl/blob/697bd0c0c7365e77fa311f2d32eade70f43a8d50/src/solutions/ode_solutions.jl#L31
Base.size(field::Field) = ()
Base.length(field::Fields.Field) = 1


function slab(field::Field, h)
    Field(slab(field_values(field), h), slab(space(field), h))
end


Topologies.nlocalelems(field::Field) = Topologies.nlocalelems(space(field))



# nice printing
# follow x-array like printing?
# repl: #https://earth-env-data-science.github.io/lectures/xarray/xarray.html
# html: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html
function Base.show(io::IO, field::Field)
    print(io, eltype(field), "-valued Field:")
    _show_compact_field(io, field, "  ", true)
    # print(io, "\non ", space(field)) # TODO: write a better space print
end
function _show_compact_field(io, field, prefix, isfirst = false)
    #print(io, prefix1)
    if eltype(field) <: Number
        if isfirst
            print(io, "\n", prefix)
        end
        print(
            IOContext(io, :compact => true, :limit => true),
            vec(parent(field)),
        )
    else
        names = propertynames(field)
        for name in names
            print(io, "\n", prefix)
            print(io, name, ": ")
            _show_compact_field(io, getproperty(field, name), prefix * "  ")
        end
    end
end


# https://github.com/gridap/Gridap.jl/blob/master/src/Fields/DiffOperators.jl#L5
# https://github.com/gridap/Gridap.jl/blob/master/src/Fields/FieldsInterfaces.jl#L70


Base.similar(field::Field, ::Type{Eltype}) where {Eltype} =
    Field(similar(field_values(field), Eltype), space(field))
Base.similar(field::Field) = similar(field, eltype(field))
Base.similar(field::F, ::Type{F}) where {F <: Field} = similar(field)


# fields on different spaces
function Base.similar(field::Field, (space_to,)::Tuple{AbstractSpace})
    similar(field, (space_to,), eltype(field))
end
function Base.similar(
    field::Field,
    (space_to,)::Tuple{AbstractSpace},
    ::Type{Eltype},
) where {Eltype}
    Field(similar(space_to.coordinates, Eltype), space_to)
end

Base.copy(field::Field) = Field(copy(field_values(field)), space(field))

function Base.copyto!(dest::Field{V, M}, src::Field{V, M}) where {V, M}
    @assert space(dest) == space(src)
    copyto!(field_values(dest), field_values(src))
    return dest
end

function Base.zeros(::Type{FT}, space::AbstractSpace) where {FT}
    field = Field(similar(Spaces.coordinates(space), FT), space)
    data = parent(field)
    fill!(data, zero(eltype(data)))
    return field
end

function Base.ones(::Type{FT}, space::AbstractSpace) where {FT}
    field = Field(similar(Spaces.coordinates(space), FT), space)
    data = parent(field)
    fill!(data, one(eltype(data)))
    return field
end

function Base.zero(field::Field)
    zfield = similar(field)
    zarray = parent(zfield)
    fill!(zarray, zero(eltype(zarray)))
    return zfield
end


"""
    coordinate_field(space::AbstractSpace)

Construct a `Field` of the coordinates of the space.
"""
coordinate_field(space::AbstractSpace) = Field(Spaces.coordinates(space), space)
coordinate_field(field::Field) = coordinate_field(space(field))

include("broadcast.jl")
include("mapreduce.jl")
include("compat_diffeq.jl")

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

"""
    Spaces.variational_solve!(field)

Divide `field` by the mass matrix.
"""
function Spaces.variational_solve!(field::Field)
    Spaces.variational_solve!(field_values(field), space(field))
    return field
end

function Spaces.horizontal_dss!(field::Field)
    Spaces.horizontal_dss!(field_values(field), space(field))
    return field
end

end # module
