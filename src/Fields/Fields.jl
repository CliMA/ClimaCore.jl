module Fields

import ClimaComms
import ..slab, ..slab_args, ..column, ..column_args, ..level
import ..DataLayouts: DataLayouts, AbstractData, DataStyle
import ..Domains
import ..Topologies
import ..Spaces: Spaces, AbstractSpace, AbstractPointSpace
import ..Geometry: Geometry, Cartesian12Vector
import ..Utilities: PlusHalf, half

using ..RecursiveApply
using CUDA
using ClimaComms
import Adapt

import StaticArrays, LinearAlgebra, Statistics, InteractiveUtils

"""
    Field(values, space)

A set of `values` defined at each point of a `space`.
"""
struct Field{V <: AbstractData, S <: AbstractSpace}
    values::V
    space::S
    # add metadata/attributes?
    function Field{V, S}(values::V, space::S) where {V, S}
        #TODOneed to enforce that the data size matches the space
        return new{V, S}(values, space)
    end
end
Field(values::V, space::S) where {V <: AbstractData, S <: AbstractSpace} =
    Field{V, S}(values, space)

Field(::Type{T}, space::S) where {T, S <: AbstractSpace} =
    Field(similar(Spaces.coordinates_data(space), T), space)

ClimaComms.context(field::Field) = ClimaComms.context(axes(field))

ClimaComms.context(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    ClimaComms.context(Spaces.horizontal_space(space))
ClimaComms.context(space::Spaces.SpectralElementSpace2D) =
    ClimaComms.context(space.topology)
ClimaComms.context(space::S) where {S <: Spaces.AbstractSpace} =
    ClimaComms.context(space.topology)

ClimaComms.context(topology::Topologies.Topology2D) = topology.context
ClimaComms.context(topology::T) where {T <: Topologies.AbstractTopology} =
    topology.context

Adapt.adapt_structure(to, field::Field) = Field(
    Adapt.adapt(to, Fields.field_values(field)),
    Adapt.adapt(to, axes(field)),
)


# Point Field
const PointField{V, S} =
    Field{V, S} where {V <: AbstractData, S <: Spaces.PointSpace}

# Spectral Element Field
const SpectralElementField{V, S} = Field{
    V,
    S,
} where {V <: AbstractData, S <: Spaces.AbstractSpectralElementSpace}
const SpectralElementField1D{V, S} =
    Field{V, S} where {V <: AbstractData, S <: Spaces.SpectralElementSpace1D}
const SpectralElementField2D{V, S} =
    Field{V, S} where {V <: AbstractData, S <: Spaces.SpectralElementSpace2D}

const FiniteDifferenceField{V, S} =
    Field{V, S} where {V <: AbstractData, S <: Spaces.FiniteDifferenceSpace}
const FaceFiniteDifferenceField{V, S} =
    Field{V, S} where {V <: AbstractData, S <: Spaces.FaceFiniteDifferenceSpace}
const CenterFiniteDifferenceField{V, S} = Field{
    V,
    S,
} where {V <: AbstractData, S <: Spaces.CenterFiniteDifferenceSpace}

# Extruded Fields
const ExtrudedFiniteDifferenceField{V, S} = Field{
    V,
    S,
} where {V <: AbstractData, S <: Spaces.ExtrudedFiniteDifferenceSpace}
const FaceExtrudedFiniteDifferenceField{V, S} = Field{
    V,
    S,
} where {V <: AbstractData, S <: Spaces.FaceExtrudedFiniteDifferenceSpace}
const CenterExtrudedFiniteDifferenceField{V, S} = Field{
    V,
    S,
} where {V <: AbstractData, S <: Spaces.CenterExtrudedFiniteDifferenceSpace}

# Cubed Sphere Fields
const CubedSphereSpectralElementField2D{V, S} = Field{
    V,
    S,
} where {V <: AbstractData, S <: Spaces.CubedSphereSpectralElementSpace2D}


Base.propertynames(field::Field) = propertynames(getfield(field, :values))
@inline field_values(field::Field) = getfield(field, :values)

# Define the axes field to be the todata(bc) of the return field
@inline Base.axes(field::Field) = getfield(field, :space)

# Define device and device array type
ClimaComms.device(field::Field) = ClimaComms.device(axes(field))
ClimaComms.array_type(field::Field) =
    ClimaComms.array_type(ClimaComms.device(field))

# need to define twice to avoid ambiguities
@inline Base.dotgetproperty(field::Field, prop) = Base.getproperty(field, prop)

@inline Base.getproperty(field::Field, name::Symbol) = Field(
    DataLayouts._getproperty(field_values(field), Val{name}()),
    axes(field),
)

@inline Base.getproperty(field::Field, name::Integer) =
    Field(getproperty(field_values(field), name), axes(field))

Base.eltype(::Type{<:Field{V}}) where {V} = eltype(V)
Base.parent(field::Field) = parent(field_values(field))

# to play nice with DifferentialEquations; may want to revisit this
# https://github.com/SciML/SciMLBase.jl/blob/697bd0c0c7365e77fa311f2d32eade70f43a8d50/src/solutions/ode_solutions.jl#L31
Base.size(field::Field) = ()
Base.length(field::Fields.Field) = 1

Topologies.nlocalelems(field::Field) = Topologies.nlocalelems(axes(field))

# Methods for Slab and Column fields
const SlabField{V, S} =
    Field{V, S} where {V <: AbstractData, S <: Spaces.SpectralElementSpaceSlab}

const SlabField1D{V, S} = Field{
    V,
    S,
} where {
    V <: DataLayouts.DataSlab1D,
    S <: Spaces.SpectralElementSpaceSlab1D,
}

const SlabField2D{V, S} = Field{
    V,
    S,
} where {
    V <: DataLayouts.DataSlab2D,
    S <: Spaces.SpectralElementSpaceSlab2D,
}

const ColumnField{V, S} =
    Field{V, S} where {V <: DataLayouts.DataColumn, S <: Spaces.AbstractSpace}

Base.@propagate_inbounds slab(field::Field, inds...) =
    Field(slab(field_values(field), inds...), slab(axes(field), inds...))

Base.@propagate_inbounds function column(field::Field, inds...)
    Field(column(field_values(field), inds...), column(axes(field), inds...))
end
@inline column(field::FiniteDifferenceField, inds...) = field



# nice printing
# follow x-array like printing?
# repl: #https://earth-env-data-science.github.io/lectures/xarray/xarray.html
# html: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html
function Base.show(io::IO, field::Field)
    print(io, eltype(field), "-valued Field:")
    _show_compact_field(io, field, "  ", true)
    # print(io, "\non ", axes(field)) # TODO: write a better space print
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
            subfield = getproperty(field, name)
            if sizeof(eltype(subfield)) == 0
                continue
            end
            print(io, "\n", prefix)
            print(io, name, ": ")
            _show_compact_field(io, getproperty(field, name), prefix * "  ")
        end
    end
end


# https://github.com/gridap/Gridap.jl/blob/master/src/Fields/DiffOperators.jl#L5
# https://github.com/gridap/Gridap.jl/blob/master/src/Fields/FieldsInterfaces.jl#L70


Base.similar(field::Field) = Field(similar(field_values(field)), axes(field))
Base.similar(field::Field, ::Type{T}) where {T} =
    Field(similar(field_values(field), T), axes(field))

# fields on different spaces
function Base.similar(field::Field, space_to::AbstractSpace)
    similar(field, space_to, eltype(field))
end
function Base.similar(
    field::Field,
    space_to::AbstractSpace,
    ::Type{Eltype},
) where {Eltype}
    Field(Eltype, space_to)
end

Base.copy(field::Field) = Field(copy(field_values(field)), axes(field))

Base.deepcopy_internal(field::Field, stackdict::IdDict) =
    Field(Base.deepcopy_internal(field_values(field), stackdict), axes(field))

function Base.copyto!(dest::Field{V, M}, src::Field{V, M}) where {V, M}
    @assert axes(dest) == axes(src)
    copyto!(field_values(dest), field_values(src))
    return dest
end

"""
    fill!(field::Field, value)

Fill `field` with `value`.
"""
function Base.fill!(field::Field, value)
    fill!(field_values(field), value)
    return field
end
"""
    fill(value, space::AbstractSpace)

Create a new `Field` on `space` and fill it with `value`.
"""
function Base.fill(value::FT, space::AbstractSpace) where {FT}
    field = Field(FT, space)
    return fill!(field, value)
end

"""
    zeros(space::AbstractSpace)

Construct a field on `space` that is zero everywhere.
"""
function Base.zeros(::Type{FT}, space::AbstractSpace) where {FT}
    field = Field(FT, space)
    data = parent(field)
    fill!(data, zero(eltype(data)))
    return field
end
Base.zeros(space::AbstractSpace) = zeros(Spaces.undertype(space), space)

"""
    ones(space::AbstractSpace)

Construct a field on `space` that is one everywhere.
"""
function Base.ones(::Type{FT}, space::AbstractSpace) where {FT}
    field = Field(FT, space)
    data = parent(field)
    fill!(data, one(eltype(data)))
    return field
end
Base.ones(space::AbstractSpace) = ones(Spaces.undertype(space), space)

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
coordinate_field(space::AbstractSpace) =
    Field(Spaces.coordinates_data(space), space)
coordinate_field(field::Field) = coordinate_field(axes(field))

"""
    local_geometry_field(space::AbstractSpace)

Construct a `Field` of the `LocalGeometry` of the space.
"""
local_geometry_field(space::AbstractSpace) =
    Field(Spaces.local_geometry_data(space), space)
local_geometry_field(field::Field) = local_geometry_field(axes(field))

Base.@deprecate dz_field(space::AbstractSpace) Δz_field(space) false
Base.@deprecate dz_field(field::Field) Δz_field(field) false

"""
    Δz_field(field::Field)
    Δz_field(space::AbstractSpace)

A `Field` containing the `Δz` values on the same space as the given field.
"""
Δz_field(field::Field) = Δz_field(axes(field))
Δz_field(space::AbstractSpace) = Field(Spaces.Δz_data(space), space)

include("broadcast.jl")
include("mapreduce_cuda.jl")
include("mapreduce.jl")
include("compat_diffeq.jl")
include("fieldvector.jl")
include("field_iterator.jl")
include("indices.jl")

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
    Spaces.variational_solve!(field_values(field), axes(field))
    return field
end

"""
    Spaces.weighted_dss!(f::Field[, ghost_buffer = Spaces.create_dss_buffer(field)])

Apply weighted direct stiffness summation (DSS) to `f`. This operates in-place
(i.e. it modifies the `f`). `ghost_buffer` contains the necessary information
for communication in a distributed setting, see [`Spaces.create_ghost_buffer`](@ref).

This is a projection operation from the piecewise polynomial space
``\\mathcal{V}_0`` to the continuous space ``\\mathcal{V}_1 = \\mathcal{V}_0
\\cap \\mathcal{C}_0``, defined as the field ``\\theta \\in \\mathcal{V}_1``
such that for all ``\\phi \\in \\mathcal{V}_1``
```math
\\int_\\Omega \\phi \\theta \\,d\\Omega = \\int_\\Omega \\phi f \\,d\\Omega
```

In matrix form, we define ``\\bar \\theta`` to be the unique global node
representation, and ``Q`` to be the "scatter" operator which maps to the
redundant node representation ``\\theta``
```math
\\theta = Q \\bar \\theta
```
Then the problem can be written as
```math
(Q \\bar\\phi)^\\top W J Q \\bar\\theta = (Q \\bar\\phi)^\\top W J f
```
which reduces to
```math
\\theta = Q \\bar\\theta = Q (Q^\\top W J Q)^{-1} Q^\\top W J f
```
"""
function Spaces.weighted_dss!(
    field::Field,
    dss_buffer = Spaces.create_dss_buffer(field),
)
    Spaces.weighted_dss!(field_values(field), axes(field), dss_buffer)
    return field
end
Spaces.weighted_dss_start!(field::Field, dss_buffer) =
    Spaces.weighted_dss_start!(field_values(field), axes(field), dss_buffer)

Spaces.weighted_dss_internal!(field::Field, dss_buffer) =
    Spaces.weighted_dss_internal!(field_values(field), axes(field), dss_buffer)

Spaces.weighted_dss_ghost!(field::Field, dss_buffer) =
    Spaces.weighted_dss_ghost!(field_values(field), axes(field), dss_buffer)

"""
    Spaces.create_ghost_buffer(field::Field)

Create a buffer for communicating neighbour information of `field`.
"""
Spaces.create_ghost_buffer(field::Field) = Spaces.create_dss_buffer(field)

"""
    Spaces.create_dss_buffer(field::Field)

Create a buffer for communicating neighbour information of `field`.
"""
function Spaces.create_dss_buffer(field::Field)
    space = axes(field)
    hspace = Spaces.horizontal_space(space)
    Spaces.create_dss_buffer(field_values(field), hspace)
end
# Add definitions for backward compatibility
Spaces.weighted_dss2!(
    field::Field,
    dss_buffer = Spaces.create_dss_buffer(field),
) = Spaces.weighted_dss!(field, dss_buffer)

Spaces.weighted_dss_start2!(field::Field, ghost_buffer) =
    Spaces.weighted_dss_start!(field, ghost_buffer)

Spaces.weighted_dss_internal2!(field::Field, ghost_buffer) =
    Spaces.weighted_dss_internal!(field, ghost_buffer)

Spaces.weighted_dss_ghost2!(field, ghost_buffer) =
    Spaces.weighted_dss_ghost!(field, ghost_buffer)

Base.@propagate_inbounds function level(
    field::Union{
        CenterFiniteDifferenceField,
        CenterExtrudedFiniteDifferenceField,
    },
    v::Int,
)
    hspace = level(axes(field), v)
    data = level(field_values(field), v)
    Field(data, hspace)
end
Base.@propagate_inbounds function level(
    field::Union{FaceFiniteDifferenceField, FaceExtrudedFiniteDifferenceField},
    v::PlusHalf,
)
    hspace = level(axes(field), v)
    @inbounds data = level(field_values(field), v.i + 1)
    Field(data, hspace)
end

Base.getindex(field::Field, ::Colon) = field

Base.@propagate_inbounds Base.getindex(field::PointField) =
    getindex(field_values(field))
Base.@propagate_inbounds Base.setindex!(field::PointField, val) =
    setindex!(field_values(field), val)

"""
    set!(f::Function, field::Field, args = ())

Apply function `f` to populate
values in field `field`. `f` must
have a function signature with signature
`f(::LocalGeometry[, args...])`.
Additional arguments may be passed to
`f` with `args`.
"""
function set!(f::Function, field::Field, args = ())
    space = axes(field)
    local_geometry = local_geometry_field(space)
    field .= f.(local_geometry, args...)
    return nothing
end

#=
This function can be used to truncate the printing
of ClimaCore `Field` types, which can get rather
long.

# Example
```
import ClimaCore
ClimaCore.Fields.truncate_printing_field_types() = true
```
=#
truncate_printing_field_types() = false

function Base.show(io::IO, ::Type{T}) where {T <: Fields.Field}
    if truncate_printing_field_types()
        print(io, truncated_field_type_string(T))
    else
        invoke(show, Tuple{IO, Type}, io, T)
    end
end

# Defined for testing
function truncated_field_type_string(::Type{T}) where {T <: Fields.Field}
    values_type(::Type{T}) where {V, T <: Fields.Field{V}} = V

    _apply!(f, ::T, match_list) where {T} = nothing # sometimes we need this...
    function _apply!(f, ::Type{T}, match_list) where {T}
        if f(T)
            push!(match_list, T)
        end
        for p in T.parameters
            _apply!(f, p, match_list)
        end
    end
    #     apply(::T) where {T <: Any}
    # Recursively traverse type `T` and apply
    # `f` to the types (and type parameters).
    # Returns a list of matches where `f(T)` is true.
    apply(f, ::T) where {T} = apply(f, T)
    function apply(f, ::Type{T}) where {T}
        match_list = []
        _apply!(f, T, match_list)
        return match_list
    end

    # We can't gaurantee that printing for all
    # field types will succeed, so fallback to
    # printing `Field{...}` if this fails.
    try
        V = values_type(T)
        nts = apply(x -> x <: NamedTuple, eltype(V))
        syms = unique(map(nt -> fieldnames(nt), nts))
        s = join(syms, ",")
        return "Field{$s} (trunc disp)"
    catch
        @warn "Could not print field. Please open a an issue with the runscript."
        return "Field{...} (trunc disp)"
    end
end

end # module
