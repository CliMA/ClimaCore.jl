module Fields

import ClimaComms
import MultiBroadcastFusion as MBF
import ..slab, ..slab_args, ..column, ..column_args, ..level, ..level_args
import ..DebugOnly: call_post_op_callback, post_op_callback
import ..DataLayouts:
    DataLayouts, DataLayout, DataStyle, FusedMultiBroadcast, @fused_direct
import ..Domains
import ..Topologies
import ..Quadratures
import ..Grids: ColumnIndex, local_geometry_type
import ..Spaces: Spaces, AbstractSpace, AbstractPointSpace, cuda_synchronize
import ..Spaces: nlevels, ncolumns
import ..Spaces: get_mask, set_mask!
import ..Geometry: Geometry, Cartesian12Vector
import ..Utilities: PlusHalf, half, safe_eltype, unsafe_eltype
import ..Utilities: drop_auto_broadcasters, auto_broadcasted

using UnrolledUtilities
using ClimaComms
import Adapt

import StaticArrays, LinearAlgebra, Statistics

"""
    Field(values, space)

A set of `values` defined at each point of a `space`.
"""
struct Field{V <: DataLayout, S <: AbstractSpace}
    values::V
    space::S
end
Field(::Type{T}, space::AbstractSpace) where {T} =
    Field(similar(Spaces.coordinates_data(space), T), space)

local_geometry_type(::Field{V, S}) where {V, S} = local_geometry_type(S)

ClimaComms.context(field::Field) = ClimaComms.context(axes(field))

Adapt.adapt_structure(to, field::Field) =
    Field(Adapt.adapt(to, field_values(field)), Adapt.adapt(to, axes(field)))

## aliases
# Point Field
const PointField{V, S} =
    Field{V, S} where {V <: DataLayout, S <: Spaces.PointSpace}

# TODO: do we need to make this distinction? what about inside cuda kernels
#       when we replace with a PlaceHolerSpace?
const PointDataField{V, S} =
    Field{V, S} where {V <: DataLayout{<:Any, 0}, S <: Spaces.AbstractSpace}

# Spectral Element Field
const SpectralElementField{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.AbstractSpectralElementSpace}
const SpectralElementField1D{V, S} =
    Field{V, S} where {V <: DataLayout, S <: Spaces.SpectralElementSpace1D}
const SpectralElementField2D{V, S} =
    Field{V, S} where {V <: DataLayout, S <: Spaces.SpectralElementSpace2D}

const FiniteDifferenceField{V, S} =
    Field{V, S} where {V <: DataLayout, S <: Spaces.FiniteDifferenceSpace}
# Backwards-compatibility alias for the pre-rewrite ColumnField, which was a
# Field with DataLayouts.DataColumn values. Single-column fields are now
# identified by their space (a FiniteDifferenceSpace, which may wrap a
# ColumnGrid returned by `column(::ExtrudedFiniteDifferenceSpace, ...)`), so
# the old ColumnField is the same set of fields as FiniteDifferenceField.
const ColumnField = FiniteDifferenceField
const FaceFiniteDifferenceField{V, S} =
    Field{V, S} where {V <: DataLayout, S <: Spaces.FaceFiniteDifferenceSpace}
const CenterFiniteDifferenceField{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.CenterFiniteDifferenceSpace}

# Extruded Fields
const ExtrudedFiniteDifferenceField{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.ExtrudedFiniteDifferenceSpace}
const ExtrudedFiniteDifferenceField2D{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.ExtrudedFiniteDifferenceSpace2D}
const ExtrudedFiniteDifferenceField3D{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.ExtrudedFiniteDifferenceSpace3D}
const FaceExtrudedFiniteDifferenceField{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.FaceExtrudedFiniteDifferenceSpace}
const CenterExtrudedFiniteDifferenceField{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.CenterExtrudedFiniteDifferenceSpace}

#
const SpectralElementField1D{V, S} =
    Field{V, S} where {V <: DataLayout, S <: Spaces.SpectralElementSpace1D}
const ExtrudedSpectralElementField2D{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.ExtrudedSpectralElementSpace2D}

const RectilinearSpectralElementField2D{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.RectilinearSpectralElementSpace2D}
const ExtrudedRectilinearSpectralElementField3D{V, S} = Field{
    V,
    S,
} where {
    V <: DataLayout,
    S <: Spaces.ExtrudedRectilinearSpectralElementSpace3D,
}


# Cubed Sphere Fields

const CubedSphereSpectralElementField2D{V, S} = Field{
    V,
    S,
} where {V <: DataLayout, S <: Spaces.CubedSphereSpectralElementSpace2D}
const ExtrudedCubedSphereSpectralElementField3D{V, S} = Field{
    V,
    S,
} where {
    V <: DataLayout,
    S <: Spaces.ExtrudedCubedSphereSpectralElementSpace3D,
}

Base.propertynames(field::Field) = propertynames(getfield(field, :values))
Base.ndims(::Type{Field{V, S}}) where {V, S} = Base.ndims(V)
@inline field_values(field::Field) = getfield(field, :values)

# Define the axes field to be the todata(bc) of the return field
@inline Base.axes(field::Field) = getfield(field, :space)

# Define device and device array type
ClimaComms.device(field::Field) = ClimaComms.device(axes(field))
ClimaComms.array_type(field::Field) =
    ClimaComms.array_type(ClimaComms.device(field))

@inline Base.dotgetproperty(field::Field, prop) = Base.getproperty(field, prop)
@inline Base.getproperty(field::Field, i::Integer) =
    Field(getproperty(field_values(field), i), axes(field))
@inline Base.getproperty(field::Field, name::Symbol) =
    Field(getproperty(field_values(field), name), axes(field))

Base.eltype(::Type{<:Field{V}}) where {V} = eltype(V)
Base.parent(field::Field) = parent(field_values(field))

# to play nice with DifferentialEquations; may want to revisit this
# https://github.com/SciML/SciMLBase.jl/blob/697bd0c0c7365e77fa311f2d32eade70f43a8d50/src/solutions/ode_solutions.jl#L31
Base.size(field::Field) = ()
Base.length(field::Field) = 1

Topologies.nlocalelems(field::Field) = Topologies.nlocalelems(axes(field))

Base.@propagate_inbounds slab(field::Field, inds...) =
    Field(slab(field_values(field), inds...), slab(axes(field), inds...))

Base.@propagate_inbounds function column(field::Field, inds...)
    column_space = column(axes(field), inds...)
    column_data = column(field_values(field), inds...)
    Field(level_data(column_space, column_data), column_space)
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

function Base.copyto!(dest::Field, src::Field; mask = get_mask(axes(dest)))
    @assert axes(dest) == axes(src)
    copyto!(field_values(dest), field_values(src); mask)
    return dest
end

"""
    fill!(field::Field, value; mask = get_mask(axes(field)))

Fill `field` with `value`. The mask is extracted from the field's space,
and `fill!` is only applied where the `mask` is true.
"""
function Base.fill!(field::Field, value; mask = get_mask(axes(field)))
    fill!(field_values(field), value; mask)
    return field
end
"""
    fill(value, space::AbstractSpace)

Create a new `Field` on `space` and fill it with `value`.
"""
Base.fill(value::FT, space::AbstractSpace) where {FT} = fill!(Field(FT, space), value)

"""
    zeros(space::AbstractSpace)

Create a new field on `space` that is zero everywhere. Unlike `fill`, this also
zeroes out data at points that are masked out, so that the field does not
contain any uninitialized values.
"""
function Base.zeros(::Type{FT}, space::AbstractSpace) where {FT}
    field = Field(FT, space)
    fill!(parent(field), zero(eltype(parent(field))))
    return field
end
Base.zeros(space::AbstractSpace) = zeros(Spaces.undertype(space), space)

"""
    ones(space::AbstractSpace)

Create a new field on `space` that is one everywhere.
"""
function Base.ones(::Type{FT}, space::AbstractSpace) where {FT}
    field = Field(FT, space)
    fill!(parent(field), one(eltype(parent(field))))
    return field
end
Base.ones(space::AbstractSpace) = ones(Spaces.undertype(space), space)

function Base.zero(field::Field)
    zfield = similar(field)
    fill!(parent(zfield), zero(eltype(parent(zfield))))
    return zfield
end


"""
    coordinate_field(space::AbstractSpace)

Return a pointer to the input space's coordinates `Field`.
"""
coordinate_field(space::AbstractSpace) =
    Field(Spaces.coordinates_data(space), space)
coordinate_field(field::Field) = coordinate_field(axes(field))

"""
    local_geometry_field(space::AbstractSpace)

Return a pointer to the input space's `LocalGeometry` `Field`.
"""
local_geometry_field(space::AbstractSpace) =
    Field(Spaces.local_geometry_data(space), space)
local_geometry_field(field::Field) = local_geometry_field(axes(field))

Fields.local_geometry_field(bc::Base.Broadcast.Broadcasted) =
    Fields.local_geometry_field(axes(bc))

"""
    Δz_field(field::Field)
    Δz_field(space::AbstractSpace)

Return a pointer to the input space's `Field` containing the `Δz` values on the
same space as the given field.
"""
Δz_field(field::Field) = Δz_field(axes(field))
Δz_field(space::AbstractSpace) = Field(Spaces.Δz_data(space), space)

include("broadcast.jl")
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
    Spaces.weighted_dss!(f::Field, dss_buffer = Spaces.create_dss_buffer(field))

Apply weighted direct stiffness summation (DSS) to `f`. This operates in-place
(i.e. it modifies the `f`). `ghost_buffer` contains the necessary information
for communication in a distributed setting, see [`Spaces.create_dss_buffer`](@ref).

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
    Spaces.weighted_dss!(field1 => ghost_buffer1, field2 => ghost_buffer2, ...)

Call [`Spaces.weighted_dss!`](@ref) on multiple fields at once, overlapping
communication as much as possible.
"""
function Spaces.weighted_dss!(
    (field1, dss_buffer1)::Pair,
    field_buffer_pairs::Pair...,
)
    device = ClimaComms.device(axes(field1))
    Spaces.weighted_dss_prepare!(
        field_values(field1),
        axes(field1),
        dss_buffer1,
    )
    for (field, dss_buffer) in field_buffer_pairs
        Spaces.weighted_dss_prepare!(
            field_values(field),
            axes(field),
            dss_buffer,
        )
    end

    cuda_synchronize(device; blocking = true)
    dss_buffer1 isa Topologies.DSSBuffer &&
        ClimaComms.start(dss_buffer1.graph_context)
    for (field, dss_buffer) in field_buffer_pairs
        dss_buffer isa Topologies.DSSBuffer &&
            ClimaComms.start(dss_buffer.graph_context)
    end

    Spaces.weighted_dss_internal!(field1, dss_buffer1)
    for (field, dss_buffer) in field_buffer_pairs
        Spaces.weighted_dss_internal!(field, dss_buffer)
    end

    Spaces.weighted_dss_ghost!(field1, dss_buffer1)
    for (field, dss_buffer) in field_buffer_pairs
        Spaces.weighted_dss_ghost!(field, dss_buffer)
    end

    return nothing
end

"""
    Spaces.create_dss_buffer(field::Field)

Create a buffer for communicating neighbour information of `field`.
"""
Spaces.create_dss_buffer(field::Field) =
    Spaces.create_dss_buffer(field_values(field), axes(field))

Base.@propagate_inbounds function level(
    field::Union{
        CenterFiniteDifferenceField,
        CenterExtrudedFiniteDifferenceField,
    },
    v::Int,
)
    hspace = level(axes(field), v)
    data = level(field_values(field), v)
    Field(level_data(hspace, data), hspace)
end
Base.@propagate_inbounds function level(
    field::Union{FaceFiniteDifferenceField, FaceExtrudedFiniteDifferenceField},
    v::PlusHalf,
)
    hspace = level(axes(field), v)
    data = level(field_values(field), v.i + 1)
    Field(level_data(hspace, data), hspace)
end

# Levels of fields on column spaces are single points, so their data is
# converted to a DataF to match the local geometry of a PointSpace.
Base.@propagate_inbounds level_data(::Spaces.AbstractPointSpace, data) =
    Spaces.point_data(data)
@inline level_data(hspace, data) = data

Base.getindex(field::Field, ::Colon) = field

Base.@propagate_inbounds Base.getindex(field::PointField) =
    getindex(field_values(field))
Base.@propagate_inbounds Base.setindex!(field::PointField, val) =
    setindex!(field_values(field), val)

"""
    set!(f::Function, field::Field, args = ())

Apply function `f` to populate values in field `field`. `f` must have a function
signature with signature `f(::LocalGeometry[, args...])`. Additional arguments
may be passed to `f` with `args`.

## Example

```julia
using ClimaCore.Fields
using ClimaCore.CommonSpaces
ᶜspace = ExtrudedCubedSphereSpace(Float64;
    z_elem = 10,
    z_min = 0,
    z_max = 1,
    radius = 10,
    h_elem = 10,
    n_quad_points = 4,
    staggering = CellCenter(),
)
x = Fields.Field(Float64, ᶜspace)
Fields.set!(x) do lg
    sin(lg.coordinates.z)
end
```
"""
function set!(f::Function, field::Field, args = ())
    space = axes(field)
    local_geometry = local_geometry_field(space)
    field .= f.(local_geometry, args...)
    return nothing
end

if VERSION < v"1.10"
    #=
    This function can be used to truncate the printing
    of ClimaCore `Field` types, which can get rather
    long.

    # Example
    ```
    import ClimaCore
    ClimaCore.truncate_printing_field_types() = true
    ```
    =#
    truncate_printing_field_types() = false

    function Base.show(io::IO, ::Type{T}) where {T <: Field}
        if truncate_printing_field_types()
            print(io, truncated_field_type_string(T))
        else
            invoke(show, Tuple{IO, Type}, io, T)
        end
    end

    # Defined for testing
    function truncated_field_type_string(::Type{T}) where {T <: Field}
        values_type(::Type{T}) where {V, T <: Field{V}} = V

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
end


"""
    array2field(array, space)

Wraps `array` in a `ClimaCore` `Field` that is defined over `space`. Can be used
to simplify the process of getting and setting values in an `RRTMGPModel`; e.g.

```
    array2field(center_temperature, center_space) .= center_temperature_field
    face_flux_field .= array2field(model.face_flux, face_space)
```

The struct type of the resulting `Field` is set to the array's element type.
"""
function array2field(array, space)
    data = Spaces.local_geometry_data(space)
    array_size = DataLayouts.add_f_dim(size(data), 1, Val(DataLayouts.f_dim(data)))
    parent_array = reshape(array, array_size)
    return Field(DataLayouts.rebuild(data, parent_array, eltype(array)), space)
end

"""
    field2array(field)

Extracts a view of a `ClimaCore` `Field`'s underlying array. Can be used to
simplify the process of getting and setting values in an `RRTMGPModel`; e.g.
```
    center_temperature .= field2array(center_temperature_field)
    field2array(face_flux_field) .= face_flux
```

The dimensions of the resulting array are `([number of vertical nodes], number
of horizontal nodes)`, with the first dimension dropped for fields defined over
horizontal spaces. Only fields of scalars are supported; i.e., the element type
of the array must be the same as the struct type of `field`.
"""
function field2array(field::Field)
    if sizeof(eltype(field)) != sizeof(eltype(parent(field)))
        f_axis_size = sizeof(eltype(field)) ÷ sizeof(eltype(parent(field)))
        error("unable to use field2array because each Field element is \
               represented by $f_axis_size array elements (must be 1)")
    end
    Spaces.has_vertical(axes(field)) || return vec(parent(field))
    return reshape(parent(field), nlevels(axes(field)), :)
end

set_mask!(space::Spaces.AbstractSpace, field::Field) =
    set_mask!(Spaces.horizontal_space(space), field_values(field))

end # module
