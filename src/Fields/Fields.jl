module Fields

import ..slab, ..column
import ..DataLayouts: DataLayouts, AbstractData, DataStyle
import ..Spaces: Spaces, AbstractSpace
import ..Geometry: Geometry, Cartesian12Vector
import ..RecursiveApply
import ..Topologies


import LinearAlgebra

using Requires

function __init__()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        function OrdinaryDiffEq.calculate_residuals!(
            out::Fields.Field,
            ũ::Fields.Field,
            u₀::Fields.Field,
            u₁::Fields.Field,
            α,
            ρ,
            internalnorm,
            t,
        )
            OrdinaryDiffEq.calculate_residuals!(
                parent(out),
                parent(ũ),
                parent(u₀),
                parent(u₁),
                α,
                ρ,
                internalnorm,
                t,
            )
        end
    end
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

function CentField(cs::Spaces.FiniteDifferenceSpace)
    FT = Spaces.undertype(cs)
    return Field(DataLayouts.VF{FT}(zeros(FT, Spaces.n_cells(cs), 1)), cs)
end

function FaceField(cs::Spaces.FiniteDifferenceSpace)
    FT = Spaces.undertype(cs)
    return Field(DataLayouts.VF{FT}(zeros(FT, Spaces.n_faces(cs), 1)), cs)
end

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
Base.size(field::Field) = () # to play nice with DifferentialEquations; may want to revisit this

function slab(field::Field, h)
    Field(slab(field_values(field), h), slab(space(field), h))
end


Topologies.nlocalelems(field::Field) = Topologies.nlocalelems(space(field))


# printing
function Base.show(io::IO, field::Field)
    print(io, eltype(field), "-valued Field:")
    _show_compact_field(io, field, "  ", true)
    print(io, "\non ", space(field))
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

# TODO: nice printing
# follow x-array like printing?
# repl: #https://earth-env-data-science.github.io/lectures/xarray/xarray.html
# html: https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html


# Broadcasting
struct FieldStyle{DS <: DataStyle} <: Base.BroadcastStyle end
FieldStyle(::DS) where {DS <: DataStyle} = FieldStyle{DS}()

Base.Broadcast.BroadcastStyle(::Type{Field{V, M}}) where {V, M} =
    FieldStyle(DataStyle(V))


Base.Broadcast.BroadcastStyle(
    a::Base.Broadcast.AbstractArrayStyle{0},
    b::FieldStyle,
) = b

Base.Broadcast.broadcastable(field::Field{V, M}) where {V, M} = field

Base.axes(field::Field) = (space(field),)

# TODO: we may want to rethink how we handle this to allow for more lazy operations
todata(obj) = obj
todata(field::Field) = field_values(field)
function todata(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    Base.Broadcast.Broadcasted{DS}(bc.f, map(todata, bc.args))
end

# we specialize handling of +, *, muladd, so that we can support broadcasting over NamedTuple element types
# these are required for ODE solvers
# TODO: it may be more efficient to handle this at the array level?
Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(+), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.:⊞, args...)
Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(-), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.:⊟, args...)
Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(*), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.:⊠, args...)
Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(/), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.rdiv, args...)
Base.Broadcast.broadcasted(fs::FieldStyle, ::typeof(muladd), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.rmuladd, args...)


function space(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    return axes(bc)[1]
end

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

# we implement our own to avoid the type-widening code, and throw a more useful error
@inline function Base.copy(
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: FieldStyle}
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    if Base.isconcretetype(ElType)
        # We can trust it and defer to the simpler `copyto!`
        return copyto!(similar(bc, ElType), bc)
    end
    error("cannot infer concrete eltype of $(bc.f) on $(map(eltype, bc.args))")
end

function Base.copyto!(dest::Field{V, M}, src::Field{V, M}) where {V, M}
    @assert space(dest) == space(src)
    copyto!(field_values(dest), field_values(src))
    return dest
end


function Base.zero(field::Field)
    zfield = similar(field::Field)
    zarray = parent(zfield)
    fill!(zarray, zero(eltype(zarray)))
    return zfield
end

function Base.similar(
    bc::Base.Broadcast.Broadcasted{FieldStyle{DS}},
    ::Type{Eltype},
) where {DS, Eltype}
    return Field(similar(todata(bc), Eltype), space(bc))
end

function Base.copyto!(
    dest::Field{V, M},
    bc::Base.Broadcast.Broadcasted{FieldStyle{DS}},
) where {V, M, DS}
    copyto!(field_values(dest), todata(bc))
    return dest
end

function Base.Broadcast.check_broadcast_shape(
    (space1,)::Tuple{AbstractSpace},
    (space2,)::Tuple{AbstractSpace},
)
    if space1 !== space2
        error("Mismatched spaces\n$space1\n$space2")
    end
    return nothing
end
function Base.Broadcast.check_broadcast_shape(
    ax1::Tuple{AbstractSpace},
    ax2::Tuple{},
)
    return nothing
end
function Base.Broadcast.check_broadcast_shape(
    ax1::Tuple{AbstractSpace},
    ax2::Tuple,
)
    error("$ax2 is not a AbstractSpace")
end


# useful operations
Base.map(fn, field::Field) = Base.broadcast(fn, field)

weighted_jacobian(cm::Spaces.FaceFiniteDifferenceSpace) = cm.face.Δh
weighted_jacobian(cm::Spaces.CenterFiniteDifferenceSpace) = cm.cent.Δh
weighted_jacobian(m::Spaces.SpectralElementSpace2D) = m.local_geometry.WJ
weighted_jacobian(field) = weighted_jacobian(space(field))

# sum will give the integral over the field
function Base.sum(field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}})
    Base.reduce(
        RecursiveApply.radd,
        Base.Broadcast.broadcasted(
            RecursiveApply.rmul,
            weighted_jacobian(field),
            todata(field),
        ),
    )
end
function Base.sum(fn, field::Field)
    # Can't just call mapreduce as we need to weight _after_ applying the function
    Base.sum(Base.Broadcast.broadcasted(fn, field))
end

function LinearAlgebra.norm(field::Field, p::Real = 2)
    if p == 2
        # currently only one which supports structured types
        sqrt(sum(LinearAlgebra.norm_sqr, field))
    elseif p == 1
        sum(abs, field)
    elseif p == Inf
        error("Inf norm not yet supported")
    else
        sum(x -> x^p, field)^(1 / p)
    end
end

function Base.isapprox(
    x::Field,
    y::Field;
    atol::Real = 0,
    rtol::Real = Base.rtoldefault(eltype(parent(x)), eltype(parent(y)), atol),
    nans::Bool = false,
    norm::Function = LinearAlgebra.norm,
)
    d = norm(x .- y)
    return isfinite(d) && d <= max(atol, rtol * max(norm(x), norm(y)))
end



# for compatibility with OrdinaryDiffEq
# Based on ApproxFun definitions
#  https://github.com/SciML/RecursiveArrayTools.jl/blob/6e779acb321560c75e27739a89ae553cd0f332f1/src/init.jl#L8-L12
# and
#  https://discourse.julialang.org/t/shallow-water-equations-with-differentialequations-jl/2691/16?u=simonbyrne

# This one is annoying
#  https://github.com/SciML/OrdinaryDiffEq.jl/blob/181dcf265351ed3c02437c89a8d2af3f6967fa85/src/initdt.jl#L80
Base.any(f, field::Field) = any(f, parent(field))

import RecursiveArrayTools

RecursiveArrayTools.recursive_unitless_eltype(field::Field) = typeof(field)
RecursiveArrayTools.recursive_unitless_bottom_eltype(field::Field) =
    RecursiveArrayTools.recursive_unitless_bottom_eltype(parent(field))
RecursiveArrayTools.recursive_bottom_eltype(field::Field) =
    RecursiveArrayTools.recursive_bottom_eltype(parent(field))

function RecursiveArrayTools.recursivecopy!(dest::F, src::F) where {F <: Field}
    copy!(parent(dest), parent(src))
    return dest
end

RecursiveArrayTools.recursivecopy(field::Field) = copy(field)
# avoid call to deepcopy
RecursiveArrayTools.recursivecopy(a::AbstractArray{<:Field}) =
    map(RecursiveArrayTools.recursivecopy, a)

"""
    coordinate_field(space::AbstractSpace)

Construct a `Field` of the coordinates of the space.
"""
coordinate_field(space::AbstractSpace) = Field(space.coordinates, space)
coordinate_field(field::Field) = coordinates(space(field))

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
