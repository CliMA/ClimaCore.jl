"""
    AbstractFieldStyle

The supertype of all broadcasting-like operations on Fields.
"""
abstract type AbstractFieldStyle <: Base.BroadcastStyle end

"""
    FieldStyle{DS <: DataStyle}

Standard broadcasting on Fields. Delegates the actual work to `DS`.
"""
struct FieldStyle{DS <: DataStyle} <: AbstractFieldStyle end

FieldStyle(::DS) where {DS <: DataStyle} = FieldStyle{DS}()

Base.Broadcast.BroadcastStyle(::Type{Field{V, M}}) where {V, M} =
    FieldStyle(DataStyle(V))

Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.AbstractArrayStyle{0},
    b::AbstractFieldStyle,
) = b

Base.Broadcast.broadcastable(field::Field) = field

# Specialize handling of +, *, muladd, so that we can support broadcasting over NamedTuple element types
# Required for ODE solvers
Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(+), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.:⊞, args...)

Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(-), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.:⊟, args...)

Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(*), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.:⊠, args...)

Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(/), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.rdiv, args...)

Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(muladd), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.rmuladd, args...)

Base.eltype(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle}) =
    Base.Broadcast.combine_eltypes(bc.f, bc.args)

# we implement our own to avoid the type-widening code, and throw a more useful error
@inline function Base.copy(
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractFieldStyle}
    ElType = eltype(bc)
    if Base.isconcretetype(ElType)
        # We can trust it and defer to the simpler `copyto!`
        return copyto!(similar(bc, ElType), bc)
    end
    error("cannot infer concrete eltype of $(bc.f) on $(map(eltype, bc.args))")
end

# Return underlying DataLayout object, DataStyle of broadcasted
# for `Base.similar` of a Field
todata(obj) = obj
todata(field::Field) = Fields.field_values(field)
function todata(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    Base.Broadcast.Broadcasted{DS}(bc.f, map(todata, bc.args))
end

Fields.space(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle}) = axes(bc)[1]

function Base.similar(
    bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle},
    ::Type{Eltype},
) where {Eltype}
    return Field(similar(todata(bc), Eltype), Fields.space(bc))
end

function Base.copyto!(
    dest::Field,
    bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle},
)
    copyto!(field_values(dest), todata(bc))
    return dest
end

# Define the axes field to be a 1-tuple containing the space of the return field
# for checking Base.Broadcast.Broadcasted axes shapes
Base.axes(field::Field) = (Fields.space(field),)

# Overload broadcast axes shape checking for more useful error message for Field Spaces
function Base.Broadcast.check_broadcast_shape(
    (space1,)::Tuple{AbstractSpace},
    (space2,)::Tuple{AbstractSpace},
)
    if space1 !== space2
        error("Mismatched spaces\n$space1\n$space2")
    end
    return nothing
end

function Base.Broadcast.check_broadcast_shape(::Tuple{AbstractSpace}, ::Tuple{})
    return nothing
end

function Base.Broadcast.check_broadcast_shape(
    ::Tuple{AbstractSpace},
    ax2::Tuple,
)
    error("$ax2 is not a AbstractSpace")
end
