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
FieldStyle(x::Base.Broadcast.Unknown) = x

Base.Broadcast.BroadcastStyle(::Type{Field{V, S}}) where {V, S} =
    FieldStyle(DataStyle(V))

# Broadcasting over scalars (Ref or Tuple)
Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.AbstractArrayStyle{0},
    fs::AbstractFieldStyle,
) = fs
Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.Style{Tuple},
    fs::AbstractFieldStyle,
) = fs

Base.Broadcast.BroadcastStyle(
    ::FieldStyle{DS1},
    ::FieldStyle{DS2},
) where {DS1, DS2} = FieldStyle(Base.Broadcast.BroadcastStyle(DS1(), DS2()))

Base.Broadcast.broadcastable(field::Field) = field

function Adapt.adapt_structure(
    to,
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractFieldStyle}
    Base.Broadcast.Broadcasted{Style}(
        Adapt.adapt(to, bc.f),
        Adapt.adapt(to, bc.args),
        Adapt.adapt(to, bc.axes),
    )
end

Base.eltype(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle}) =
    Base.Broadcast.combine_eltypes(bc.f, bc.args)

# _first: recursively get the first element
function _first end

# If we haven't caught the datatype, then this
# may just result in a method error-- but all
# we're trying to do is throw a more helpful
# error message. So, let's throw it here instead.
_first(bc, ::Any) = throw(BroadcastInferenceError(bc))
_first_data_layout(data::DataLayouts.VF) = data[1]
_first_data_layout(data::DataLayouts.DataF) = data[]
_first(bc, x::Real) = x
_first(bc, x::Geometry.LocalGeometry) = x
_first(bc, data::DataLayouts.VF) = data[]
_first(bc, field::Field) =
    _first_data_layout(field_values(column(field, 1, 1, 1)))
_first(bc, space::Spaces.AbstractSpace) =
    _first_data_layout(field_values(column(space, 1, 1, 1)))
_first(bc, x::Base.Broadcast.Broadcasted) = _first(bc, copy(x))
_first(bc, x::Ref{T}) where {T} = x.x
_first(bc, x::Tuple{T}) where {T} = x[1]

function call_with_first(bc)
    # Try calling with first applied to all arguments:
    bc′ = Base.Broadcast.preprocess(nothing, bc)
    first_args = map(arg -> _first(bc, arg), bc′.args)
    bc.f(first_args...)
end

# we implement our own to avoid the type-widening code, and throw a more useful error
struct BroadcastInferenceError <: Exception
    bc::Base.Broadcast.Broadcasted
end

function Base.showerror(io::IO, err::BroadcastInferenceError)
    print(io, "BroadcastInferenceError: cannot infer eltype.\n")
    bc = err.bc
    f = bc.f
    eltypes = map(eltype, bc.args)
    if !hasmethod(f, eltypes)
        print(io, "  function $(f) does not have a method for $(eltypes)")
    else
        InteractiveUtils.code_warntype(io, f, eltypes)
    end
end

function Base.copy(
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractFieldStyle}
    ElType = eltype(bc)
    if !Base.isconcretetype(ElType)
        call_with_first(bc)
        throw(BroadcastInferenceError(bc))
    end
    # We can trust it and defer to the simpler `copyto!`
    return copyto!(similar(bc, ElType), bc)
end

Base.@propagate_inbounds function slab(
    bc::Base.Broadcast.Broadcasted{Style},
    v,
    h,
) where {Style <: AbstractFieldStyle}
    _args = slab_args(bc.args, v, h)
    _axes = slab(axes(bc), v, h)
    Base.Broadcast.Broadcasted{Style}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function column(
    bc::Base.Broadcast.Broadcasted{Style},
    i,
    j,
    h,
) where {Style <: AbstractFieldStyle}
    _args = column_args(bc.args, i, j, h)
    _axes = column(axes(bc), i, j, h)
    Base.Broadcast.Broadcasted{Style}(bc.f, _args, _axes)
end

# Return underlying DataLayout object, DataStyle of broadcasted
# for `Base.similar` of a Field
_todata_args(args::Tuple) = (todata(args[1]), _todata_args(Base.tail(args))...)
_todata_args(args::Tuple{Any}) = (todata(args[1]),)
_todata_args(::Tuple{}) = ()

todata(obj) = obj
todata(field::Field) = Fields.field_values(field)
function todata(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    _args = _todata_args(bc.args)
    Base.Broadcast.Broadcasted{DS}(bc.f, _args)
end

# same logic as Base.Broadcasted (which only defines it for Tuples)
Base.axes(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle}) =
    _axes(bc, bc.axes)
_axes(bc, ::Nothing) = Base.Broadcast.combine_axes(bc.args...)
_axes(bc, axes) = axes

function Base.similar(
    bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle},
    ::Type{Eltype},
) where {Eltype}
    return Field(similar(todata(bc), Eltype), axes(bc))
end

@inline function Base.copyto!(
    dest::Field,
    bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle},
)
    copyto!(field_values(dest), Base.Broadcast.instantiate(todata(bc)))
    return dest
end

@noinline function error_mismatched_spaces(space1::Type, space2::Type)
    error("Broacasted spaces are not the same.")
end

@inline function Base.Broadcast.broadcast_shape(
    space1::AbstractSpace,
    space2::AbstractSpace,
)
    if space1 !== space2
        if Spaces.issubspace(space2, space1)
            return space1
        elseif Spaces.issubspace(space1, space2)
            return space2
        else
            error_mismatched_spaces(typeof(space1), typeof(space2))
        end
    end
    return space1
end
@inline Base.Broadcast.broadcast_shape(space::AbstractSpace, ::Tuple) = space
@inline Base.Broadcast.broadcast_shape(::Tuple, space::AbstractSpace) = space

@inline Base.Broadcast.broadcast_shape(
    pointspace::AbstractPointSpace,
    space::AbstractSpace,
) = space
@inline Base.Broadcast.broadcast_shape(
    space::AbstractSpace,
    pointspace::AbstractPointSpace,
) = space

# Avoid method ambiguity:
@inline Base.Broadcast.broadcast_shape(
    a::AbstractPointSpace,
    b::AbstractPointSpace,
) = a

# Overload broadcast axes shape checking for more useful error message for Field Spaces
@inline function Base.Broadcast.check_broadcast_shape(
    space1::AbstractSpace,
    space2::AbstractSpace,
)
    if space1 !== space2
        if Spaces.issubspace(space2, space1) ||
           Spaces.issubspace(space1, space2)
            nothing
        else
            error_mismatched_spaces(typeof(space1), typeof(space2))
        end
    end
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    space::AbstractSpace,
    ax2::Tuple,
)
    error_mismatched_spaces(typeof(space), typeof(ax2))
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractSpace,
    ::Tuple{},
)
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractSpace,
    ::Tuple{T},
) where {T}
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractSpace,
    ::AbstractPointSpace,
)
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractPointSpace,
    ::AbstractSpace,
)
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractPointSpace,
    ::AbstractPointSpace,
)
    return nothing
end

# types aren't isbits
Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::Type{T},
    args...,
) where {T} = Base.Broadcast.broadcasted(fs, (x...) -> T(x...), args...)

# GPU support for type wrappers, like `Geometry.AxisTensor`s
Base.Broadcast.broadcasted(
    fs::Base.Broadcast.DefaultArrayStyle{0},
    ::Type{T},
    args...,
) where {T <: Geometry.AxisTensor} =
    Base.Broadcast.broadcasted(fs, (x...) -> T(x...), args...)

Base.Broadcast.broadcasted(
    ::typeof(Base.literal_pow),
    ::typeof(^),
    f::Field,
    ::Val{n},
) where {n} = Base.Broadcast.broadcasted(x -> Base.literal_pow(^, x, Val(n)), f)

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

# Specialize handling of vector-based functions to automatically add LocalGeometry information
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.norm),
    arg,
)
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry._norm,
        arg,
        local_geometry_field(space),
    )
end
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.norm_sqr),
    arg,
)
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry._norm_sqr,
        arg,
        local_geometry_field(space),
    )
end

function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.cross),
    arg1,
    arg2,
)
    space = Fields.axes(arg1)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry._cross,
        arg1,
        arg2,
        local_geometry_field(space),
    )
end
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(Geometry.transform),
    arg1,
    arg2,
)
    space = Fields.axes(arg2)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry.transform,
        arg1,
        arg2,
        local_geometry_field(space),
    )
end
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(Geometry.project),
    arg1,
    arg2,
)
    space = Fields.axes(arg2)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry.project,
        arg1,
        arg2,
        local_geometry_field(space),
    )
end

function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::Type{V},
    arg,
) where {V <: Geometry.AxisVector}
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(fs, V, arg, local_geometry_field(space))
end

function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::Type{V},
    arg,
) where {V <: Geometry.CartesianVector}
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        V,
        arg,
        tuple(Spaces.global_geometry(space)),
        local_geometry_field(space),
    )
end

function Base.Broadcast.copyto!(
    field::Field,
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
)
    copyto!(Fields.field_values(field), bc)
    return field
end
function Base.Broadcast.copyto!(
    field::Field,
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple}},
)
    copyto!(Fields.field_values(field), bc)
    return field
end

function Base.Broadcast.copyto!(field::Field, nt::NamedTuple)
    copyto!(
        field,
        Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}(
            identity,
            (nt,),
            axes(field),
        ),
    )
end
