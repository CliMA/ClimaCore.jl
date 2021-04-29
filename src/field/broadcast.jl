
struct FieldStyle{DS <: DataStyle} <: Base.BroadcastStyle
    datastyle::DS
end

Base.Broadcast.BroadcastStyle(::Type{Field{V, M}}) where {V, M} =
    FieldStyle(DataStyle(V))

Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.AbstractArrayStyle{0},
    b::FieldStyle,
) = b

Base.Broadcast.broadcastable(field::Field{V, M}) where {V, M} = field

Base.axes(field::Field) = (field_mesh(field),)

_todata(obj) = obj
_todata(field::Field) = field_values(field)

function _todata(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    Base.Broadcast.Broadcasted{DS}(bc.f, map(_todata, bc.args))
end

function field_mesh(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    if bc.axes isa Nothing
        error("Call instantiate to access mesh of Broadcasted")
    end
    return bc.axes[1]
end

function Base.similar(
    bc::Base.Broadcast.Broadcasted{FieldStyle{DS}},
    ::Type{Eltype},
) where {DS, Eltype}
    return Field(similar(_todata(bc), Eltype), field_mesh(bc))
end

function Base.copyto!(
    dest::Field{V, M},
    bc::Base.Broadcast.Broadcasted{FieldStyle{DS}},
) where {V, M, DS}
    copyto!(field_values(dest), _todata(bc))
    return dest
end