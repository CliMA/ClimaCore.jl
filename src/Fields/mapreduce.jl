Base.map(fn, field::Field) = Base.broadcast(fn, field)

# useful operations
weighted_jacobian(field) = weighted_jacobian(axes(field))
weighted_jacobian(space::Spaces.AbstractSpace) =
    Spaces.local_geometry_data(space).WJ

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
