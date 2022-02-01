Base.map(fn, field::Field) = Base.broadcast(fn, field)

# useful operations
weighted_jacobian(field) = weighted_jacobian(axes(field))
weighted_jacobian(space::Spaces.AbstractSpace) =
    Spaces.local_geometry_data(space).WJ

"""
    sum([f=identity,]v::Field)

Approximate integration of `v` or `f.(v)` over the domain. This is computed by
summation of the field values multiplied by the Jacobians and the quadrature weights:

```math
\\sum_i f(v_i) W_i J_i
\\approx
\\int_\\Omega f(v) \\, d \\Omega
```
where ``v_i``` is the value at each node, and ``f``` is the identity function if not specified.
"""
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

Base.maximum(fn, field::Field) = mapreduce(fn, max, todata(field))
Base.maximum(field::Field) = maximum(identity, field)

Base.minimum(fn, field::Field) = mapreduce(fn, min, todata(field))
Base.minimum(field::Field) = minimum(identity, field)

# somewhat inefficient
Base.extrema(fn, field::Field) = (minimum(fn, field), maximum(fn, field))
Base.extrema(field::Field) = extrema(identity, field)

"""
    mean([f=identity, ]v::Field)

The mean value of `field` or `f.(field)` over the domain, weighted by area. This is computed by
summation of the field values multiplied by the Jacobians and the quadrature weights:

```math
\\frac{\\sum_i f(v_i) W_i J_i}{\\sum_i W_i J_i}
\\approx
\\frac{\\int_\\Omega f(v) \\, d \\Omega}{\\int_\\Omega \\, d \\Omega}
```
where ``v_i`` is the value at each node, and ``f`` is the identity function if not specified.
"""
function Statistics.mean(field::Field)
    Base.sum(field) / Base.sum(weighted_jacobian(field))
end
function Statistics.mean(fn, field::Field)
    Base.sum(fn, field) / Base.sum(weighted_jacobian(field))
end

"""
    norm(v::Field, p=2; normalize=true)

The approximate ``L^p`` norm of `v`. This is computed by summation of the norms
of the field values multiplied by the Jacobians and the quadrature weights. If
`normalize=true` (the default), then it is divided by the sum of the weights:
```math
\\left(\\frac{\\sum_i \\|v_i\\|^p W_i J_i}{\\sum_i W_i J_i}\\right)^{1/p}
\\approx
\\left(\\frac{\\int_\\Omega \\|v\\|^p \\, d \\Omega}{\\int_\\Omega \\, d \\Omega}\\right)^{1/p}
```
where ``v_i`` is the value at each node. If `p=Inf`, then the norm is the
maximum of the absolute values
```math
\\max_i \\|v_i\\| \\approx \\sup_{\\Omega} \\|v\\|
```

Consequently all norms should have the same units for all ``p`` (being the same
as calling `norm` on a single value).

If `normalize=false`, then the denominator term is omitted, and so will be
multiplied by the length/area/volume of ``\\Omega`` to the power of ``1/p``.
"""
function LinearAlgebra.norm(field::Field, p::Real = 2; normalize = true)
    if p == 2
        # currently only one which supports structured types
        if normalize
            sqrt(Statistics.mean(LinearAlgebra.norm_sqr, field))
        else
            sqrt(sum(LinearAlgebra.norm_sqr, field))
        end
    elseif p == 1
        if normalize
            Statistics.mean(abs, field)
        else
            sum(abs, field)
        end
    elseif p == Inf
        maximum(abs, field)
    else
        if normalize
            Statistics.mean(x -> x^p, field)^(1 / p)
        else
            sum(x -> x^p, field)^(1 / p)
        end
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
