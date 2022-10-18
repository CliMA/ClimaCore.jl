Base.map(fn, field::Field) = Base.broadcast(fn, field)

# useful operations
weighted_jacobian(field) = weighted_jacobian(axes(field))
weighted_jacobian(space::Spaces.AbstractSpace) =
    Spaces.local_geometry_data(space).WJ

_local_area(field) = Base.sum(weighted_jacobian(field))
_area(field) =
    ClimaComms.allreduce(comm_context(axes(field)), _local_area(field), +)

_local_sum(field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}}) =
    Base.reduce(
        RecursiveApply.radd,
        Base.Broadcast.broadcasted(
            RecursiveApply.rmul,
            weighted_jacobian(field),
            todata(field),
        ),
    )
_local_sum(fn, field::Field) = _local_sum(Base.Broadcast.broadcasted(fn, field))
"""
    sum([f=identity,]v::Field)

Approximate integration of `v` or `f.(v)` over the domain. In an `AbstractSpectralElementSpace`,
an integral over the entire space is computed by summation over the elements of the integrand
multiplied by the Jacobian determinants and the quadrature weights at each node within an element.
Hence, `sum` is computed by summation of the field values multiplied by the Jacobian determinants
and quadrature weights:

```math
\\sum_i f(v_i) W_i J_i
\\approx
\\int_\\Omega f(v) \\, d \\Omega
```
where ``v_i`` is the value at each node, and ``f`` is the identity function if not specified.
"""
function Base.sum(field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}})
    context = comm_context(axes(field))
    local_sum = _local_sum(field)
    if local_sum isa Number
        ClimaComms.allreduce(context, local_sum, +)
    elseif local_sum isa NamedTuple
        S = typeof(local_sum)
        DataLayouts.DataF{S}(ClimaComms.allreduce(context, [local_sum...], +))[]
    end

end

Base.sum(fn, field::Field) = Base.sum(Base.Broadcast.broadcasted(fn, field))

function Base.maximum(fn, field::Field)
    context = comm_context(axes(field))
    localmax = mapreduce(fn, max, todata(field))
    if localmax isa Number
        ClimaComms.allreduce(context, localmax, max)
    elseif localmax isa NamedTuple
        S = typeof(localmax)
        DataLayouts.DataF{S}(ClimaComms.allreduce(context, [localmax...], max))[]
    end
end

Base.maximum(field::Field) = maximum(identity, field)

function Base.minimum(fn, field::Field)
    context = comm_context(axes(field))
    localmin = mapreduce(fn, min, todata(field))
    if localmin isa Number
        ClimaComms.allreduce(context, localmin, min)
    elseif localmin isa NamedTuple
        S = typeof(localmin)
        DataLayouts.DataF{S}(ClimaComms.allreduce(context, [localmin...], min))[]
    end
end

Base.minimum(field::Field) = minimum(identity, field)

# somewhat inefficient
Base.extrema(fn, field::Field) = (minimum(fn, field), maximum(fn, field))
Base.extrema(field::Field) = extrema(identity, field)

"""
    mean([f=identity, ]v::Field)

The mean value of `field` or `f.(field)` over the domain, weighted by area.
Similar to `sum`, in an `AbstractSpectralElementSpace`, this is computed by
summation of the field values multiplied by the Jacobian determinants and quadrature weights:

```math
\\frac{\\sum_i f(v_i) W_i J_i}{\\sum_i W_i J_i}
\\approx
\\frac{\\int_\\Omega f(v) \\, d \\Omega}{\\int_\\Omega \\, d \\Omega}
```
where ``v_i`` is the Field value at each node, and ``f`` is the identity function if not specified.
"""
function Statistics.mean(field::Field)
    context = comm_context(axes(field))
    RecursiveApply.rdiv(
        ClimaComms.allreduce(context, _local_sum(field), +),
        ClimaComms.allreduce(context, _local_area(field), +),
    )
end

function Statistics.mean(fn, field::Field)
    context = comm_context(axes(field))
    RecursiveApply.rdiv(
        ClimaComms.allreduce(context, _local_sum(fn, field), +),
        ClimaComms.allreduce(context, _local_area(field), +),
    )
end

"""
    norm(v::Field, p=2; normalize=true)

The approximate ``L^p`` norm of `v`, where ``L^p`` represents the space of measurable
functions for which the p-th power of the absolute value is Lebesgue integrable, that is:
```math
\\| v \\|_p = \\left( \\int_\\Omega |v|^p d \\Omega \\right)^{1/p}
```
where ``|v|`` is defined to be the absolute value if ``v`` is a scalar-valued Field, or the 2-norm
if it is a vector-valued Field or composite Field (see [LinearAlgebra.norm](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.norm)).
Similar to `sum` and `mean`, in an `AbstractSpectralElementSpace`, this is computed by
summation of the field values multiplied by the Jacobian determinants and quadrature weights.
If `normalize=true` (the default), then internally the discrete norm is divided
by the sum of the Jacobian determinants and quadrature weights:
```math
\\left(\\frac{\\sum_i |v_i|^p W_i J_i}{\\sum_i W_i J_i}\\right)^{1/p}
\\approx
\\left(\\frac{\\int_\\Omega |v|^p \\, d \\Omega}{\\int_\\Omega \\, d \\Omega}\\right)^{1/p}
```
If `p=Inf`, then the norm is the maximum of the absolute values
```math
\\max_i |v_i| \\approx \\sup_{\\Omega} |v|
```

Consequently all norms should have the same units for all ``p`` (being the same
as calling `norm` on a single value).

If `normalize=false`, then the denominator term is omitted, and so the result will be
the norm as described above multiplied by the length/area/volume of ``\\Omega`` to
the power of ``1/p``.
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

Base.:(==)(field1::Field, field2::Field) =
    axes(field1) === axes(field2) && parent(field1) == parent(field2)
