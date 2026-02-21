Base.map(fn, fields::Field...) = Base.broadcast(fn, fields...)
Base.map!(fn, dest::Field, fields::Field...) =
    Base.broadcast!(fn, dest, fields...)

"""
    Fields.local_sum(v::Field)

Compute the approximate integral of `v` over the domain local to the current
context.

See [`sum`](@ref) for the integral over the full domain.
"""
function local_sum(
    field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}},
    dev::ClimaComms.AbstractCPUDevice,
)
    result = Base.sum(
        Base.Broadcast.broadcasted(
            *,
            Spaces.weighted_jacobian(axes(field)),
            todata(field),
        ),
    )
    call_post_op_callback() && post_op_callback(result, field, dev)
    result
end
local_sum(field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}}) =
    local_sum(field, ClimaComms.device(axes(field)))
"""
    sum([f=identity,] v::Field)

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

If `v` is a distributed field, this uses a `ClimaComms.allreduce` operation.
"""
Base.sum(field::Field) = Base.sum(identity, field)
function Base.sum(
    field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}},
    ::ClimaComms.AbstractCPUDevice,
)
    context = ClimaComms.context(axes(field))
    data_sum = DataLayouts.DataF(local_sum(field))
    ClimaComms.allreduce!(context, parent(data_sum), +)
    return data_sum[]
end
Base.sum(fn, field::Field, ::ClimaComms.AbstractCPUDevice) =
    Base.sum(Base.Broadcast.broadcasted(fn, field))
Base.sum(field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}}) =
    Base.sum(field, ClimaComms.device(axes(field)))
Base.sum(fn, field::Field) = Base.sum(fn, field, ClimaComms.device(field))

"""
    maximum([f=identity,]v::Field)

Approximate maximum of `v` or `f.(v)` over the domain.

If `v` is a distributed field, this uses a `ClimaComms.allreduce` operation.
"""
function Base.maximum(fn, field::Field, ::ClimaComms.AbstractCPUDevice)
    context = ClimaComms.context(axes(field))
    data_max = DataLayouts.DataF(mapreduce(fn, max, todata(field)))
    ClimaComms.allreduce!(context, parent(data_max), max)
    return data_max[]
end
Base.maximum(field::Field, device::ClimaComms.AbstractCPUDevice) =
    maximum(identity, field, device)
Base.maximum(fn, field::Field) =
    Base.maximum(fn, field, ClimaComms.device(field))
Base.maximum(field::Field) = Base.maximum(field, ClimaComms.device(field))

function Base.minimum(fn, field::Field, ::ClimaComms.AbstractCPUDevice)
    context = ClimaComms.context(axes(field))
    data_min = DataLayouts.DataF(mapreduce(fn, min, todata(field)))
    ClimaComms.allreduce!(context, parent(data_min), min)
    return data_min[]
end
Base.minimum(field::Field, device::ClimaComms.AbstractCPUDevice) =
    minimum(identity, field, device)
Base.minimum(fn, field::Field) =
    Base.minimum(fn, field, ClimaComms.device(field))
Base.minimum(field::Field) = Base.minimum(field, ClimaComms.device(field))
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

If `v` is a distributed field, this uses a `ClimaComms.allreduce` operation.
"""
Statistics.mean(field::Field) = Statistics.mean(identity, field)
function Statistics.mean(
    field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}},
    ::ClimaComms.AbstractCPUDevice,
)
    space = axes(field)
    context = ClimaComms.context(space)
    data_combined =
        DataLayouts.DataF((local_sum(field), Spaces.local_area(space)))
    ClimaComms.allreduce!(context, parent(data_combined), +)
    sum_v, area_v = data_combined[]
    return sum_v / area_v
end
Statistics.mean(fn, field::Field, ::ClimaComms.AbstractCPUDevice) =
    Statistics.mean(Base.Broadcast.broadcasted(fn, field))

Statistics.mean(field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}}) =
    Statistics.mean(field, ClimaComms.device(axes(field)))
Statistics.mean(fn, field::Field) =
    Statistics.mean(fn, field, ClimaComms.device(axes(field)))

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
LinearAlgebra.norm(field::Field, p::Real = 2; normalize = true) =
    LinearAlgebra.norm(
        field,
        ClimaComms.device(axes(field)),
        p,
        normalize = normalize,
    )

function LinearAlgebra.norm(
    field::Field,
    ::ClimaComms.AbstractCPUDevice,
    p::Real = 2;
    normalize = true,
)
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
    x::T,
    y::T;
    atol::Real = 0,
    rtol::Real = Base.rtoldefault(eltype(parent(x)), eltype(parent(y)), atol),
    nans::Bool = false,
    norm::Function = LinearAlgebra.norm,
) where {T <: Field}
    Δ = (x .- y)
    FT = Spaces.undertype(axes(x))
    d = norm(Δ)::FT
    nx = norm(x)::FT
    ny = norm(y)::FT
    return isfinite(d) && d <= max(atol, rtol * max(nx, ny))
end

Base.:(==)(field1::Field, field2::Field) =
    axes(field1) === axes(field2) && parent(field1) == parent(field2)
