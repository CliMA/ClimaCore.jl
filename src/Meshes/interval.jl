"""
    IntervalMesh <: AbstractMesh

A 1D mesh on an `IntervalDomain`.

# Constuctors

    IntervalMesh(domain::IntervalDomain, faces::AbstractVector)

Construct a 1D mesh with face locations at `faces`.

    IntervalMesh(domain::IntervalDomain[, stetching=Uniform()]; nelems=)

Constuct a 1D mesh on `domain` with `nelems` elements, using `stretching`. Possible values of `stretching` are:

- [`Uniform()`](@ref)
- [`ExponentialStretching(H)`](@ref)
"""
struct IntervalMesh{FT, I <: IntervalDomain, V <: AbstractVector} <:
       AbstractMesh{FT}
    domain::I
    faces::V
end

function IntervalMesh(
    domain::I,
    faces::V,
) where {
    I <:
    IntervalDomain{CT},
    V <:
    AbstractVector{CT},
} where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    IntervalMesh{FT, I, V}(domain, faces)
end


abstract type StretchingRule end

"""
    Uniform()

Use uniformly-sized elements.
"""
struct Uniform <: StretchingRule end

function IntervalMesh(
    domain::IntervalDomain{CT},
    ::Uniform = Uniform();
    nelems,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    faces = range(domain.coord_min, domain.coord_max; length = nelems + 1)
    IntervalMesh(domain, faces)
end


"""
    ExponentialStretching(H)

Apply exponential stretching to the domain when constructing elements. `H` is
the scale height (a typical atmospheric scale height `H ≈ 7.5e3`km).

For an interval ``[z_0,z_1]``, this makes the elements uniformally spaced in
``\\zeta``, where
```math
\\zeta = \\frac{1 - e^{-\\eta/h}}{1-e^{-1/h}},
```
where ``\\eta = \\frac{z - z_0}{z_1-z_0}``, and ``h = \\frac{H}{z_1-z_0}`` is
the non-dimensional scale height.
"""
struct ExponentialStretching{FT} <: StretchingRule
    H::FT
end

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::ExponentialStretching;
    nelems,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    cmin = Geometry.component(domain.coord_min, 1)
    cmax = Geometry.component(domain.coord_max, 1)
    R = cmax - cmin
    h = stretch.H / R
    η(ζ) = -h * log1p((expm1(-1 / h)) * ζ)
    faces =
        [CT(cmin + R * η(ζ)) for ζ in range(FT(0), FT(1); length = nelems + 1)]
    IntervalMesh(domain, faces)
end


"""
    NoStretching

Apply no stretching to the domain when constructing elements. 
"""
struct NoStretching <: StretchingRule end

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::NoStretching;
    nelems,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    cmin = Geometry.component(domain.coord_min, 1)
    cmax = Geometry.component(domain.coord_max, 1)
    Δc = (cmax - cmin) / (nelems)

    faces = [CT(cmin + i * Δc) for i in 0:nelems]
    IntervalMesh(domain, faces)
end



function Base.show(io::IO, mesh::IntervalMesh)
    nelements = length(mesh.faces) - 1
    print(io, nelements, "-element IntervalMesh of ")
    print(io, mesh.domain)
end
