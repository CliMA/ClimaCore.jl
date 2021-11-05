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



"""
    GeneralizedExponentialStretching

Apply Tapio's stretching to the domain when constructing elements. 
"""
struct GeneralizedExponentialStretching <: StretchingRule end


function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::GeneralizedExponentialStretching;
    nelems,
    # target grid spacings at surface and top (m)
    dz_s = 20.0,
    dz_t = 7000.0,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}

    cmin = Geometry.component(domain.coord_min, 1)
    cmax = Geometry.component(domain.coord_max, 1)

    exp_stretch(ζ_n, h) = -h .* log.(1 .- (1 .- exp.(-1 ./ h)) .* ζ_n)

    # nondimensional vertical coordinate
    ζ_n = LinRange(1, nelems, nelems) / nelems


    f_s(h) = dz_s - cmax * exp_stretch(ζ_n[1], h)
    h_s = find_zero(f_s, -dz_s / cmax / log(1 - 1 / nelems))
    f_t(h) = dz_t - cmax * (1 - exp_stretch(ζ_n[end - 1], h))
    h_t = find_zero(f_t, ((cmax - cmin) - dz_t) / cmax / log(nelems))

    # scale height variation with height
    h = h_s .+ (ζ_n .- ζ_n[1]) * (h_t - h_s) / (ζ_n[end - 1] - ζ_n[1])


    faces = cmin .+ (cmax - cmin) * exp_stretch(ζ_n, h)

    # add the bottom level
    faces = [cmin; faces...]

    IntervalMesh(domain, faces)
end


"""
    TableStretching

Apply stretching to the domain based on a table when constructing elements. 
"""
struct TableStretching <: StretchingRule end


function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::TableStretching;
    nelems,
    levels,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}

    cmin = Geometry.component(domain.coord_min, 1)
    cmax = Geometry.component(domain.coord_max, 1)

    @assert(length(levels) == nelems + 1)
    α = (cmax - cmin) / (levels[end] - levels[1])
    β = cmin - (cmax - cmin) / (levels[end] - levels[1]) * level[1]
    faces = α * levels .+ β

    IntervalMesh(domain, faces)
end

function Base.show(io::IO, mesh::IntervalMesh)
    nelements = length(mesh.faces) - 1
    print(io, nelements, "-element IntervalMesh of ")
    print(io, mesh.domain)
end
