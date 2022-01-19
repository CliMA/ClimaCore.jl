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
struct IntervalMesh{I <: IntervalDomain, V <: AbstractVector} <: AbstractMesh1D
    domain::I
    faces::V
end

domain(mesh::IntervalMesh) = mesh.domain
nelements(mesh::IntervalMesh) = length(mesh.faces) - 1
elements(mesh::IntervalMesh) = Base.OneTo(nelements(mesh))

function Base.show(io::IO, mesh::IntervalMesh)
    print(io, nelements(mesh), "-element IntervalMesh of ")
    print(io, mesh.domain)
end

coordinates(mesh::IntervalMesh, elem::Integer, vert::Integer) =
    mesh.faces[elem + vert - 1]

function coordinates(
    mesh::IntervalMesh,
    elem::Integer,
    (ξ1,)::StaticArrays.SVector{1},
)
    ca = mesh.faces[elem]
    cb = mesh.faces[elem + 1]
    Geometry.linear_interpolate((ca, cb), ξ1)
end

function containing_element(mesh::IntervalMesh, coord)
    return min(searchsortedlast(mesh.faces, coord), nelements(mesh))
end
function reference_coordinates(mesh::IntervalMesh, elem::Integer, coord)
    lo = Geometry.component(mesh.faces[elem], 1)
    hi = Geometry.component(mesh.faces[elem + 1], 1)
    val = Geometry.component(coord, 1)
    ξ1 = ((val - lo) + (val - hi)) / (hi - lo)
    return StaticArrays.SVector(ξ1)
end

function is_boundary_face(mesh::IntervalMesh, elem::Integer, face)
    !Domains.isperiodic(mesh.domain) &&
        ((elem == 1 && face == 1) || (elem == nelements(mesh) && face == 2))
end

function boundary_face_name(mesh::IntervalMesh, elem::Integer, face)
    if !Domains.isperiodic(mesh.domain)
        if elem == 1 && face == 1
            return mesh.domain.boundary_names[1]
        elseif elem == nelements(mesh) && face == 2
            return mesh.domain.boundary_names[2]
        end
    end
    return nothing
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
the scale height (a typical atmospheric scale height `H ≈ 7.5`km).

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
