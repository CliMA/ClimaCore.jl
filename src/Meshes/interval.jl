"""
    IntervalMesh <: AbstractMesh

A 1D mesh on an `IntervalDomain`.

# Constuctors

    IntervalMesh(domain::IntervalDomain, faces::AbstractVector)

Construct a 1D mesh with face locations at `faces`.

    IntervalMesh(domain::IntervalDomain[, stretching=Uniform()]; nelems=)

Constuct a 1D mesh on `domain` with `nelems` elements, using `stretching`. Possible values of `stretching` are:

- [`Uniform()`](@ref)
- [`ExponentialStretching(H)`](@ref)
- [`GeneralizedExponentialStretching(dz_bottom, dz_top)`](@ref)
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

function monotonic_check(faces)
    if !(hasproperty(first(faces), :z) || eltype(faces) <: Real)
        return nothing
    end
    z(face::AbstractFloat) = face
    z(face::Geometry.AbstractPoint) = face.z
    n = length(faces) - 1
    monotonic_incr = all(map(i -> z(faces[i]) < z(faces[i + 1]), 1:n))
    monotonic_decr = all(map(i -> z(faces[i]) > z(faces[i + 1]), 1:n))
    if !(monotonic_incr || monotonic_decr)
        error(
            string(
                "Faces in vertical mesh are not increasing monotonically. ",
                "We need to have dz_bottom <= z_max / z_elem and dz_top >= z_max / z_elem.",
            ),
        )
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
    nelems::Int,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems < 1
        throw(ArgumentError("`nelems` must be ≥ 1"))
    end
    faces = range(domain.coord_min, domain.coord_max; length = nelems + 1)
    monotonic_check(faces)
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
    stretch::ExponentialStretching{FT};
    nelems::Int,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems < 1
        throw(ArgumentError("`nelems` must be ≥ 1"))
    end
    cmin = Geometry.component(domain.coord_min, 1)
    cmax = Geometry.component(domain.coord_max, 1)
    R = cmax - cmin
    h = stretch.H / R
    η(ζ) = ζ == 1 ? ζ : -h * log1p((expm1(-1 / h)) * ζ)
    faces =
        [CT(cmin + R * η(ζ)) for ζ in range(FT(0), FT(1); length = nelems + 1)]
    monotonic_check(faces)
    IntervalMesh(domain, faces)
end

"""
    GeneralizedExponentialStretching(dz_bottom, dz_top)

Apply a generalized form of exponential stretching to the domain when constructing elements.
`dz_bottom` and `dz_top` are target element grid spacings at surface and at the top of the
vertical column domain (m).
"""
struct GeneralizedExponentialStretching{FT} <: StretchingRule
    dz_bottom::FT
    dz_top::FT
end

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::GeneralizedExponentialStretching{FT};
    nelems::Int,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems ≤ 1
        throw(ArgumentError("`nelems` must be ≥ 2"))
    end
    dz_bottom, dz_top = stretch.dz_bottom, stretch.dz_top
    if !(dz_bottom ≤ dz_top)
        throw(ArgumentError("dz_bottom must be ≤ dz_top"))
    end
    # surface coord height value
    z_bottom = Geometry.component(domain.coord_min, 1)
    # top coord height value
    z_top = Geometry.component(domain.coord_max, 1)

    # define the inverse σ⁻¹ exponential stretching function
    exp_stretch(ζ, h) = -h * log(1 - (1 - exp(-1 / h)) * ζ)

    # nondimensional vertical coordinate ([0.0, 1.0])
    ζ_n = LinRange(one(FT), nelems, nelems) / nelems

    # find surface height variation
    find_surface(h) = dz_bottom - z_top * exp_stretch(ζ_n[1], h)
    # we use linearization
    # h_bottom ≈ -dz_bottom / z_top / log(1 - 1/nelems)
    # to approx bracket the lower / upper bounds of root sol
    guess₋ = -dz_bottom / z_top / log(1 - FT(1 / (nelems - 1)))
    guess₊ = -dz_bottom / z_top / log(1 - FT(1 / (nelems + 1)))
    h_bottom_sol = RootSolvers.find_zero(
        find_surface,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT(1e-3)),
    )
    if h_bottom_sol.converged !== true
        error(
            "h_bottom root failed to converge for dz_bottom: $dz_bottom on domain ($z_bottom, $z_top)",
        )
    end
    h_bottom = h_bottom_sol.root

    # find top height variation
    find_top(h) = dz_top - z_top * (1 - exp_stretch(ζ_n[end - 1], h))
    # we use the linearization
    # h_top ≈ (z_top - dz_top) / z_top / log(nelem)
    # to approx braket the lower, upper bounds of root sol
    guess₋ = ((z_top - z_bottom) - dz_top) / z_top / FT(log(nelems + 1))
    guess₊ = ((z_top - z_bottom) - dz_top) / z_top / FT(log(nelems - 1))
    h_top_sol = RootSolvers.find_zero(
        find_top,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT(1e-3)),
    )
    if h_top_sol.converged !== true
        error(
            "h_top root failed to converge for dz_top: $dz_top on domain ($z_bottom, $z_top)",
        )
    end
    h_top = h_top_sol.root

    # scale height variation with height
    h =
        h_bottom .+
        (ζ_n .- ζ_n[1]) * (h_top - h_bottom) / (ζ_n[end - 1] - ζ_n[1])
    faces = (z_bottom + (z_top - z_bottom)) * exp_stretch.(ζ_n, h)

    # add the bottom level
    faces = [z_bottom; faces...]
    monotonic_check(faces)
    IntervalMesh(domain, CT.(faces))
end
