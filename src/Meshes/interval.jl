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
                "Faces in vertical mesh are not increasing or decreasing monotonically. ",
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
    ExponentialStretching(H::FT)

Apply exponential stretching to the domain when constructing elements. `H` is
the scale height (a typical atmospheric scale height `H ≈ 7.5`km).

For an interval ``[z_0,z_1]``, this makes the elements uniformally spaced in
``\\zeta``, where
```math
\\zeta = \\frac{1 - e^{-\\eta/h}}{1-e^{-1/h}},
```
where ``\\eta = \\frac{z - z_0}{z_1-z_0}``, and ``h = \\frac{H}{z_1-z_0}`` is
the non-dimensional scale height. If `reverse_mode` is `true`, the smallest
element is at the top, and the largest at the bottom (this is typical for land
model configurations).

Then, the user can define a stretched mesh via

    ClimaCore.Meshes.IntervalMesh(interval_domain, ExponentialStretching(H); nelems::Int, reverse_mode = false)
"""
struct ExponentialStretching{FT} <: StretchingRule
    H::FT
end

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::ExponentialStretching{FT};
    nelems::Int,
    reverse_mode::Bool = false,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems < 1
        throw(ArgumentError("`nelems` must be ≥ 1"))
    end
    cmin = Geometry.component(domain.coord_min, 1)
    cmax = Geometry.component(domain.coord_max, 1)
    R = cmax - cmin
    h = stretch.H / R

    η(ζ) = ζ == 1 ? ζ : (-h) * log1p((expm1(-1 / h)) * ζ)
    faces = [
        CT(reverse_mode ? cmax + R * η(ζ) : cmin + R * η(ζ)) for
        ζ in range(FT(0), FT(1); length = nelems + 1)
    ]

    if reverse_mode
        faces = map(f -> eltype(faces)(-f.z), faces)
        faces[1] = faces[1] == -cmax ? cmax : faces[1]
        reverse!(faces)
    end
    monotonic_check(faces)
    IntervalMesh(domain, faces)
end

"""
    GeneralizedExponentialStretching(dz_bottom::FT, dz_top::FT)

Apply a generalized form of exponential stretching to the domain when constructing elements.
`dz_bottom` and `dz_top` are target element grid spacings at the bottom and at the top of the
vertical column domain (m). In typical atmosphere configurations, `dz_bottom` is the smallest
grid spacing and `dz_top` the largest one. On the other hand, for typical land configurations,
`dz_bottom` is the largest grid spacing and `dz_top` the smallest one.

For land configurations, use `reverse_mode` = `true` (default value `false`).

Then, the user can define a generalized stretched mesh via

    ClimaCore.Meshes.IntervalMesh(interval_domain, GeneralizedExponentialStretching(dz_bottom, dz_top); nelems::Int, reverse_mode = false)
"""
struct GeneralizedExponentialStretching{FT} <: StretchingRule
    dz_bottom::FT
    dz_top::FT
end

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::GeneralizedExponentialStretching{FT};
    nelems::Int,
    reverse_mode::Bool = false,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems ≤ 1
        throw(ArgumentError("`nelems` must be ≥ 2"))
    end

    dz_bottom, dz_top = stretch.dz_bottom, stretch.dz_top
    if !(dz_bottom ≤ dz_top) && !reverse_mode
        throw(ArgumentError("dz_bottom must be ≤ dz_top"))
    end

    if !(dz_bottom ≥ dz_top) && reverse_mode
        throw(ArgumentError("dz_top must be ≤ dz_bottom"))
    end

    # bottom coord height value, always min, for both atmos and land, since z-axis does not change
    z_bottom = Geometry.component(domain.coord_min, 1)
    # top coord height value, always max, for both atmos and land, since z-axis does not change
    z_top = Geometry.component(domain.coord_max, 1)
    # but in case of reverse_mode, we temporarily swap them together with dz_bottom and dz_top
    # so that the following root solve algorithm does not need to change
    if reverse_mode
        z_bottom, z_top = Geometry.component(domain.coord_max, 1),
        -Geometry.component(domain.coord_min, 1)
        dz_top, dz_bottom = dz_bottom, dz_top
    end

    # define the inverse σ⁻¹ exponential stretching function
    exp_stretch(ζ, h) = ζ == 1 ? ζ : -h * log(1 - (1 - exp(-1 / h)) * ζ)

    # nondimensional vertical coordinate (]0.0, 1.0])
    ζ_n = LinRange(one(FT), nelems, nelems) / nelems

    # find bottom height variation
    find_bottom(h) = dz_bottom - z_top * exp_stretch(ζ_n[1], h)
    # we use linearization
    # h_bottom ≈ -dz_bottom / (z_top - z_bottom) / log(1 - 1/nelems)
    # to approx bracket the lower / upper bounds of root sol
    guess₋ = -dz_bottom / (z_top - z_bottom) / log(1 - FT(1 / (nelems - 1)))
    guess₊ = -dz_bottom / (z_top - z_bottom) / log(1 - FT(1 / (nelems + 1)))
    h_bottom_sol = RootSolvers.find_zero(
        find_bottom,
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
    if reverse_mode
        reverse!(faces)
        faces = map(f -> eltype(faces)(-f), faces)
        faces[end] = faces[end] == -z_bottom ? z_bottom : faces[1]
    end
    monotonic_check(faces)
    IntervalMesh(domain, CT.(faces))
end


"""
    truncate_mesh(
        parent_mesh::AbstractMesh,
        trunc_domain::IntervalDomain{CT},
    )
Constructs an `IntervalMesh`, truncating the given `parent_mesh` defined on a
truncated `trunc_domain`. The truncation preserves the number of
degrees of freedom covering the space from the `trunc_domain`'s `z_bottom` to `z_top`,
adjusting the stretching.
"""
function truncate_mesh(
    parent_mesh::IntervalMesh,
    trunc_domain::IntervalDomain{CT},
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}

    # Get approximate top
    faces = parent_mesh.faces
    k_top = length(faces)
    z_top = trunc_domain.coord_max
    z_bottom = trunc_domain.coord_min

    for (k_f, face) in enumerate(faces)
        if face.z ≥ z_top.z
            k_top = k_f
            break
        end
    end

    trunc_faces = faces[1:k_top]
    new_nelems = length(trunc_faces) - 1

    Δz_top = trunc_faces[end].z - trunc_faces[end - 1].z
    Δz_bottom = trunc_faces[2].z - trunc_faces[1].z

    new_stretch = GeneralizedExponentialStretching(Δz_bottom, Δz_top)
    new_domain = IntervalDomain(
        z_bottom,
        Geometry.ZPoint{FT}(z_top),
        boundary_names = trunc_domain.boundary_names,
    )
    return IntervalMesh(new_domain, new_stretch; nelems = new_nelems)
end
