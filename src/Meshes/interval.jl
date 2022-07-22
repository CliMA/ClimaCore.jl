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
- [`GeneralizedExponentialStretching(dz_surface, dz_top)`](@ref)
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
    nelems::Int,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems < 1
        throw(ArgumentError("`nelems` must be ≥ 1"))
    end
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
    IntervalMesh(domain, faces)
end

"""
    GeneralizedExponentialStretching(dz_surface, dz_top)

Apply a generalized form of exponential stretching to the domain when constructing elements.
`dz_surface` and `dz_top` are target element grid spacings at surface and at the top of the
vertical column domain (m).
"""
struct GeneralizedExponentialStretching{FT} <: StretchingRule
    dz_surface::FT
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
    dz_surface, dz_top = stretch.dz_surface, stretch.dz_top
    if !(dz_surface ≤ dz_top)
        throw(ArgumentError("dz_surface must be ≤ dz_top"))
    end
    # surface coord height value
    zₛ = Geometry.component(domain.coord_min, 1)
    # top coord height value
    zₜ = Geometry.component(domain.coord_max, 1)

    # define the inverse σ⁻¹ exponential stretching function
    exp_stretch(ζ, h) = -h * log(1 - (1 - exp(-1 / h)) * ζ)

    # nondimensional vertical coordinate ([0.0, 1.0])
    ζ_n = LinRange(one(FT), nelems, nelems) / nelems

    # find surface height variation
    find_surface(h) = dz_surface - zₜ * exp_stretch(ζ_n[1], h)
    # we use linearization
    # hₛ ≈ -dz_surface / zₜ / log(1 - 1/nelems)
    # to approx bracket the lower / upper bounds of root sol
    guess₋ = -dz_surface / zₜ / log(1 - FT(1 / (nelems - 1)))
    guess₊ = -dz_surface / zₜ / log(1 - FT(1 / (nelems + 1)))
    hₛsol = RootSolvers.find_zero(
        find_surface,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT(1e-3)),
    )
    if hₛsol.converged !== true
        error(
            "hₛ root failed to converge for dz_surface: $dz_surface on domain ($zₛ, $zₜ)",
        )
    end
    hₛ = hₛsol.root

    # find top height variation
    find_top(h) = dz_top - zₜ * (1 - exp_stretch(ζ_n[end - 1], h))
    # we use the linearization
    # hₜ ≈ (zₜ - dz_top) / zₜ / log(nelem)
    # to approx braket the lower, upper bounds of root sol
    guess₋ = ((zₜ - zₛ) - dz_top) / zₜ / FT(log(nelems + 1))
    guess₊ = ((zₜ - zₛ) - dz_top) / zₜ / FT(log(nelems - 1))
    hₜsol = RootSolvers.find_zero(
        find_top,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT(1e-3)),
    )
    if hₜsol.converged !== true
        error(
            "hₜ root failed to converge for dz_top: $dz_surface on domain ($zₛ, $zₜ)",
        )
    end
    hₜ = hₜsol.root

    # scale height variation with height
    h = hₛ .+ (ζ_n .- ζ_n[1]) * (hₜ - hₛ) / (ζ_n[end - 1] - ζ_n[1])
    faces = (zₛ + (zₜ - zₛ)) * exp_stretch.(ζ_n, h)

    # add the bottom level
    faces = [zₛ; faces...]
    IntervalMesh(domain, CT.(faces))
end

"""
    TruncatedIntervalMesh(
        domain::IntervalDomain{CT},
        stretch::GeneralizedExponentialStretching{FT};
        nelems::Int,
        z_top::FT,
    )

Constructs an `IntervalMesh`, truncating the given `domain` exactly at `z_top`.
The truncation preserves the number of degrees of freedom covering the space
from the `:bottom` to `z_top`, adjusting the stretching so that `:top` is at `z_top`. 
"""
function TruncatedIntervalMesh(
    domain::IntervalDomain{CT},
    stretch::GeneralizedExponentialStretching{FT};
    nelems::Int,
    z_top::FT,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}

    parent_mesh = IntervalMesh(domain, stretch; nelems = nelems)
    # Get approximate top
    faces = parent_mesh.faces
    k_top = length(faces)
    for (k_f, face) in enumerate(faces)
        if face.z ≥ z_top
            k_top = k_f
            break
        end
    end

    z₀ = faces[1]
    z_approx_top = faces[k_top]
    trunc_faces = faces[1:k_top]
    new_nelems = length(trunc_faces) - 1

    Δz_top = trunc_faces[end].z - trunc_faces[end - 1].z
    Δz_surf = trunc_faces[2].z - trunc_faces[1].z

    new_stretch = GeneralizedExponentialStretching(Δz_surf, Δz_top)
    new_domain = IntervalDomain(
        z₀,
        Geometry.ZPoint{FT}(z_top),
        boundary_tags = (:bottom, :top),
    )
    return IntervalMesh(new_domain, new_stretch; nelems = new_nelems)
end
