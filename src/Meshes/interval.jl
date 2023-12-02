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
- [`HyperbolicTangentStretching(dz_bottom)`](@ref)
"""
struct IntervalMesh{I <: IntervalDomain, V <: AbstractVector} <: AbstractMesh1D
    domain::I
    faces::V
end

# implies isequal
Base.:(==)(mesh1::IntervalMesh, mesh2::IntervalMesh) =
    mesh1.domain == mesh2.domain && mesh1.faces == mesh2.faces
function Base.hash(mesh::IntervalMesh, h::UInt)
    h = hash(Meshes.IntervalMesh, h)
    h = hash(mesh.domain, h)
    h = hash(mesh.faces, h)
    return h
end
domain(mesh::IntervalMesh) = mesh.domain
nelements(mesh::IntervalMesh) = length(mesh.faces) - 1
elements(mesh::IntervalMesh) = Base.OneTo(nelements(mesh))

function Base.summary(io::IO, mesh::IntervalMesh)
    print(io, nelements(mesh), "-element IntervalMesh")
end
function Base.show(io::IO, mesh::IntervalMesh)
    summary(io, mesh)
    print(io, " of ")
    print(io, mesh.domain)
end
function element_horizontal_length_scale(mesh::IntervalMesh)
    cmax = Geometry.component(mesh.domain.coord_max, 1)
    cmin = Geometry.component(mesh.domain.coord_min, 1)
    return (cmax - cmin) / nelements(mesh)
end

coordinates(mesh::IntervalMesh, elem::Integer, vert::Integer) =
    mesh.faces[elem + vert - 1]

function coordinates(
    mesh::IntervalMesh,
    elem::Integer,
    (ξ1,)::Union{StaticArrays.SVector{1}, Tuple{<:Real}},
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

monotonic_check(
    faces::Union{
        LinRange{<:Geometry.AbstractPoint},
        Vector{<:Geometry.AbstractPoint},
    },
) = :no_check

function monotonic_check(
    faces::Union{
        LinRange{<:Geometry.ZPoint},
        LinRange{<:Real},
        Vector{<:Geometry.ZPoint},
        Vector{<:Real},
    },
)
    n = length(faces) - 1
    if eltype(faces) <: Geometry.AbstractPoint
        monotonic_incr = all(i -> faces[i].z < faces[i + 1].z, 1:n)
        monotonic_decr = all(i -> faces[i].z > faces[i + 1].z, 1:n)
    else
        monotonic_incr = all(i -> faces[i] < faces[i + 1], 1:n)
        monotonic_decr = all(i -> faces[i] > faces[i + 1], 1:n)
    end
    if !(monotonic_incr || monotonic_decr)
        error(
            string(
                "Faces in vertical mesh are not increasing or decreasing monotonically. ",
                "We need to have dz_bottom <= z_max / z_elem and dz_top >= z_max / z_elem.",
            ),
        )
    end
    return :pass
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

`faces` contain reference z without any warping.
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
        faces = map(f -> eltype(faces)(-f), faces)
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

`faces` contain reference z without any warping.
"""
struct GeneralizedExponentialStretching{FT} <: StretchingRule
    dz_bottom::FT
    dz_top::FT
end

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::GeneralizedExponentialStretching{FT};
    nelems::Int,
    FT_solve = Float64,
    tol = 1e-3,
    reverse_mode::Bool = false,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems ≤ 1
        throw(ArgumentError("`nelems` must be ≥ 2"))
    end

    dz_bottom = FT_solve(stretch.dz_bottom)
    dz_top = FT_solve(stretch.dz_top)
    if !(dz_bottom ≤ dz_top) && !reverse_mode
        throw(ArgumentError("dz_bottom must be ≤ dz_top"))
    end

    if !(dz_bottom ≥ dz_top) && reverse_mode
        throw(ArgumentError("dz_top must be ≤ dz_bottom"))
    end

    # bottom coord height value is always min and top coord height value is always max
    # since the vertical coordinate is positive upward
    z_bottom = Geometry.component(domain.coord_min, 1)
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

    # nondimensional vertical coordinate ([0.0, 1.0])
    ζ_n = LinRange(one(FT_solve), nelems, nelems) / nelems

    # find bottom height variation
    find_bottom(h) = dz_bottom - z_top * exp_stretch(ζ_n[1], h)
    # we use linearization
    # h_bottom ≈ -dz_bottom / (z_top - z_bottom) / log(1 - 1/nelems)
    # to approx bracket the lower / upper bounds of root sol
    guess₋ =
        -dz_bottom / (z_top - z_bottom) / log(1 - FT_solve(1 / (nelems - 1)))
    guess₊ =
        -dz_bottom / (z_top - z_bottom) / log(1 - FT_solve(1 / (nelems + 1)))
    h_bottom_sol = RootSolvers.find_zero(
        find_bottom,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT_solve(tol)),
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
    guess₋ = ((z_top - z_bottom) - dz_top) / z_top / FT_solve(log(nelems + 1))
    guess₊ = ((z_top - z_bottom) - dz_top) / z_top / FT_solve(log(nelems - 1))
    h_top_sol = RootSolvers.find_zero(
        find_top,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT_solve(tol)),
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
    faces = z_bottom .+ (z_top - z_bottom) * exp_stretch.(ζ_n, h)

    # add the bottom level
    faces = FT_solve[z_bottom; faces...]
    if reverse_mode
        reverse!(faces)
        faces = map(f -> eltype(faces)(-f), faces)
        faces[end] = faces[end] == -z_bottom ? z_bottom : faces[1]
    end
    monotonic_check(faces)
    IntervalMesh(domain, CT.(faces))
end

"""
    HyperbolicTangentStretching(dz_surface::FT)

Apply a hyperbolic tangent stretching to the domain when constructing elements.
`dz_surface` is the target element grid spacing at the surface. In typical atmosphere
configuration, it is the grid spacing at the bottom of the
vertical column domain (m). On the other hand, for typical land configurations,
it is the grid spacing at the top of the vertical column domain.

For an interval ``[z_0,z_1]``, this makes the elements uniformally spaced in
``\\zeta``, where
```math
\\eta = 1 - \\frac{tanh[\\gamma(1-\\zeta)]}{tanh(\\gamma)},
```
where ``\\eta = \\frac{z - z_0}{z_1-z_0}``. The stretching parameter ``\\gamma``
is chosen to achieve a given resolution `dz_surface` at the surface. 

Then, the user can define a stretched mesh via

    ClimaCore.Meshes.IntervalMesh(interval_domain, HyperbolicTangentStretching(dz_surface); nelems::Int, reverse_mode)

`reverse_mode` is default to false for atmosphere configurations. For land configurations, 
use `reverse_mode` = `true`.

`faces` contain reference z without any warping.
"""
struct HyperbolicTangentStretching{FT} <: StretchingRule
    dz_surface::FT
end

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::HyperbolicTangentStretching{FT};
    nelems::Int,
    FT_solve = Float64,
    tol::Union{FT, Nothing} = nothing,
    reverse_mode::Bool = false,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems ≤ 1
        throw(ArgumentError("`nelems` must be ≥ 2"))
    end

    dz_surface = FT_solve(stretch.dz_surface)
    tol === nothing && (tol = dz_surface * FT_solve(1e-6))

    # bottom coord height value is always min and top coord height value is always max
    # since the vertical coordinate is positive upward
    z_bottom = Geometry.component(domain.coord_min, 1)
    z_top = Geometry.component(domain.coord_max, 1)
    # but in case of reverse_mode, we temporarily swap them
    # so that the following root solve algorithm does not need to change
    if reverse_mode
        z_bottom, z_top = Geometry.component(domain.coord_max, 1),
        -Geometry.component(domain.coord_min, 1)
    end

    # define the hyperbolic tangent stretching function
    tanh_stretch(ζ, γ) = 1 - tanh(γ * (1 - ζ)) / tanh(γ)

    # nondimensional vertical coordinate ([0.0, 1.0])
    ζ_n = LinRange(one(FT_solve), nelems, nelems) / nelems

    # find the stretching parameter given the grid spacing at the surface
    find_surface(γ) = dz_surface - z_top * tanh_stretch(ζ_n[1], γ)
    γ_sol = RootSolvers.find_zero(
        find_surface,
        RootSolvers.NewtonsMethodAD(FT_solve(1.0)),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT_solve(tol)),
    )
    if γ_sol.converged !== true
        error(
            "gamma root failed to converge for dz_surface: $dz_surface on domain ($z_bottom, $z_top)",
        )
    end

    faces = z_bottom .+ (z_top - z_bottom) * tanh_stretch.(ζ_n, γ_sol.root)

    # add the bottom level
    faces = FT_solve[z_bottom; faces...]
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

## aliases

const XIntervalMesh = IntervalMesh{<:Domains.XIntervalDomain}
function XIntervalMesh(;
    x_min::Real, x_max::Real, x_periodic::Bool=false, x_boundary_names=(:west, :east),
    x_elem::Integer, x_stretch=Uniform(),
)
    domain = Domains.XIntervalDomain(; x_min, x_max, x_periodic, x_boundary_names)
    mesh = IntervalMesh(domain, x_stretch; nelem=x_elem)
end

const ZIntervalMesh = IntervalMesh{<:Domains.ZIntervalDomain}
function ZIntervalMesh(;
    z_min::Real, z_max::Real, z_periodic::Bool=false, z_boundary_names=(:bottom, :top),
    z_elem::Integer, z_stretch=Uniform(),
)
    domain = Domains.ZIntervalDomain(; z_min, z_max, z_periodic, z_boundary_names)
    mesh = IntervalMesh(domain, z_stretch; nelem=z_elem)
end
