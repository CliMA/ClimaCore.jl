"""
    IntervalMesh <: AbstractMesh

One-dimensional mesh on an `IntervalDomain`. Elements are numbered `1:nelems`.

# Fields

  - `stretch`: The stretching rule used to place the faces.
  - `domain`: The `IntervalDomain`.
  - `faces`: Vector of face coordinates, of length `nelems + 1`.
  - `meta`: Stretching-specific metadata (e.g. the solved stretching parameter), or
    `nothing`.
  - `reverse_mode`: Whether the smallest element is at the top of the domain.

# Constructor

    IntervalMesh(domain::IntervalDomain, faces::AbstractVector)

Construct a 1D mesh with face locations at `faces`.

    IntervalMesh(domain::IntervalDomain, stretching = Uniform(); nelems, reverse_mode = false)

Construct a 1D mesh on `domain` with `nelems` elements, using `stretching`. Possible
values of `stretching` are:

  - [`Uniform()`](@ref)
  - [`ExponentialStretching(H)`](@ref)
  - [`GeneralizedExponentialStretching(dz_bottom, dz_top)`](@ref)
  - [`HyperbolicTangentStretching(dz_surface)`](@ref)
  - [`BoundaryLayerHyperbolicTangentStretching(dz_surface, dz_top, z_pbl)`](@ref)

If `reverse_mode` is `true`, the smallest element is at the top, and the largest at the bottom
(this is typical for land model configurations).

# Examples

```julia
using ClimaCore: Domains, Geometry, Meshes
domain = Domains.IntervalDomain(
    Geometry.ZPoint(0.0),
    Geometry.ZPoint(30000.0);
    boundary_names = (:bottom, :top),
)
uniform_mesh = Meshes.IntervalMesh(domain; nelems = 30)
face_mesh = Meshes.IntervalMesh(domain, Geometry.ZPoint.([0.0, 100.0, 1000.0, 30000.0]))
```
"""
struct IntervalMesh{S, I <: IntervalDomain, V <: AbstractVector, M} <:
       AbstractMesh1D
    stretch::S
    domain::I
    faces::V
    meta::M
    reverse_mode::Bool
end

# implies isequal
Base.:(==)(mesh1::IntervalMesh, mesh2::IntervalMesh) =
    mesh1.domain == mesh2.domain && mesh1.faces == mesh2.faces
function Base.hash(mesh::IntervalMesh, h::UInt)
    h = hash(IntervalMesh, h)
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

function coordinates(mesh::IntervalMesh, elem::Integer, vert::Integer)
    check_vertex_index(vert, 2)
    return mesh.faces[elem + vert - 1]
end

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
                "A stretched mesh needs the surface spacing below (z_max - z_min) / nelems ",
                "and the spacing at the far end above it.",
            ),
        )
    end
    return :pass
end

"""
    StretchingRule

Abstract type of the rules that place the faces of an [`IntervalMesh`](@ref)
constructed from a number of elements. Each subtype `S` has a method
`IntervalMesh(domain, rule::S; nelems, kwargs...)` that builds the faces, and,
if it has parameters, a method `Meshes._convert_stretch(FT, rule::S)` that
converts them to the float type of the domain. Subtypes: [`Uniform`](@ref),
[`ExponentialStretching`](@ref), [`GeneralizedExponentialStretching`](@ref),
[`HyperbolicTangentStretching`](@ref), and
[`BoundaryLayerHyperbolicTangentStretching`](@ref).
"""
abstract type StretchingRule end

"""
    UnknownStretch()

Stretching rule recorded by an `IntervalMesh` constructed from explicit faces.
It is not a [`StretchingRule`](@ref), since no constructor builds faces from it.
"""
struct UnknownStretch end

function IntervalMesh(domain::IntervalDomain, faces::AbstractArray)
    nelems = length(faces) - 1
    nelems < 1 && throw(ArgumentError("`nelems` must be ≥ 1"))
    monotonic_check(faces)
    reverse_mode = false
    IntervalMesh(UnknownStretch(), domain, faces, nothing, reverse_mode)
end

"""
    Uniform()

Use uniformly-sized elements.
"""
struct Uniform <: StretchingRule end

"""
    Meshes._convert_stretch(FT, stretch)

Convert the parameters of the stretching rule `stretch` to the floating point type `FT`.
"""
function _convert_stretch end
_convert_stretch(::Type{FT}, stretch::Uniform) where {FT} = stretch

"""
    _faces_from_surface(domain, u_faces, reverse_mode)

Place faces at the heights `u_faces` above the surface, increasing from `0` to
the domain height. The surface is `coord_min` and heights count upward; in
`reverse_mode` the surface is `coord_max` and heights count downward, which
mirrors the profile about the domain midpoint. The faces are points of the
domain's coordinate type, and the end faces equal the domain bounds whatever
the float type of `u_faces`. Called from the [`StretchingRule`](@ref)
constructors of `IntervalMesh`.
"""
function _faces_from_surface(
    domain::IntervalDomain{CT},
    u_faces::AbstractVector,
    reverse_mode::Bool,
) where {CT}
    z_min = Geometry.component(domain.coord_min, 1)
    z_max = Geometry.component(domain.coord_max, 1)
    faces = reverse_mode ? z_max .- reverse(u_faces) : z_min .+ u_faces
    faces[1] = z_min
    faces[end] = z_max
    return CT.(faces)
end

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::Uniform = Uniform();
    nelems::Int,
    reverse_mode::Bool = false,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems < 1
        throw(ArgumentError("`nelems` must be ≥ 1"))
    end
    faces = range(domain.coord_min, domain.coord_max; length = nelems + 1)
    monotonic_check(faces)
    reverse_mode = false
    IntervalMesh(stretch, domain, faces, nothing, reverse_mode)
end


"""
    ExponentialStretching(H::FT)

Apply exponential stretching to the domain when constructing elements. `H` is
the scale height [m]; a typical atmospheric scale height is `H ≈ 7.5e3`.

For an interval ``[z_0, z_1]``, the elements are uniformly spaced in ``ζ``, where

```math
\\zeta = \\frac{1 - e^{-\\eta/h}}{1-e^{-1/h}},
```

where ``\\eta = \\frac{z - z_0}{z_1-z_0}``, and ``h = \\frac{H}{z_1-z_0}`` is
the nondimensional scale height. The resulting `faces` are reference heights
without terrain warping.

# Fields

  - `H`: Scale height [m].

# Keyword Arguments

Keyword arguments of `IntervalMesh(domain, ExponentialStretching(H); ...)`:

  - `nelems`: Number of elements.
  - `reverse_mode = false`: Whether the smallest element is at the top and the
    largest at the bottom, as in land model configurations.

# Examples

```julia
using ClimaCore: Domains, Geometry, Meshes
domain = Domains.IntervalDomain(
    Geometry.ZPoint(0.0),
    Geometry.ZPoint(30000.0);
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain, Meshes.ExponentialStretching(7500.0); nelems = 30)
```
"""
struct ExponentialStretching{FT} <: StretchingRule
    H::FT
end

_convert_stretch(::Type{FT}, stretch::ExponentialStretching) where {FT} =
    ExponentialStretching{FT}(stretch.H)

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::ExponentialStretching;
    nelems::Int,
    reverse_mode::Bool = false,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems < 1
        throw(ArgumentError("`nelems` must be ≥ 1"))
    end
    stretch = _convert_stretch(FT, stretch)
    cmin = Geometry.component(domain.coord_min, 1)
    cmax = Geometry.component(domain.coord_max, 1)
    R = cmax - cmin
    h = stretch.H / R

    η(ζ) = ζ == 1 ? ζ : (-h) * log1p((expm1(-1 / h)) * ζ)
    u_faces = [R * η(ζ) for ζ in range(FT(0), FT(1); length = nelems + 1)]
    faces = _faces_from_surface(domain, u_faces, reverse_mode)
    monotonic_check(faces)
    IntervalMesh(stretch, domain, faces, nothing, reverse_mode)
end

"""
    GeneralizedExponentialStretching(dz_bottom::FT, dz_top::FT)

Apply exponential stretching with a scale height that varies with height, so
that the elements have spacing `dz_bottom` at the bottom and `dz_top` at the top
of the domain [m].

For an interval ``[z_0, z_1]`` with ``N`` elements, the elements are uniformly
spaced in ``\\zeta``, where

```math
\\eta = -h(\\zeta) \\ln\\left[1 - \\left(1 - e^{-1/h(\\zeta)}\\right)\\zeta\\right]
```

and ``\\eta = \\frac{z - z_0}{z_1-z_0}``. The nondimensional scale height
``h(\\zeta)`` varies linearly from ``h_\\text{bottom}`` at ``\\zeta = 1/N`` to
``h_\\text{top}`` at ``\\zeta = 1 - 1/N``; both are solved so that the first and
last elements have the target spacings. With constant ``h``, this is
[`ExponentialStretching`](@ref). In typical atmosphere configurations,
`dz_bottom` is the smallest spacing and `dz_top` the largest. In typical land
configurations, `dz_bottom` is the largest and `dz_top` the smallest, and
`reverse_mode = true` mirrors the profile so that its fine end is at the top.
The resulting `faces` are reference heights without terrain warping.

# Fields

  - `dz_bottom`: Target element spacing at the bottom of the domain [m].
  - `dz_top`: Target element spacing at the top of the domain [m].

# Keyword Arguments

Keyword arguments of
`IntervalMesh(domain, GeneralizedExponentialStretching(dz_bottom, dz_top); ...)`:

  - `nelems`: Number of elements, at least 2.
  - `FT_solve = Float64`: Float type of the root solves for ``h_\\text{bottom}``
    and ``h_\\text{top}``.
  - `tol = sqrt(eps(FT_solve)) * min(dz_bottom, dz_top)`: Residual tolerance of
    the root solves, a spacing [m].
  - `reverse_mode = false`: Whether `dz_bottom ≥ dz_top`, with the fine end of
    the profile at the top, as in land model configurations.

# Examples

```julia
using ClimaCore: Domains, Geometry, Meshes
domain = Domains.IntervalDomain(
    Geometry.ZPoint(-10.0),
    Geometry.ZPoint(0.0);
    boundary_names = (:bottom, :top),
)
stretch = Meshes.GeneralizedExponentialStretching(1.5, 0.1)
mesh = Meshes.IntervalMesh(domain, stretch; nelems = 20, reverse_mode = true)
```
"""
struct GeneralizedExponentialStretching{FT} <: StretchingRule
    dz_bottom::FT
    dz_top::FT
end

# More specific than the generated `(::FT, ::FT) where {FT}`, so every `Real`
# pair lands here: call the parametric constructor rather than recursing.
function GeneralizedExponentialStretching(dz_bottom::Real, dz_top::Real)
    dz_bottom, dz_top = promote(dz_bottom, dz_top)
    return GeneralizedExponentialStretching{typeof(dz_bottom)}(
        dz_bottom,
        dz_top,
    )
end

_convert_stretch(
    ::Type{FT},
    stretch::GeneralizedExponentialStretching,
) where {FT} =
    GeneralizedExponentialStretching{FT}(stretch.dz_bottom, stretch.dz_top)

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::GeneralizedExponentialStretching;
    nelems::Int,
    FT_solve = Float64,
    tol::Union{Real, Nothing} = nothing,
    reverse_mode::Bool = false,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems ≤ 1
        throw(ArgumentError("`nelems` must be ≥ 2"))
    end
    stretch = _convert_stretch(FT, stretch)

    dz_bottom = FT_solve(stretch.dz_bottom)
    dz_top = FT_solve(stretch.dz_top)
    if !(dz_bottom ≤ dz_top) && !reverse_mode
        throw(ArgumentError("dz_bottom must be ≤ dz_top"))
    end

    if !(dz_bottom ≥ dz_top) && reverse_mode
        throw(ArgumentError("dz_top must be ≤ dz_bottom"))
    end

    # The solve runs from the surface upward with `dz_bottom` the surface
    # spacing; in `reverse_mode` the surface is the top, so the two spacings
    # swap and the profile is mirrored when the faces are placed.
    if reverse_mode
        dz_top, dz_bottom = dz_bottom, dz_top
    end
    z_min = Geometry.component(domain.coord_min, 1)
    z_max = Geometry.component(domain.coord_max, 1)
    H = FT_solve(z_max - z_min)
    # The residuals are spacings, so the tolerance scales with the finer one.
    # In a sweep of 12 configurations, Float32 secant solves reach the default
    # in all 12 and miss `1e-5 * min(dz_bottom, dz_top)` in 4.
    tol_solve =
        isnothing(tol) ? sqrt(eps(FT_solve)) * min(dz_bottom, dz_top) :
        FT_solve(tol)

    # The stretching function η(ζ, h) and its complement 1 - η. The complement
    # has its own formula because the subtraction 1 - η cancels near the top,
    # where the Float32 top solve then fails to converge.
    exp_stretch(ζ, h) = ζ == 1 ? ζ : -h * log1p(expm1(-1 / h) * ζ)
    exp_stretch_from_top(ζ, h) = h * log1p(expm1(1 / h) * (1 - ζ))

    # Faces above the surface in the nondimensional coordinate ζ ∈ (0, 1].
    ζ_n = LinRange(one(FT_solve), nelems, nelems) / nelems

    # For small h, η(1 / nelems, h) ≈ -h log(1 - 1 / nelems); the secant solve
    # starts from this estimate with nelems ∓ 1.
    find_bottom(h) = dz_bottom - H * exp_stretch(ζ_n[1], h)
    guess₋ = -dz_bottom / H / log(1 - FT_solve(1 / (nelems - 1)))
    guess₊ = -dz_bottom / H / log(1 - FT_solve(1 / (nelems + 1)))
    h_bottom_sol = RootSolvers.find_zero(
        find_bottom,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(tol_solve),
    )
    if h_bottom_sol.converged !== true
        error(
            "h_bottom root failed to converge for the surface spacing $dz_bottom on domain ($z_min, $z_max)",
        )
    end
    h_bottom = h_bottom_sol.root

    # For large 1 / h, 1 - η(1 - 1 / nelems, h) ≈ 1 - h log(nelems); the
    # secant solve starts from this estimate with nelems ± 1.
    find_top(h) = dz_top - H * exp_stretch_from_top(ζ_n[end - 1], h)
    guess₋ = (H - dz_top) / H / FT_solve(log(nelems + 1))
    guess₊ = (H - dz_top) / H / FT_solve(log(nelems - 1))
    h_top_sol = RootSolvers.find_zero(
        find_top,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(tol_solve),
    )
    if h_top_sol.converged !== true
        error(
            "h_top root failed to converge for the far-end spacing $dz_top on domain ($z_min, $z_max)",
        )
    end
    h_top = h_top_sol.root

    # h varies linearly in ζ between the two solved values.
    h =
        h_bottom .+
        (ζ_n .- ζ_n[1]) * (h_top - h_bottom) / (ζ_n[end - 1] - ζ_n[1])
    u_faces = FT_solve[zero(FT_solve); (H * exp_stretch.(ζ_n, h))...]
    faces = _faces_from_surface(domain, u_faces, reverse_mode)
    monotonic_check(faces)
    IntervalMesh(stretch, domain, faces, (; h_top), reverse_mode)
end

"""
    HyperbolicTangentStretching(dz_surface::FT)

Apply hyperbolic tangent stretching with element spacing `dz_surface` at the
surface [m]. The surface is the bottom of the domain, or the top with
`reverse_mode = true`, as in land model configurations.

For an interval ``[z_0, z_1]``, the elements are uniformly spaced in ``\\zeta``,
where

```math
\\eta = 1 - \\frac{\\tanh[\\gamma(1-\\zeta)]}{\\tanh(\\gamma)}
```

and ``\\eta = \\frac{z - z_0}{z_1-z_0}``. The stretching parameter
``\\gamma`` is solved so that the first element has spacing `dz_surface`, which
requires `dz_surface < (z_1 - z_0) / nelems`. The resulting `faces` are
reference heights without terrain warping.

# Fields

  - `dz_surface`: Target element spacing at the surface [m].

# Keyword Arguments

Keyword arguments of
`IntervalMesh(domain, HyperbolicTangentStretching(dz_surface); ...)`:

  - `nelems`: Number of elements, at least 2.
  - `FT_solve = Float64`: Float type of the root solve for ``\\gamma``.
  - `tol = 1e-6 * dz_surface`: Residual tolerance of the root solve, a spacing
    [m].
  - `reverse_mode = false`: Whether the surface, and with it the smallest
    element, is at the top, as in land model configurations.

# Examples

```julia
using ClimaCore: Domains, Geometry, Meshes
domain = Domains.IntervalDomain(
    Geometry.ZPoint(0.0),
    Geometry.ZPoint(30000.0);
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain, Meshes.HyperbolicTangentStretching(30.0); nelems = 43)
```
"""
struct HyperbolicTangentStretching{FT} <: StretchingRule
    dz_surface::FT
end

_convert_stretch(
    ::Type{FT},
    stretch::HyperbolicTangentStretching,
) where {FT} = HyperbolicTangentStretching{FT}(stretch.dz_surface)

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::HyperbolicTangentStretching;
    nelems::Int,
    FT_solve = Float64,
    tol::Union{Real, Nothing} = nothing,
    reverse_mode::Bool = false,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems ≤ 1
        throw(ArgumentError("`nelems` must be ≥ 2"))
    end
    stretch = _convert_stretch(FT, stretch)

    dz_surface = FT_solve(stretch.dz_surface)
    z_min = Geometry.component(domain.coord_min, 1)
    z_max = Geometry.component(domain.coord_max, 1)
    H = FT_solve(z_max - z_min)
    if !(zero(FT_solve) < dz_surface < H / nelems)
        throw(
            ArgumentError(
                "HyperbolicTangentStretching requires 0 < dz_surface < (z_max - z_min) / nelems; got dz_surface = $dz_surface on a domain of height $H with nelems = $nelems",
            ),
        )
    end
    tol_solve = isnothing(tol) ? FT_solve(1e-6) * dz_surface : FT_solve(tol)

    # The stretching function η = 1 - tanh(γ (1 - ζ)) / tanh(γ), rewritten
    # with tanh a - tanh b = sinh(a - b) / (cosh a cosh b) so that it has no
    # cancellation near the surface, where η ≪ 1. In Float32, the subtraction
    # there errs by up to 25% of the first spacing. The product of sinh and cosh
    # overflows for γ ≳ 45 in Float32 (≳ 355 in Float64), where η(1 / nelems)
    # is below Float32's normal range.
    tanh_stretch(ζ, γ) = sinh(γ * ζ) / (cosh(γ * (1 - ζ)) * sinh(γ))

    # Faces above the surface in the nondimensional coordinate ζ ∈ (0, 1].
    ζ_n = LinRange(one(FT_solve), nelems, nelems) / nelems

    find_surface(γ) = dz_surface - H * tanh_stretch(ζ_n[1], γ)
    γ_sol = RootSolvers.find_zero(
        find_surface,
        RootSolvers.NewtonsMethodAD(FT_solve(1.0)),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(tol_solve),
    )
    if γ_sol.converged !== true
        error(
            "gamma root failed to converge for dz_surface: $dz_surface on domain ($z_min, $z_max)",
        )
    end

    u_faces = FT_solve[zero(FT_solve); (H * tanh_stretch.(ζ_n, γ_sol.root))...]
    faces = _faces_from_surface(domain, u_faces, reverse_mode)
    monotonic_check(faces)
    IntervalMesh(
        stretch,
        domain,
        faces,
        (; γ_sol = γ_sol.root),
        reverse_mode,
    )
end

"""
    BoundaryLayerHyperbolicTangentStretching(dz_surface::FT, dz_top::FT, z_pbl::FT)

Apply two-scale hyperbolic tangent stretching that refines the boundary layer
with scale height `z_pbl` [m] while matching element spacing `dz_surface` at the
surface [m] and `dz_top` at the top of the domain [m].

For an interval ``[z_0, z_1]`` of height ``H = z_1 - z_0`` with ``N`` elements
and height above the surface ``u = z - z_0 \\in [0, H]`` with ``\\eta = u / H``,
the cumulative level index ``k(u) \\in [0, N]`` is the sum of a background
hyperbolic tangent profile and a boundary-layer hyperbolic tangent profile:

```math
k(u; \\gamma) = n_{\\text{bg}}(\\gamma)
  \\left[1 - \\frac{\\operatorname{atanh}[(1 - \\eta)\\tanh(\\gamma)]}
                  {\\gamma}\\right]
  + \\big(N - n_{\\text{bg}}(\\gamma)\\big)\\,
  \\frac{\\tanh(u / z_{\\text{pbl}})}{\\tanh(H / z_{\\text{pbl}})},
```

where
``n_{\\text{bg}}(\\gamma) = \\gamma / \\operatorname{atanh}[(\\text{dz\\_top} / H)\\tanh(\\gamma)]``
sets the top element spacing of the background profile to `dz_top`, and
``\\gamma > 0`` is solved so that ``k(\\text{dz\\_surface}; \\gamma) = 1``. The
faces are the heights at which ``k`` takes integer values, and the element
spacings increase monotonically from `dz_surface` to `dz_top`. The resulting
`faces` are reference heights without terrain warping.

`nelems` has a lower and an upper bound. Outside them, the constructor throws
an `ArgumentError` that names the admissible range.

# Fields

  - `dz_surface`: Target element spacing at the surface [m].
  - `dz_top`: Target element spacing at the far end of the domain [m].
  - `z_pbl`: Scale height of the boundary-layer refinement [m].

# Keyword Arguments

Keyword arguments of
`IntervalMesh(domain, BoundaryLayerHyperbolicTangentStretching(dz_surface, dz_top, z_pbl); ...)`:

  - `nelems`: Number of elements, within the admissible range.
  - `FT_solve = Float64`: Float type of the root solves.
  - `tol = sqrt(eps(FT_solve))`: Residual tolerance of the root solves for
    ``\\gamma`` and the faces, in level-index units [-].
  - `reverse_mode = false`: Whether the profile is mirrored so that
    `dz_surface` is the spacing at the top of the domain and `dz_top` the
    spacing at the bottom, as in land model configurations.

# Examples

```julia
using ClimaCore: Domains, Geometry, Meshes
domain = Domains.IntervalDomain(
    Geometry.ZPoint(0.0),
    Geometry.ZPoint(60000.0);
    boundary_names = (:bottom, :top),
)
stretch = Meshes.BoundaryLayerHyperbolicTangentStretching(28.0, 3000.0, 2200.0)
mesh = Meshes.IntervalMesh(domain, stretch; nelems = 75)
```

# Extended help

The lower bound on `nelems` is the number of levels of the single-scale
[`HyperbolicTangentStretching`](@ref) profile that matches both `dz_surface`
and `dz_top`; at it, the boundary-layer weight ``N - n_{\\text{bg}}`` is zero.
The upper bound is the count beyond which the boundary-layer profile, which
receives the levels the background profile does not, places the first face
below `dz_surface` for every ``\\gamma``. In between, up to two values of
``\\gamma`` satisfy the surface condition. The constructor takes the larger,
which assigns the most levels to the background profile.

Since ``n_{\\text{bg}}`` increases with ``\\gamma``, ``N \\ge n_{\\text{bg}}(\\gamma)``
holds for every admissible ``\\gamma``, so ``k'(u) > 0`` and ``k''(u) < 0`` on
``(0, H)`` and the spacings increase monotonically. The boundary-layer profile
is exponentially small near the top, so the top element spacing matches
`dz_top` to within a relative error of order ``e^{-2H / z_{\\text{pbl}}}``.
"""
struct BoundaryLayerHyperbolicTangentStretching{FT} <: StretchingRule
    dz_surface::FT
    dz_top::FT
    z_pbl::FT
end

# Promotes mixed `Real` arguments; see `GeneralizedExponentialStretching`.
function BoundaryLayerHyperbolicTangentStretching(
    dz_surface::Real,
    dz_top::Real,
    z_pbl::Real,
)
    dz_surface, dz_top, z_pbl = promote(dz_surface, dz_top, z_pbl)
    return BoundaryLayerHyperbolicTangentStretching{typeof(dz_surface)}(
        dz_surface,
        dz_top,
        z_pbl,
    )
end

_convert_stretch(
    ::Type{FT},
    stretch::BoundaryLayerHyperbolicTangentStretching,
) where {FT} = BoundaryLayerHyperbolicTangentStretching{FT}(
    stretch.dz_surface,
    stretch.dz_top,
    stretch.z_pbl,
)

"""
    _golden_section_argmin(f, a, b)

Minimizer of a unimodal function `f` on `[a, b]`, located by golden-section
search to a relative width of `sqrt(eps)`. The search brackets interior points
only, so the result is compared with `a` and `b`, where a monotonic `f` has its
minimum. Called from the [`BoundaryLayerHyperbolicTangentStretching`](@ref)
constructor.
"""
function _golden_section_argmin(f, a::FT, b::FT) where {FT}
    a₀, b₀ = a, b
    r = (sqrt(FT(5)) - 1) / 2
    x1 = b - r * (b - a)
    x2 = a + r * (b - a)
    f1, f2 = f(x1), f(x2)
    while b - a > sqrt(eps(FT)) * b
        if f1 ≤ f2
            b, x2, f2 = x2, x1, f1
            x1 = b - r * (b - a)
            f1 = f(x1)
        else
            a, x1, f1 = x1, x2, f2
            x2 = a + r * (b - a)
            f2 = f(x2)
        end
    end
    x, fx = f1 ≤ f2 ? (x1, f1) : (x2, f2)
    f(a₀) < fx && return a₀
    f(b₀) < fx && return b₀
    return x
end

function IntervalMesh(
    domain::IntervalDomain{CT},
    stretch::BoundaryLayerHyperbolicTangentStretching;
    nelems::Int,
    FT_solve = Float64,
    tol::Union{Real, Nothing} = nothing,
    reverse_mode::Bool = false,
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    if nelems ≤ 1
        throw(ArgumentError("`nelems` must be ≥ 2"))
    end
    stretch = _convert_stretch(FT, stretch)

    dz_surface = FT_solve(stretch.dz_surface)
    dz_top = FT_solve(stretch.dz_top)
    z_pbl = FT_solve(stretch.z_pbl)
    # The residuals are level indices, so the tolerance is absolute; `sqrt(eps)`
    # is reachable in Float32 and places faces to within ~1e-8 dz in Float64.
    tol_solve = isnothing(tol) ? sqrt(eps(FT_solve)) : FT_solve(tol)

    z_min = FT_solve(Geometry.component(domain.coord_min, 1))
    z_max = FT_solve(Geometry.component(domain.coord_max, 1))
    H = z_max - z_min
    if !(
        zero(FT_solve) < dz_surface < H / nelems < dz_top < H &&
        z_pbl > zero(FT_solve)
    )
        throw(
            ArgumentError(
                "BoundaryLayerHyperbolicTangentStretching requires 0 < dz_surface < (z_max - z_min) / nelems < dz_top < z_max - z_min and z_pbl > 0",
            ),
        )
    end

    N = FT_solve(nelems)
    th_pbl = tanh(H / z_pbl)
    n_bg_of_gamma(γ) = γ / atanh((dz_top / H) * tanh(γ))

    # The background level index
    # n_bg [atanh(tanh γ) - atanh((1 - η) tanh γ)] / γ is written as a
    # difference of logarithms. The direct form subtracts two nearly equal
    # numbers near the surface and errs by up to 1e-3 levels in Float32, above
    # the default tolerance. `expm1(2γ)` overflows for γ ≳ 44 in Float32
    # (≳ 354 in Float64), which `γ_max` reaches only for surface-to-top
    # spacing ratios below ~1e-38 (~1e-307). The coefficients that depend on
    # `γ` alone are computed once per `γ`.
    level_coeffs(γ) = (n_bg_of_gamma(γ), expm1(2 * γ) / 2, expm1(-2 * γ) / 2)
    function cumulative_level(u, γ, coeffs = level_coeffs(γ))
        n_bg, a₊, a₋ = coeffs
        η = u / H
        k_bg = n_bg * (log1p(η * a₊) - log1p(η * a₋)) / (2 * γ)
        return k_bg + (N - n_bg) * (tanh(u / z_pbl) / th_pbl)
    end

    # The root `γ` lies between the uniform limit `γ → 0` and `γ_max`, at
    # which the background profile alone already has surface spacing
    # `dz_surface`. With `n_bg` increasing in `γ`, `n_bg(γ) ≤ n_bg(γ_max) ≤ N`
    # on that interval, which keeps the boundary-layer weight `N - n_bg`
    # nonnegative and the spacing monotonic.
    find_gamma(γ) = cumulative_level(dz_surface, γ) - one(FT_solve)
    # `γ_max = atanh(τ_max) = log(1 + τ_max) - log(1 - τ_max²) / 2`, with
    # 1 - τ_max² = `δ_max` in closed form. `τ_max` itself rounds to 1, and
    # `atanh(τ_max)` to infinity, for surface-to-top spacing ratios below
    # ~6e-8 in Float32.
    η_surface = dz_surface / H
    η_top = dz_top / H
    τ_max = sqrt((η_top - η_surface) / ((one(FT_solve) - η_surface) * η_top))
    δ_max = η_surface * (one(FT_solve) - η_top) / ((one(FT_solve) - η_surface) * η_top)
    γ_min = FT_solve(1e-6)
    γ_max = log1p(τ_max) - log(δ_max) / 2
    # `find_gamma` first decreases and then increases in `γ` (in all 2880
    # configurations of a parameter sweep), so it can have two roots. The
    # solve brackets the root on the increasing branch, between the minimizer
    # `γ_low` and `γ_max`; it is the larger root, which assigns the most levels
    # to the background profile.
    γ_low = _golden_section_argmin(find_gamma, γ_min, γ_max)
    if !(find_gamma(γ_low) ≤ zero(FT_solve) ≤ find_gamma(γ_max))
        # `find_gamma` is `N c_pbl` plus a function of `γ` alone, so its
        # minimizer is independent of `N`, and its zeros at `γ_low` and
        # `γ_max` bound the admissible range of `nelems`.
        c_pbl = tanh(dz_surface / z_pbl) / th_pbl
        n_lo = ceil(Int, n_bg_of_gamma(γ_max))
        n_hi = floor(Int, N - find_gamma(γ_low) / c_pbl)
        admissible =
            n_lo ≤ n_hi ? "$n_lo ≤ nelems ≤ $n_hi" : "no value of nelems"
        throw(
            ArgumentError(
                "BoundaryLayerHyperbolicTangentStretching with dz_surface = $dz_surface, dz_top = $dz_top, z_pbl = $z_pbl on a domain of height $H admits $admissible; got nelems = $nelems",
            ),
        )
    end
    γ_sol = RootSolvers.find_zero(
        find_gamma,
        RootSolvers.BrentsMethod(γ_low, γ_max),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(tol_solve),
    )
    if γ_sol.converged !== true
        error(
            "gamma root failed to converge for dz_surface: $dz_surface on domain ($z_min, $z_max)",
        )
    end
    γ = γ_sol.root

    # Face heights above the surface; `k(u)` is increasing, so each solve is
    # bracketed by the previous face and the domain top.
    u_faces = Vector{FT_solve}(undef, nelems + 1)
    u_faces[1] = zero(FT_solve)
    u_faces[nelems + 1] = H
    coeffs = level_coeffs(γ)
    for m in 1:(nelems - 1)
        target_m = FT_solve(m)
        find_face(u) = cumulative_level(u, γ, coeffs) - target_m
        u_sol = RootSolvers.find_zero(
            find_face,
            RootSolvers.BrentsMethod(u_faces[m], H),
            RootSolvers.CompactSolution(),
            RootSolvers.ResidualTolerance(tol_solve),
        )
        if u_sol.converged !== true
            error(
                "face $m root failed to converge on domain ($z_min, $z_max)",
            )
        end
        u_faces[m + 1] = u_sol.root
    end

    faces = _faces_from_surface(domain, u_faces, reverse_mode)
    monotonic_check(faces)
    return IntervalMesh(stretch, domain, faces, (; γ_sol = γ), reverse_mode)
end

"""
    truncate_mesh(parent_mesh::IntervalMesh, trunc_domain::IntervalDomain)

Construct an `IntervalMesh` on `trunc_domain` by truncating `parent_mesh`. The
truncated parent faces run from the last parent face at or below the bottom of
`trunc_domain` to the first at or above its top. The result has as many
elements as these faces span, and uses a `GeneralizedExponentialStretching`
with their bottom and top spacings; `reverse_mode` is set when the bottom
spacing is the larger, as in land model meshes.
"""
function truncate_mesh(
    parent_mesh::IntervalMesh,
    trunc_domain::IntervalDomain{CT},
) where {CT <: Geometry.Abstract1DPoint{FT}} where {FT}
    faces = parent_mesh.faces
    z_bottom = trunc_domain.coord_min
    z_top = trunc_domain.coord_max
    k_bottom = something(findlast(face -> face.z ≤ z_bottom.z, faces), 1)
    k_top = something(findfirst(face -> face.z ≥ z_top.z, faces), length(faces))

    trunc_faces = faces[k_bottom:k_top]
    new_nelems = length(trunc_faces) - 1

    Δz_top = trunc_faces[end].z - trunc_faces[end - 1].z
    Δz_bottom = trunc_faces[2].z - trunc_faces[1].z

    new_stretch = GeneralizedExponentialStretching(Δz_bottom, Δz_top)
    new_domain = IntervalDomain(
        z_bottom,
        Geometry.ZPoint{FT}(z_top),
        boundary_names = trunc_domain.boundary_names,
    )
    return IntervalMesh(
        new_domain,
        new_stretch;
        nelems = new_nelems,
        reverse_mode = Δz_bottom > Δz_top,
    )
end
