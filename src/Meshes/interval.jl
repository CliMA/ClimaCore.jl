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
                "We need to have dz_bottom <= z_max / z_elem and dz_top >= z_max / z_elem.",
            ),
        )
    end
    return :pass
end

abstract type StretchingRule end

"""
    UnknownStretch()

Placeholder stretching rule of an `IntervalMesh` constructed from explicit faces.
"""
struct UnknownStretch end

function IntervalMesh(domain::IntervalDomain, faces::AbstractArray)
    nelems = length(faces)
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

Convert the parameters of the stretching rule `stretch` to the floating point type `FT`
"""
function _convert_stretch end
_convert_stretch(::Type{FT}, stretch::Uniform) where {FT} = stretch

"""
    _faces_from_surface(domain, u_faces, reverse_mode)

Place faces at the heights `u_faces` above the surface, increasing from `0` to
the domain height. The surface is `coord_min` and heights count upward; in
`reverse_mode` the surface is `coord_max` and heights count downward, which
mirrors the profile about the domain midpoint. The end faces are pinned to the
domain bounds.
"""
function _faces_from_surface(
    domain::IntervalDomain,
    u_faces::AbstractVector{FT_solve},
    reverse_mode::Bool,
) where {FT_solve}
    z_min = FT_solve(Geometry.component(domain.coord_min, 1))
    z_max = FT_solve(Geometry.component(domain.coord_max, 1))
    faces = reverse_mode ? z_max .- reverse(u_faces) : z_min .+ u_faces
    faces[1] = z_min
    faces[end] = z_max
    return faces
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
the non-dimensional scale height. If `reverse_mode` is `true`, the smallest
element is at the top, and the largest at the bottom (this is typical for land
model configurations).

Construct a stretched mesh via

    IntervalMesh(interval_domain, ExponentialStretching(H); nelems, reverse_mode = false)

The resulting `faces` are reference heights without terrain warping.
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
    faces = CT.(_faces_from_surface(domain, u_faces, reverse_mode))
    monotonic_check(faces)
    IntervalMesh(stretch, domain, faces, nothing, reverse_mode)
end

"""
    GeneralizedExponentialStretching(dz_bottom::FT, dz_top::FT)

Apply a generalized form of exponential stretching to the domain when constructing
elements. `dz_bottom` and `dz_top` are the target element spacings at the bottom and
at the top of the vertical column domain [m]. In typical atmosphere configurations,
`dz_bottom` is the smallest spacing and `dz_top` the largest; in typical land
configurations, `dz_bottom` is the largest and `dz_top` the smallest. For land
configurations, use `reverse_mode = true` (default `false`).

Construct a generalized stretched mesh via

    IntervalMesh(
        interval_domain,
        GeneralizedExponentialStretching(dz_bottom, dz_top);
        nelems,
        FT_solve = Float64,
        tol = 1e-3,
        reverse_mode = false,
    )

where `FT_solve` is the float type and `tol` the residual tolerance of the root
solve for the stretching parameters. `nelems` must be at least 2. The resulting
`faces` are reference heights without terrain warping.
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
    tol = 1e-3,
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

    # define the inverse σ⁻¹ exponential stretching function
    exp_stretch(ζ, h) = ζ == 1 ? ζ : -h * log(1 - (1 - exp(-1 / h)) * ζ)

    # nondimensional vertical coordinate ([0.0, 1.0])
    ζ_n = LinRange(one(FT_solve), nelems, nelems) / nelems

    # find bottom height variation
    find_bottom(h) = dz_bottom - H * exp_stretch(ζ_n[1], h)
    # we use linearization
    # h_bottom ≈ -dz_bottom / H / log(1 - 1/nelems)
    # to approx bracket the lower / upper bounds of root sol
    guess₋ = -dz_bottom / H / log(1 - FT_solve(1 / (nelems - 1)))
    guess₊ = -dz_bottom / H / log(1 - FT_solve(1 / (nelems + 1)))
    h_bottom_sol = RootSolvers.find_zero(
        find_bottom,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT_solve(tol)),
    )
    if h_bottom_sol.converged !== true
        error(
            "h_bottom root failed to converge for dz_bottom: $dz_bottom on domain ($z_min, $z_max)",
        )
    end
    h_bottom = h_bottom_sol.root

    # find top height variation
    find_top(h) = dz_top - H * (1 - exp_stretch(ζ_n[end - 1], h))
    # we use the linearization
    # h_top ≈ (H - dz_top) / H / log(nelem)
    # to approx braket the lower, upper bounds of root sol
    guess₋ = (H - dz_top) / H / FT_solve(log(nelems + 1))
    guess₊ = (H - dz_top) / H / FT_solve(log(nelems - 1))
    h_top_sol = RootSolvers.find_zero(
        find_top,
        RootSolvers.SecantMethod(guess₋, guess₊),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT_solve(tol)),
    )
    if h_top_sol.converged !== true
        error(
            "h_top root failed to converge for dz_top: $dz_top on domain ($z_min, $z_max)",
        )
    end
    h_top = h_top_sol.root

    # scale height variation with height
    h =
        h_bottom .+
        (ζ_n .- ζ_n[1]) * (h_top - h_bottom) / (ζ_n[end - 1] - ζ_n[1])
    u_faces = FT_solve[zero(FT_solve); (H * exp_stretch.(ζ_n, h))...]
    faces = CT.(_faces_from_surface(domain, u_faces, reverse_mode))
    monotonic_check(faces)
    IntervalMesh(stretch, domain, faces, (; h_top), reverse_mode)
end

"""
    HyperbolicTangentStretching(dz_surface::FT)

Apply hyperbolic tangent stretching to the domain when constructing elements.
`dz_surface` is the target element spacing at the surface [m]: in typical
atmosphere configurations, the spacing at the bottom of the vertical column domain;
in typical land configurations, the spacing at the top.

For an interval ``[z_0, z_1]``, the elements are uniformly spaced in ``ζ``, where

```math
\\eta = 1 - \\frac{tanh[\\gamma(1-\\zeta)]}{tanh(\\gamma)},
```

where ``\\eta = \\frac{z - z_0}{z_1-z_0}``. The stretching parameter ``\\gamma``
is chosen to achieve a given resolution `dz_surface` at the surface.

Construct a stretched mesh via

    IntervalMesh(
        interval_domain,
        HyperbolicTangentStretching(dz_surface);
        nelems,
        FT_solve = Float64,
        tol = nothing,
        reverse_mode = false,
    )

where `FT_solve` is the float type and `tol` the residual tolerance of the root
solve for ``γ`` (default `1e-6 * dz_surface`, floored at ten ulps of the domain
height so that Float32 solves converge). `nelems` must be at least 2. For
land configurations, use `reverse_mode = true`. The resulting `faces` are reference
heights without terrain warping.
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
    # The residual is a height, and rounding in `H * tanh_stretch` is a few
    # ulps of `H`; the floor keeps the default reachable in Float32.
    if tol === nothing
        tol = max(dz_surface * FT_solve(1e-6), H * 10 * eps(FT_solve))
    end

    # define the hyperbolic tangent stretching function
    tanh_stretch(ζ, γ) = 1 - tanh(γ * (1 - ζ)) / tanh(γ)

    # nondimensional vertical coordinate ([0.0, 1.0])
    ζ_n = LinRange(one(FT_solve), nelems, nelems) / nelems

    # find the stretching parameter given the grid spacing at the surface
    find_surface(γ) = dz_surface - H * tanh_stretch(ζ_n[1], γ)
    γ_sol = RootSolvers.find_zero(
        find_surface,
        RootSolvers.NewtonsMethodAD(FT_solve(1.0)),
        RootSolvers.CompactSolution(),
        RootSolvers.ResidualTolerance(FT_solve(tol)),
    )
    if γ_sol.converged !== true
        error(
            "gamma root failed to converge for dz_surface: $dz_surface on domain ($z_min, $z_max)",
        )
    end

    u_faces = FT_solve[zero(FT_solve); (H * tanh_stretch.(ζ_n, γ_sol.root))...]
    faces = CT.(_faces_from_surface(domain, u_faces, reverse_mode))
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

For an interval ``[z_0, z_1]`` of height ``H = z_1 - z_0`` and height above the
surface ``u = z - z_0 \\in [0, H]`` with ``\\eta = u / H``, the cumulative level
index ``k(u) \\in [0, N]`` is the sum of a background hyperbolic tangent profile
and a boundary-layer hyperbolic tangent profile:

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
boundary-layer profile is exponentially small near the top, so the top element
spacing matches `dz_top` to within a relative error of order
``e^{-2H / z_{\\text{pbl}}}``. When ``N \\ge n_{\\text{bg}}(\\gamma)``,
``k'(u) > 0`` and ``k''(u) < 0`` for ``u \\in (0, H)``, so element spacings
increase monotonically from `dz_surface` at the surface to `dz_top` at the top.
The faces are the heights at which ``k`` takes integer values.

The number of elements is constrained on both sides: `nelems` must be at least
the number of levels of the single-scale [`HyperbolicTangentStretching`](@ref)
profile that matches both `dz_surface` and `dz_top`, and below the count at
which the boundary-layer profile alone places the first face below
`dz_surface`. The constructor throws an `ArgumentError` naming the admissible
range when `nelems` lies outside it.

Construct a stretched mesh via

    IntervalMesh(
        interval_domain,
        BoundaryLayerHyperbolicTangentStretching(dz_surface, dz_top, z_pbl);
        nelems,
        FT_solve = Float64,
        tol = nothing,
        reverse_mode = false,
    )

where `FT_solve` is the float type and `tol` the residual tolerance, in level
index units, of the root solves (default `sqrt(eps(FT_solve))`). For land
configurations, use `reverse_mode = true`: the profile is mirrored so that
`dz_surface` is the spacing at the top of the domain and `dz_top` the spacing
at the bottom. The resulting `faces` are reference heights without terrain
warping.
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

    function cumulative_level(u, γ)
        η = u / H
        n_bg = n_bg_of_gamma(γ)
        k_bg = n_bg * (one(FT_solve) - atanh((one(FT_solve) - η) * tanh(γ)) / γ)
        return k_bg + (N - n_bg) * (tanh(u / z_pbl) / th_pbl)
    end

    # The root `γ` is bracketed between the uniform limit `γ → 0` and `γ_max`,
    # at which the background profile alone already has surface spacing
    # `dz_surface`. With `n_bg` increasing in `γ`, `n_bg(γ) ≤ n_bg(γ_max) ≤ N`
    # on the bracket, which keeps the boundary-layer weight `N - n_bg`
    # nonnegative and the spacing monotonic.
    find_gamma(γ) = cumulative_level(dz_surface, γ) - one(FT_solve)
    η_surface = dz_surface / H
    η_top = dz_top / H
    τ_max = sqrt((η_top - η_surface) / ((one(FT_solve) - η_surface) * η_top))
    γ_min = FT_solve(1e-6)
    γ_max = atanh(τ_max)
    if !(find_gamma(γ_min) < zero(FT_solve) ≤ find_gamma(γ_max))
        # `find_gamma` is affine in `N` at fixed `γ`, so its zeros at the
        # bracket ends are the admissible range of `nelems`.
        c_pbl = tanh(dz_surface / z_pbl) / th_pbl
        n_min = n_bg_of_gamma(γ_max)
        n_max = N - find_gamma(γ_min) / c_pbl
        throw(
            ArgumentError(
                "BoundaryLayerHyperbolicTangentStretching with dz_surface = $dz_surface, dz_top = $dz_top, z_pbl = $z_pbl on a domain of height $H admits $(ceil(Int, n_min)) ≤ nelems < $(ceil(Int, n_max)); got nelems = $nelems",
            ),
        )
    end
    γ_sol = RootSolvers.find_zero(
        find_gamma,
        RootSolvers.BrentsMethod(γ_min, γ_max),
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
    for m in 1:(nelems - 1)
        target_m = FT_solve(m)
        find_face(u) = cumulative_level(u, γ) - target_m
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
    return IntervalMesh(
        stretch,
        domain,
        CT.(faces),
        (; γ_sol = γ),
        reverse_mode,
    )
end

"""
    truncate_mesh(parent_mesh::IntervalMesh, trunc_domain::IntervalDomain)

Construct an `IntervalMesh` on `trunc_domain` by truncating `parent_mesh`. The
result has as many elements as `parent_mesh` has faces below the top of
`trunc_domain`, and uses a `GeneralizedExponentialStretching` with the bottom and
top spacings of the truncated parent faces.
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
