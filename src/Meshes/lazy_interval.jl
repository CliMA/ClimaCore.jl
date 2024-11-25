"""
    LazyIntervalMesh <: AbstractMesh

A 1D mesh on an `IntervalDomain`.

# Constuctors

Construct a 1D mesh with face locations at `faces`.

# Example

```julia
vmesh = IntervalMesh(domain::IntervalDomain; stretching=Uniform(); nelems=5)
end

Constuct a 1D mesh on `domain` with `nelems` elements, using `stretching`. Possible values of `stretching` are:

- [`Uniform()`](@ref)
- [`ExponentialStretching(H)`](@ref)
- [`GeneralizedExponentialStretching(dz_bottom, dz_top)`](@ref)
- [`HyperbolicTangentStretching(dz_bottom)`](@ref)
"""
struct LazyIntervalMesh{I <: IntervalDomain, V, S, FT, R} <: AbstractMesh1D
    domain::I
    faces::V
    stretch::S
    solution_root::FT # solution to nonlinear solve (if one exists)
    z_bottom::FT
    z_top::FT
    nelems::Int
end
reverse(::LazyIntervalMesh{I <: IntervalDomain, V, S, FT, R}) where {I, V, S, FT, R} = R
float_type(::LazyIntervalMesh{I <: IntervalDomain, V, S, FT, R}) where {I, V, S, FT, R} = FT
ζ_n(::Type{FT}, n) where {FT} = LinRange(one(FT), n, n) / n
tanh_stretch(ζ, γ) = 1 - tanh(γ * (1 - ζ)) / tanh(γ)

Base.@propagate_inbounds faces(lim::LazyIntervalMesh, i::Integer) =
    faces(lim, lim.domain, i)

Base.@propagate_inbounds faces(lim::LazyIntervalMesh, ::IntervalDomain{CT}, i::Integer) where {CT} =
    CT(lim.z_bottom + (lim.z_top - lim.z_bottom) * tanh_stretch(ζ_n(float_type(lim), lim.nelems)[i], lim.solution_root))

function LazyIntervalMesh(
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

    # nondimensional vertical coordinate ([0.0, 1.0])
    ζ_n = LinRange(one(FT_solve), nelems, nelems) / nelems

    # find the stretching parameter given the grid spacing at the surface
    find_surface(γ) = dz_surface - z_top * tanh_stretch(one(FT_solve), γ)
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

    R = reverse_mode
    I = typeof(domain)
    V = typeof(faces)
    S = typeof(stretch)
    # monotonic_check(faces)
    return LazyIntervalMesh{I, V, S, FT_solve, R}(domain, faces, stretch, solution_root, z_bottom, z_top, nelems)
    faces = z_bottom .+ (z_top - z_bottom) * tanh_stretch.(ζ_n, γ_sol.root)

    # add the bottom level
    faces = FT_solve[z_bottom; faces...]
    if reverse_mode
        reverse!(faces)
        faces = map(f -> eltype(faces)(-f), faces)
        faces[end] = faces[end] == -z_bottom ? z_bottom : faces[1]
    end
    IntervalMesh(domain, CT.(faces))
end
