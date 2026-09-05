# Christoffel-free Cartesian tensor divergence, shared by CG and DG.
#
# On a curved space the divergence of a rank-2 flux tensor is
#
#     (∇·T)ⁱ = (1/J) ∂_ξʲ(J Tʲⁱ) + Γⁱ_jk Tʲᵏ,
#
# the weak `Divergence` contracting the transport axis `j` (first) and carrying
# the momentum axis `i` (second) through. It computes only the first term; the
# connection term `Γⁱ_jk Tʲᵏ` on the momentum index vanishes when that axis is
# written in the spatially-constant global Cartesian basis, making the operator
# exact (Vinokur 1974).
#
# Only the interface coupling differs between discretizations, which is what an
# `AbstractTendencyCompletion` selects (DSS for CG, numerical flux for DG). Both
# are applied while the momentum axis is still Cartesian — face increments are
# Cartesian vectors and DSS of Cartesian components is a plain weighted sum in
# one global basis — so the two sides of a shared node meet in the same frame;
# the rotation back to local happens only afterwards.

"""
    cartesian_tensor_divergence(T, completion, faceargs...)

Weak-form horizontal divergence `∇ₕ·T` of a rank-2 flux tensor field `T` (e.g.
the momentum flux `ρu⊗u`). On a curved space `∇·T` carries a Christoffel
connection term on `T`'s momentum axis that the weak `Divergence` omits;
rotating that axis into the global Cartesian basis, where the Christoffel
symbols vanish, makes the omission exact. So the momentum (second) axis is
rotated to Cartesian (`Geometry.CartesianTensor`), `Divergence` is applied, and
the result rotated back to the local frame (`Geometry.LocalVector`); on a plane
(`CartesianGlobalGeometry`) both rotations are the identity and are skipped.

Like the spectral `Divergence` it is built from, this differentiates along the
two horizontal directions only, so on an extruded space the vertical flux
divergence is a separate term. That term drops the same connection terms, so
rotate its flux tensor to Cartesian as well before differencing it, as in the
example below.

The result is a 3D `Geometry.UVWVector`, whose `w` component on the sphere is
the curvature term — `-|u|²/R` for `u⊗u` under solid-body rotation, the same
order as the tangential components. A model carrying horizontal-only momentum
has to drop it, e.g. with
`Geometry.project(Geometry.UVAxis(), ...)`.

`T`'s momentum (second) axis must be a local orthonormal axis; a `UVAxis` is
read as a `UVWAxis` with `w == 0`. On a discontinuous space the transport
(first) axis must be orthonormal too, because `numflux` contracts it against
the local orthonormal face normal.

`completion` (from [`tendency_completion`](@ref)) couples element interfaces —
DSS on a CG space, the numerical flux on a DG one, the same CG↔DG switch as
[`complete_tendency!`](@ref). On a DG space `faceargs...` are the trailing
arguments of the completion's `numflux` (the Cartesian tensor is its first face
argument, and the flux must return the momentum vector it produces from that
tensor, whose `UVW` components are global Cartesian); on a CG space they
are unused, as is the completion's DSS buffer — the buffer this needs is one
for its own `UVWVector` result, which it owns — so a model can pass the
completion it built for its own tendency.

Allocates the result, and the Cartesian-tensor scratch wherever the rotation is
not the identity; [`cartesian_tensor_divergence!`](@ref) takes both buffers and
is allocation-free.

# Examples

The whole `∇·T` on an extruded space, the vertical flux divergence carrying the
same rotation. `ᶜT` is the flux tensor on cell centers and `ᶠT` the same flux
on faces:

```julia
geom = Spaces.global_geometry(axes(ᶜT))
ᶜcoords = Fields.coordinate_field(axes(ᶜT))
ᶠcoords = Fields.coordinate_field(axes(ᶠT))

ᶜdivₕ = Operators.cartesian_tensor_divergence(ᶜT, completion)

ᶠTc = @. Geometry.CartesianTensor(ᶠT, geom, ᶠcoords)
ᶜdivᵥ = Operators.DivergenceF2C().(ᶠTc)

ᶜdivT = @. ᶜdivₕ + Geometry.LocalVector(ᶜdivᵥ, geom, ᶜcoords)
```

Differencing the local-frame `ᶠT` leaves the connection terms of the vertical
derivative in place, an error that on a topographic sphere runs larger than the
divergence itself (1.8x its peak at Ne = 4, GLL{4}, 10 levels). The two terms
are rotated back one at a time here, which agrees with rotating their sum to
roundoff, the rotation being linear.

## References

  - [Vinokur1974](@cite): Cartesian momentum components remove the connection
    terms.
"""
function cartesian_tensor_divergence(T, completion, faceargs...)
    space = axes(T)
    FT = Spaces.undertype(space)
    rotation = _momentum_rotation(space)
    # `_cartesian_momentum` allocates `Tc` already rotated, so this path runs
    # the three steps itself; routing it through
    # `cartesian_tensor_divergence!` would rotate a second time.
    Tc = _cartesian_momentum(T, rotation)
    out = Fields.Field(Geometry.UVWVector{FT}, space)
    return _tensor_divergence_completed!(out, Tc, rotation, completion, faceargs...)
end

"""
    cartesian_tensor_divergence!(out, Tc, T, completion, faceargs...)

In-place [`cartesian_tensor_divergence`](@ref): writes the divergence into the
local-frame vector field `out`, using the rank-2 field `Tc` as scratch for the
Cartesian-rotated flux. Neither `out` nor `Tc` may alias `T`. `Tc` is untouched
on a plane, where the rotation is the identity and `T` is differentiated
directly. Returns `out`.

Rotating the momentum axis widens it to the full 3D `UVWAxis`, so `similar(T)`
is a wide enough scratch only when `T`'s momentum axis already is, and a
narrower `Tc` raises an error. Given a `UVAxis` momentum, allocate `Tc` from
the rotation, as the allocating form does, or promote the momentum vector
before forming the flux.

Allocation-free given the two buffers.
"""
function cartesian_tensor_divergence!(out, Tc, T, completion, faceargs...)
    rotation = _momentum_rotation(axes(T))
    Tcart = _cartesian_momentum!(Tc, T, rotation)
    return _tensor_divergence_completed!(out, Tcart, rotation, completion, faceargs...)
end

# The momentum-axis rotation as a field of matrices on the horizontal space,
# or `nothing` where it is the identity. `local_to_cartesian` reads latitude
# and longitude alone, so on an extruded space the rotation lives on
# `horizontal_space(space)` and holds `N_z` times fewer entries than the field
# it multiplies: 0.42 MiB against 12.66 MiB at Ne = 8, GLL{4}, 30 levels,
# Float64.
#
# It is memoized because `local_to_cartesian` costs four `sind`/`cosd` per node
# and the operator needs it twice per call: a CG call at that resolution takes
# 8.9 ms with the cache and 17.7 ms without. The nine numbers per node are
# released by `Utilities.Cache.clean_cache!()`. `clean_cache!(grid)` does not
# release them: it filters on the cached value, and the grid appears here in the
# key, so a sweep over many grids has to call the no-argument form (the same
# holds for `dg_connectivity` and `_momentum_dss_buffer`). The key omits the
# global geometry, which holds while `local_to_cartesian` resolves to a single
# method for spherical geometries; a geometry whose local vertical departs from
# the radial direction would need its own key.
#
# `@inline` so that `horizontal_space` folds into the caller: left as a call it
# allocates its 16 B wrapper on an extruded space, which the zero-allocation
# gate in allocs_spectral_ops.jl catches.
@inline _momentum_rotation(space) =
    _momentum_rotation(Spaces.global_geometry(space), Spaces.horizontal_space(space))

# On a plane the local orthonormal frame is the global Cartesian frame.
_momentum_rotation(::Geometry.CartesianGlobalGeometry, hspace) = nothing

function _momentum_rotation(
    global_geometry::Geometry.AbstractSphericalGlobalGeometry,
    hspace,
)
    coords = Fields.coordinate_field(hspace)
    RT = Utilities.return_type(
        Geometry.local_to_cartesian,
        Tuple{typeof(global_geometry), eltype(coords)},
    )
    rotation = get!(
        Cache.OBJECT_CACHE,
        (:MomentumRotation, Spaces.grid(hspace), typeof(hspace)),
    ) do
        field = Fields.Field(RT, hspace)
        field .= Geometry.local_to_cartesian.(Ref(global_geometry), coords)
        return field
    end
    # The assertion recovers the concrete field type lost through the untyped
    # cache, so the caller keeps a statically known type.
    return rotation::Utilities.return_type(
        Fields.Field,
        Tuple{Type{RT}, typeof(hspace)},
    )
end

# `G` rotates local→Cartesian, so post-multiplying by `G'` rotates each row's
# momentum vector the same way ((T*G')[i,j] = Σₖ T[i,k] G[j,k]), and `G'` on
# the left inverts it on the result vector.
@inline _rotate_momentum(t, G) = t * adjoint(G)

_cartesian_momentum(T, ::Nothing) = T
_cartesian_momentum(T, rotation) = @. _rotate_momentum(T, rotation)

_cartesian_momentum!(Tc, T, ::Nothing) = T
function _cartesian_momentum!(Tc, T, rotation)
    # The rotation widens a `UVAxis` momentum to the full `UVWAxis`, and
    # assigning the wider tensor into a narrower field drops the Cartesian `w`
    # column and returns a wrong answer — off by 36% of the peak value for a
    # `UVVector` ⊗ `UVVector` flux at Ne = 3 — so the scratch eltype is checked
    # here. The types are compile-time constants, so the comparison folds away
    # and the message is built only on the error path.
    rotated = Utilities.return_type(
        _rotate_momentum,
        Tuple{eltype(T), eltype(rotation)},
    )
    eltype(Tc) === rotated || error(
        "cartesian_tensor_divergence! needs a scratch field of eltype \
         $rotated for a flux tensor of eltype $(eltype(T)), but got \
         $(eltype(Tc)); `similar(T)` is wide enough only when the momentum \
         axis of `T` is already the 3D UVWAxis",
    )
    @. Tc = _rotate_momentum(T, rotation)
    return Tc
end

_local_momentum!(out, ::Nothing) = out
function _local_momentum!(out, rotation)
    @. out = adjoint(rotation) * out
    return out
end

# DG: the weak volume term on a mass-weighted (`WJ`) residual, completed at
# element faces by `numflux` (and the one-sided boundary flux, when given).
# The `WJ` normalization and the rotation back to the local frame share one
# broadcast, so the tail costs one kernel launch and one pass over `out`.
# `faceargs::Vararg{Any, N}` for the same specialization reason as the face
# operators it calls into (see `add_numerical_flux_interior!`).
function _tensor_divergence_completed!(
    out,
    Tc,
    rotation,
    completion::NumericalFluxCompletion,
    faceargs::Vararg{Any, N},
) where {N}
    lgeom = Fields.local_geometry_field(axes(Tc))
    wdiv = Divergence{WeakForm}()
    @. out = wdiv(Tc) * (-lgeom.WJ)
    add_numerical_flux_interior!(completion.numflux, out, Tc, faceargs...)
    isnothing(completion.boundary_numflux) || add_numerical_flux_boundary!(
        completion.boundary_numflux,
        out,
        Tc,
        faceargs...,
    )
    if isnothing(rotation)
        @. out = -out / lgeom.WJ
    else
        @. out = adjoint(rotation) * (-out / lgeom.WJ)
    end
    return out
end

# CG: the weak volume term assembled across elements by DSS. `weighted_dss!`
# leaves orthonormal vectors unprojected (a plain weighted sum), and the
# momentum components are still in the one global Cartesian basis here, so the
# two sides of a shared node are summed in the same frame — the rotation to
# local follows the DSS. `faceargs` are unused: a continuous space imposes
# interface coupling by DSS, not fluxes.
function _tensor_divergence_completed!(
    out,
    Tc,
    rotation,
    completion::DSSCompletion,
    faceargs...,
)
    wdiv = Divergence{WeakForm}()
    @. out = wdiv(Tc)
    Spaces.weighted_dss!(out, _momentum_dss_buffer(out))
    return _local_momentum!(out, rotation)
end

# The DSS buffer for the divergence result, memoized on the grid like the
# rotation. It is owned here because a `DSSCompletion` carries a buffer sized
# for the model's tendency, whose eltype is in general not the `UVWVector` this
# operator assembles, and `weighted_dss!` on a mismatched buffer errors inside
# the DSS transform.
function _momentum_dss_buffer(out)
    space = axes(out)
    buffer = get!(
        () -> Spaces.create_dss_buffer(out),
        Cache.OBJECT_CACHE,
        (:MomentumDSSBuffer, Spaces.grid(space), typeof(space), eltype(out)),
    )
    return buffer::Utilities.return_type(
        Spaces.create_dss_buffer,
        Tuple{typeof(out)},
    )
end
