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

Weak-form divergence `∇·T` of a rank-2 flux tensor field `T` (e.g. the momentum
flux `ρu⊗u`). On a curved space `∇·T` carries a Christoffel connection term on
`T`'s momentum axis that the weak `Divergence` omits; rotating that axis into the
global Cartesian basis, where the Christoffel symbols vanish, makes the omission
exact. So the momentum (second) axis is rotated to Cartesian
(`Geometry.CartesianTensor`), `Divergence` is applied, and the result rotated
back to the local frame (`Geometry.LocalVector`); on a plane
(`CartesianGlobalGeometry`) both rotations are the identity.

`T`'s momentum axis must be the full 3D `UVWAxis`; on a 2D horizontal shell
promote it first, e.g. `(ρu) ⊗ Geometry.project(Geometry.UVWAxis(), u)`.

`completion` (from [`tendency_completion`](@ref)) couples element interfaces —
DSS on a CG space, the numerical flux on a DG one, the same CG↔DG switch as
[`complete_tendency!`](@ref). On a DG space `faceargs...` are the trailing
arguments of the completion's `numflux` (the Cartesian tensor is its first face
argument, and the flux must return a `Cartesian123Vector`); on a CG space they
are unused.

Allocates the result and a Cartesian-tensor scratch field;
[`cartesian_tensor_divergence!`](@ref) takes both and is allocation-free.

## References

  - [Vinokur1974](@cite): Cartesian momentum components remove the connection
    terms.
"""
function cartesian_tensor_divergence(T, completion, faceargs...)
    space = axes(T)
    coords = Fields.coordinate_field(space)
    FT = Spaces.undertype(space)
    global_geom = Spaces.global_geometry(space)
    Tc = @. Geometry.CartesianTensor(T, global_geom, coords)
    out = Fields.Field(Geometry.UVWVector{FT}, space)
    return cartesian_tensor_divergence!(out, Tc, T, completion, faceargs...)
end

"""
    cartesian_tensor_divergence!(out, Tc, T, completion, faceargs...)

In-place [`cartesian_tensor_divergence`](@ref): writes the divergence into the
local-frame vector field `out`, using the rank-2 field `Tc` as scratch for the
Cartesian-rotated flux. Neither `out` nor `Tc` may alias `T`. Returns `out`.

Allocation-free given the two buffers.
"""
function cartesian_tensor_divergence!(out, Tc, T, completion, faceargs...)
    space = axes(T)
    coords = Fields.coordinate_field(space)
    global_geom = Spaces.global_geometry(space)
    @. Tc = Geometry.CartesianTensor(T, global_geom, coords)
    # Complete the interfaces in the Cartesian frame, then rotate to local.
    _tensor_divergence_interior!(out, Tc, completion, faceargs...)
    @. out = Geometry.LocalVector(out, global_geom, coords)
    return out
end

# DG: the weak volume term on a mass-weighted (`WJ`) residual, completed at
# element faces by `numflux` (and the one-sided boundary flux, when given).
# `faceargs::Vararg{Any, N}` for the same specialization reason as the face
# operators it calls into (see `add_numerical_flux_interior!`).
function _tensor_divergence_interior!(
    out,
    Tc,
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
    @. out = -out / lgeom.WJ
    return out
end

# CG: the weak volume term assembled across elements by DSS. `weighted_dss!`
# leaves orthonormal vectors unprojected (a plain weighted sum), and the
# momentum components are still in the one global Cartesian basis here, so the
# two sides of every shared node are summed in the same frame — the rotation to
# local happens only afterwards, in `cartesian_tensor_divergence!`. `faceargs`
# are unused: a continuous space imposes interface coupling by DSS, not fluxes.
function _tensor_divergence_interior!(
    out,
    Tc,
    completion::DSSCompletion,
    faceargs...,
)
    wdiv = Divergence{WeakForm}()
    @. out = wdiv(Tc)
    Spaces.weighted_dss!(out, completion.buffer)
    return out
end
