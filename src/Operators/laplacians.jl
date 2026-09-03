# Horizontal Laplacian atoms shared by CG and DG discretizations. These are
# the building blocks of ∇⁴ hyperdiffusion: two Laplacian passes with a
# `Spaces.weighted_dss!` of the intermediate fields between them. Owning the
# atoms here lets the continuity mechanics differ by discretization — DSS of
# the intermediate on CG, face terms coupling neighbouring elements inside each
# pass on DG — while callers use one calling sequence for both.

"""
    scalar_laplacian(χ; weight = nothing)

Weak horizontal Laplacian of a scalar, `wdivₕ(gradₕ(χ))`, or
`wdivₕ(weight * gradₕ(χ))` when a `weight` field (e.g., a density) is given.

On continuous (CG) spaces this returns a lazy operator expression, so it fuses
into consuming broadcasts; the result is element-local, and materialized
intermediates must be made continuous with [`Spaces.weighted_dss!`](@ref)
before they are differentiated again (batch several intermediates into one
call to share the ghost exchange). On discontinuous (DG) spaces, this returns a
materialized `Field` that already includes the face terms coupling neighbouring
elements (see [`sipg_laplacian_tendency!`](@ref)), and `weighted_dss!` is a
no-op — so the same prep → `weighted_dss!` → apply sequence is correct for
both discretizations.

Each call owns its result on both discretizations, so any number of results
may be live at once. The DG result is a fresh `Field`; in a tendency, where
that allocation is not wanted, use [`scalar_laplacian!`](@ref) to write into a
caller-owned field instead.

# Examples

Fourth-order hyperdiffusion `∇⁴χ` on either discretization:

```julia
∇²χ = similar(χ)
Operators.scalar_laplacian!(∇²χ, χ)
Spaces.weighted_dss!(∇²χ)          # no-op on DG
@. dydt -= ν * Operators.scalar_laplacian(∇²χ)
```
"""
scalar_laplacian(χ; weight = nothing) =
    scalar_laplacian(Spaces.discretization(axes(χ)), χ, weight)

function scalar_laplacian(::Grids.CG, χ, weight)
    wdiv = Divergence{WeakForm}()
    grad = Gradient()
    return isnothing(weight) ? lazy.(wdiv.(grad.(χ))) :
           lazy.(wdiv.(weight .* grad.(χ)))
end

function scalar_laplacian(::Grids.DG, χ, weight)
    q = _dg_scalar_argument(axes(χ), χ)
    return scalar_laplacian!(Grids.DG(), similar(q), q, weight)
end

"""
    scalar_laplacian!(out, χ; weight = nothing)

Write [`scalar_laplacian`](@ref)`(χ; weight)` into the caller-owned field
`out` and return `out`. Allocation-free on both discretizations, so this is
the form to use in a tendency. `out` may alias `χ`: the argument is copied
into scratch first, since the spectral operators read a whole element while
writing it. `out` may not alias `weight`.
"""
scalar_laplacian!(out, χ; weight = nothing) =
    scalar_laplacian!(Spaces.discretization(axes(χ)), out, χ, weight)

function scalar_laplacian!(::Grids.CG, out, χ, weight)
    out === χ && (χ = _aliased_argument_copy(out))
    out .= scalar_laplacian(Grids.CG(), χ, weight)
    return out
end

function scalar_laplacian!(::Grids.DG, out, χ, weight)
    space = axes(χ)
    q = _dg_scalar_argument(space, χ)
    q === out && (q = _aliased_argument_copy(out))
    T = eltype(q)
    FT = Spaces.undertype(space)
    κ = one(FT)
    G_uv = _laplacian_scratch_field(
        space,
        Geometry.UVVector{T},
        :scalar_laplacian_gradient,
    )
    R_uv = _laplacian_scratch_field(
        space,
        Geometry.UVVector{T},
        :scalar_laplacian_lifting,
    )
    # τ carries the same `weight` as the flux it balances, so the two scale
    # together (see `sipg_penalty_parameter`).
    τ = _laplacian_scratch_field(space, FT, :scalar_laplacian_penalty)
    sipg_penalty_parameter!(τ, κ; weight)
    return sipg_laplacian_tendency!(out, G_uv, R_uv, q, weight, κ, τ)
end

# An argument that aliases the destination, copied into scratch so the
# operators can read it while writing `out`. Scratch rather than `copy`, so
# that the in-place forms stay allocation-free even when aliased.
function _aliased_argument_copy(out)
    space = axes(out)
    q = _laplacian_scratch_field(
        space,
        eltype(out),
        :scalar_laplacian_aliased_argument,
    )
    q .= out
    return q
end

# Fields pass through; a lazy argument is materialized into a scratch field,
# since the face kernels index their arguments at element-boundary nodes and
# so need stored values.
_dg_scalar_argument(space, χ::Fields.Field) = χ
function _dg_scalar_argument(space, χ)
    q = _laplacian_scratch_field(
        space,
        Spaces.undertype(space),
        :scalar_laplacian_argument,
    )
    q .= χ
    return q
end

# Memoized scratch Fields for the DG Laplacian, keyed on the space's grid.
# `tag` separates the fields one call uses in turn. Like the DG staging
# buffers, a scratch field is shared by every call on the same space and
# eltype, so its contents are only valid until the next call. The grid is part
# of the key rather than the cached value, so `Cache.clean_cache!(grid)` does
# not release these entries; only the zero-argument `clean_cache!()` does. The
# type assertion recovers the concrete field type lost through the untyped
# cache, so the caller dispatches statically instead of boxing its arguments.
function _laplacian_scratch_field(space, ::Type{T}, tag::Symbol) where {T}
    field = get!(
        () -> Fields.Field(T, space),
        Cache.OBJECT_CACHE,
        (:LaplacianScratchField, tag, Spaces.grid(space), typeof(space), T),
    )
    return field::Utilities.return_type(
        Fields.Field,
        Tuple{Type{T}, typeof(space)},
    )
end

"""
    vector_laplacian(u; divergence_factor = 1)

Weak horizontal vector Laplacian of a horizontal covariant vector, as the
grad-div minus curl-curl identity
`divergence_factor * wgradₕ(divₕ(u)) − C12(wcurlₕ(C3(curlₕ(u))))`, with the
grad-div part scaled by `divergence_factor` (used for divergence damping in
the second pass of ∇⁴ hyperdiffusion). On spaces with one horizontal
dimension the curl-curl part vanishes identically and only the grad-div part
is built.

Returns a lazy operator expression on continuous (CG) spaces; the result is
element-local and must be made continuous with
[`Spaces.weighted_dss!`](@ref) before it is differentiated again. Not yet
implemented for discontinuous (DG) spaces, which need grad-div/curl-curl
face lifting.
"""
vector_laplacian(u; divergence_factor = 1) =
    vector_laplacian(Spaces.discretization(axes(u)), u, divergence_factor)

vector_laplacian(::Grids.DG, u, divergence_factor) = error(
    "vector_laplacian is not implemented for DG spaces, which need \
     grad-div/curl-curl face lifting",
)

function vector_laplacian(::Grids.CG, u, divergence_factor)
    space = axes(u)
    div = Divergence()
    wgrad = Gradient{WeakForm}()
    graddiv = lazy.(wgrad.(div.(u)))
    isone(divergence_factor) ||
        (graddiv = lazy.(divergence_factor .* graddiv))
    # The curl-curl part exists whenever the horizontal manifold has two
    # dimensions (a plane or the sphere, extruded or not); with one horizontal
    # dimension it vanishes identically.
    if Spaces.horizontal_space(space) isa Spaces.SpectralElementSpace1D
        return lazy.(Geometry.Covariant12Vector.(graddiv))
    end
    curl = Curl()
    wcurl = Curl{WeakForm}()
    curlcurl = lazy.(
        Geometry.Covariant12Vector.(
            wcurl.(Geometry.Covariant3Vector.(curl.(u))),
        ),
    )
    return lazy.(graddiv .- curlcurl)
end
