# Horizontal Laplacian atoms shared by CG and DG discretizations. These are
# the building blocks of ∇⁴ hyperdiffusion: two Laplacian passes with a
# `Spaces.weighted_dss!` of the intermediate fields between them. Owning the
# atoms here lets the continuity mechanics differ by discretization — DSS of
# the intermediate on CG, interior-penalty face fluxes inside each pass on DG —
# while callers use one calling sequence for both.

"""
    scalar_laplacian(χ; weight = nothing)

Weak horizontal Laplacian of a scalar, `wdivₕ(gradₕ(χ))`, or
`wdivₕ(weight * gradₕ(χ))` when a `weight` field (e.g. a density) is given.

On continuous (CG) spaces this returns a lazy operator expression, so it fuses
into consuming broadcasts; the result is element-local, and materialized
intermediates must be made continuous with [`Spaces.weighted_dss!`](@ref)
before they are differentiated again (batch several intermediates into one
call to share the ghost exchange). On discontinuous (DG) spaces this returns a
materialized `Field` that already includes the interior-penalty face
corrections (see [`ldg_laplacian_tendency`](@ref)), and `weighted_dss!` is a
no-op — so the same prep → `weighted_dss!` → apply sequence is correct for
both discretizations.
"""
function scalar_laplacian(χ; weight = nothing)
    space = axes(χ)
    if Spaces.is_continuous(space)
        wdiv = WeakDivergence()
        grad = Gradient()
        G = Base.Broadcast.broadcasted(grad, χ)
        isnothing(weight) || (G = Base.Broadcast.broadcasted(*, weight, G))
        return Base.Broadcast.broadcasted(wdiv, G)
    else
        q = Base.Broadcast.materialize(χ)
        κ = one(Spaces.undertype(space))
        return ldg_laplacian_tendency(
            q,
            weight,
            κ,
            ldg_penalty_parameter(κ, space),
        )
    end
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
function vector_laplacian(u; divergence_factor = 1)
    space = axes(u)
    Spaces.is_continuous(space) || error(
        "vector_laplacian is not implemented for DG spaces, which need \
         grad-div/curl-curl face lifting",
    )
    div = Divergence()
    wgrad = WeakGradient()
    graddiv =
        Base.Broadcast.broadcasted(wgrad, Base.Broadcast.broadcasted(div, u))
    # The curl-curl part exists whenever the horizontal manifold has two
    # dimensions (a plane or the sphere, extruded or not); with one horizontal
    # dimension it vanishes identically.
    hspace =
        space isa Spaces.ExtrudedFiniteDifferenceSpace ?
        Spaces.horizontal_space(space) : space
    if hspace isa Spaces.SpectralElementSpace1D
        graddiv =
            Base.Broadcast.broadcasted(Geometry.Covariant12Vector, graddiv)
        divergence_factor === 1 || (
            graddiv =
                Base.Broadcast.broadcasted(*, divergence_factor, graddiv)
        )
        return graddiv
    else
        divergence_factor === 1 || (
            graddiv =
                Base.Broadcast.broadcasted(*, divergence_factor, graddiv)
        )
        curl = Curl()
        wcurl = WeakCurl()
        curlcurl = Base.Broadcast.broadcasted(
            Geometry.Covariant12Vector,
            Base.Broadcast.broadcasted(
                wcurl,
                Base.Broadcast.broadcasted(
                    Geometry.Covariant3Vector,
                    Base.Broadcast.broadcasted(curl, u),
                ),
            ),
        )
        return Base.Broadcast.broadcasted(-, graddiv, curlcurl)
    end
end
