# Model-level CG↔DG switch: one tendency-assembly call site that completes an
# element-local weak-form tendency across element interfaces — by DSS on
# continuous spaces, by DG interface fluxes on discontinuous ones. The
# completion object is built once, from the space, so the discretization is a
# model configuration rather than a structural difference in the tendency
# code (issue #2605).

"""
    AbstractTendencyCompletion

Supertype of [`DSSCompletion`](@ref) and [`NumericalFluxCompletion`](@ref),
the two ways [`complete_tendency!`](@ref) couples element-local weak-form
tendencies across element interfaces. Construct with
[`tendency_completion`](@ref), which selects the subtype from the
discretization of the space (dispatch on [`Grids.discretization`](@ref)).
"""
abstract type AbstractTendencyCompletion end

"""
    DSSCompletion(buffer)

[`AbstractTendencyCompletion`](@ref) for continuous (CG) spaces:
[`complete_tendency!`](@ref) applies [`Spaces.weighted_dss!`](@ref) with the
stored `buffer`, projecting the element-local tendency onto the continuous
space. Constructed by [`tendency_completion`](@ref).
"""
struct DSSCompletion{B} <: AbstractTendencyCompletion
    buffer::B
end

"""
    NumericalFluxCompletion(numflux, boundary_numflux)

[`AbstractTendencyCompletion`](@ref) for discontinuous (DG) spaces:
[`complete_tendency!`](@ref) weights the tendency by `WJ`, accumulates the
interface flux `numflux` with [`add_numerical_flux_interior!`](@ref) (and,
when `boundary_numflux` is not `nothing`, the one-sided boundary flux with
[`add_numerical_flux_boundary!`](@ref)), and unweights. Constructed by
[`tendency_completion`](@ref).
"""
struct NumericalFluxCompletion{F, BF} <: AbstractTendencyCompletion
    numflux::F
    boundary_numflux::BF
end

"""
    tendency_completion(dydt; numflux, boundary_numflux = nothing)

Interface-coupling configuration for the tendency field `dydt` (or any field
with the tendency's space and value type), selected by dispatch on
[`Grids.discretization`](@ref): a [`DSSCompletion`](@ref) on `Grids.CG()`
spaces, a [`NumericalFluxCompletion`](@ref) on `Grids.DG()` ones. Built once
at model setup; the discretization choice then lives in the space, and the
tendency code is shared:

    completion = Operators.tendency_completion(dydt; numflux)
    # ... each RHS evaluation:
    @. dydt = -wdiv(physical_flux(y, params))       # element-local weak form
    Operators.complete_tendency!(completion, dydt, y, params)

`numflux(normal, argvals⁻, argvals⁺)` is the DG interface flux (see
[`add_numerical_flux_interior!`](@ref) for the contract); it must be built
from the same physical flux as the weak volume term. It is required on DG
spaces and unused on CG spaces, so models can pass it unconditionally.
`boundary_numflux(normal, argvals⁻)` is the one-sided flux at domain-boundary
faces (see [`add_numerical_flux_boundary!`](@ref)); without it, boundary
faces contribute nothing to the DG tendency (a zero-flux closure). CG
boundary conditions are imposed by the operators themselves, not here.

`dydt` may be a `Field` on either discretization, or a `FieldVector` on a
continuous one, where the completion is a single batched
[`Spaces.weighted_dss!`](@ref) over all components. A `FieldVector` is not
supported on a discontinuous space: the interface flux is evaluated on the
whole state at a face node, which requires the state to be one `Field` with a
composite (e.g. `NamedTuple`) eltype rather than a collection of `Field`s.
"""
tendency_completion(
    dydt::Union{Fields.Field, Fields.FieldVector};
    kwargs...,
) = tendency_completion(completion_discretization(dydt), dydt; kwargs...)

tendency_completion(
    ::Grids.CG,
    dydt;
    numflux = nothing,
    boundary_numflux = nothing,
) = DSSCompletion(Spaces.create_dss_buffer(dydt))

function tendency_completion(
    ::Grids.DG,
    dydt;
    numflux = nothing,
    boundary_numflux = nothing,
)
    isnothing(numflux) && error(
        "tendency_completion on a discontinuous (DG) space requires a \
         numflux keyword argument (the interface numerical flux)",
    )
    return NumericalFluxCompletion(numflux, boundary_numflux)
end

tendency_completion(
    ::Grids.DG,
    dydt::Fields.FieldVector;
    numflux = nothing,
    boundary_numflux = nothing,
) = error(
    "tendency_completion does not support a FieldVector tendency on a \
     discontinuous (DG) space: the interface numerical flux is evaluated on \
     the whole state at a face node, so the tendency must be a single Field \
     with a composite (e.g. NamedTuple) eltype. FieldVector tendencies are \
     supported on continuous (CG) spaces, where the completion is a DSS.",
)

"""
    completion_discretization(dydt)

The discretization that [`tendency_completion`](@ref) dispatches on. One
completion applies a single interface treatment, so every component of a
`FieldVector` must agree, even though components may live on different spaces
(centers and faces).
"""
completion_discretization(dydt::Fields.Field) =
    Spaces.discretization(axes(dydt))

function completion_discretization(dydt::Fields.FieldVector)
    names = propertynames(dydt)
    isempty(names) && error(
        "tendency_completion needs at least one component to determine the \
         discretization of a FieldVector tendency",
    )
    discretization = completion_discretization(getproperty(dydt, first(names)))
    for name in names
        completion_discretization(getproperty(dydt, name)) === discretization ||
            error(
                "tendency_completion requires every component of a FieldVector \
                 tendency to share one discretization; :$name differs from \
                 :$(first(names))",
            )
    end
    return discretization
end

"""
    complete_tendency!(completion, dydt, args...)

Complete the element-local weak-form tendency `dydt` across element
interfaces, using the [`tendency_completion`](@ref) built for its space. The
contract on `dydt` is the CG weak-form convention (e.g. `-wdivₕ(F)` for a
flux-form equation, with no `WJ` weighting); `args` are the arguments of the
completion's flux functions, the model state first.

On a [`DSSCompletion`](@ref) this is `Spaces.weighted_dss!(dydt, buffer)`
(`args` unused). On a [`NumericalFluxCompletion`](@ref) it is the
mass-weighted DG surface term:

    dydt *= WJ
    add_numerical_flux_interior!(numflux, dydt, args...)
    add_numerical_flux_boundary!(boundary_numflux, dydt, args...)  # if given
    dydt /= WJ

Returns `dydt`.
"""
function complete_tendency!(completion::DSSCompletion, dydt, args...)
    Spaces.weighted_dss!(dydt, completion.buffer)
    return dydt
end

# `args::Vararg{Any, N}` for the same reason as in the face-operator chain it
# calls into (see add_numerical_flux_interior!): pass-through varargs would
# otherwise compile unspecialized.
function complete_tendency!(
    completion::NumericalFluxCompletion,
    dydt,
    args::Vararg{Any, N},
) where {N}
    lgeom = Fields.local_geometry_field(axes(dydt))
    @. dydt = dydt * lgeom.WJ
    add_numerical_flux_interior!(completion.numflux, dydt, args...)
    isnothing(completion.boundary_numflux) || add_numerical_flux_boundary!(
        completion.boundary_numflux,
        dydt,
        args...,
    )
    @. dydt = dydt / lgeom.WJ
    return dydt
end
