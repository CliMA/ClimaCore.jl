import .DataLayouts: slab_index

"""
    AbstractNumericalFlux

Abstract supertype for all numerical flux functors used by
[`add_numerical_flux_internal!`](@ref) and
[`add_numerical_flux_boundary!`](@ref).

Concrete subtypes must be callable with

    (normal, argvals⁻, argvals⁺)

and return the net flux from the \"minus\" side to the \"plus\" side.
"""
abstract type AbstractNumericalFlux end

"""
    add_numerical_flux_internal!(fn, dydt, args...)

Add the numerical flux at the internal faces of the spectral space mesh.

The numerical flux is determined by evaluating

    fn(normal, argvals⁻, argvals⁺)

where:
 - `normal` is the unit normal vector, pointing from the "minus" side to the "plus" side
 - `argvals⁻` is the tuple of values of `args` on the "minus" side of the face
 - `argvals⁺` is the tuple of values of `args` on the "plus" side of the face
and should return the net flux from the "minus" side to the "plus" side.

For consistency, it should satisfy the property that

    fn(normal, argvals⁻, argvals⁺) == -fn(-normal, argvals⁺, argvals⁻)


See also:
- [`CentralNumericalFlux`](@ref)
- [`RusanovNumericalFlux`](@ref)
"""
function add_numerical_flux_internal!(fn, dydt, args...)
    space = axes(dydt)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    topology = Spaces.topology(space)
    internal_surface_geometry = Spaces.grid(space).internal_surface_geometry

    for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
        enumerate(Topologies.interior_faces(topology))

        internal_surface_geometry_slab = slab(internal_surface_geometry, iface)

        arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), elem⁻), args)
        arg_slabs⁺ = map(arg -> slab(Fields.todata(arg), elem⁺), args)

        dydt_slab⁻ = slab(Fields.field_values(dydt), elem⁻)
        dydt_slab⁺ = slab(Fields.field_values(dydt), elem⁺)

        for q in 1:Nq
            sgeom⁻ = internal_surface_geometry_slab[slab_index(q)]

            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

            numflux⁻ = fn(
                sgeom⁻.normal,
                map(
                    slab ->
                        slab isa DataSlab2D ? slab[slab_index(i⁻, j⁻)] : slab,
                    arg_slabs⁻,
                ),
                map(
                    slab ->
                        slab isa DataSlab2D ? slab[slab_index(i⁺, j⁺)] : slab,
                    arg_slabs⁺,
                ),
            )

            dydt_slab⁻[slab_index(i⁻, j⁻)] =
                dydt_slab⁻[slab_index(i⁻, j⁻)] ⊟ (sgeom⁻.sWJ ⊠ numflux⁻)
            dydt_slab⁺[slab_index(i⁺, j⁺)] =
                dydt_slab⁺[slab_index(i⁺, j⁺)] ⊞ (sgeom⁻.sWJ ⊠ numflux⁻)
        end
    end
end

"""
    CentralNumericalFlux(fluxfn)

Evaluates the central numerical flux using `fluxfn`.
"""
struct CentralNumericalFlux{F} <: AbstractNumericalFlux
    fluxfn::F
end

function (fn::CentralNumericalFlux)(normal, argvals⁻, argvals⁺)
    Favg =
        RecursiveApply.rdiv(fn.fluxfn(argvals⁻...) ⊞ fn.fluxfn(argvals⁺...), 2)
    return RecursiveApply.rmap(f -> f' * normal, Favg)
end

"""
    RusanovNumericalFlux(fluxfn, wavespeedfn)

Evaluates the Rusanov numerical flux using `fluxfn` with wavespeed `wavespeedfn`
"""
struct RusanovNumericalFlux{F, W} <: AbstractNumericalFlux
    fluxfn::F
    wavespeedfn::W
end

function (fn::RusanovNumericalFlux)(normal, argvals⁻, argvals⁺)
    y⁻ = argvals⁻[1]
    y⁺ = argvals⁺[1]
    Favg =
        RecursiveApply.rdiv(fn.fluxfn(argvals⁻...) ⊞ fn.fluxfn(argvals⁺...), 2)
    λ = max(fn.wavespeedfn(argvals⁻...), fn.wavespeedfn(argvals⁺...))
    return RecursiveApply.rmap(f -> f' * normal, Favg) ⊞ (λ / 2) ⊠ (y⁻ ⊟ y⁺)
end

"""
    KineticEnergyPreservingNumericalFlux()

Kinetic-energy-preserving numerical flux for the Bickley jet system.

This flux is based on a symmetric two-point form:
- mass flux is an average normal mass flux,
- momentum flux uses the averaged velocity dotted with the averaged mass flux
  plus an averaged pressure contribution,
- tracer flux uses the averaged specific tracer.

It is designed so that, when combined with a suitable split / SBP volume
discretization, the discrete kinetic energy is preserved in the inviscid,
periodic case (up to machine precision), following the kinetic-energy-preserving
fluxes discussed in the entropy-stable DG literature (e.g. Souza et al., 2022).
"""
struct KineticEnergyPreservingNumericalFlux <: AbstractNumericalFlux end

"""
    pressure_from_state(state, parameters)

Default equation of state used by kinetic-energy-preserving fluxes.

Users may extend this method for their own state/parameter types to supply
an appropriate pressure law.
"""
pressure_from_state(state, parameters) = parameters.g * state.ρ^2 / 2

"""
    sound_speed_from_state(state, parameters)

Default approximate sound speed used by entropy-stable fluxes.

By default this assumes an effective relation c² ≈ 2p/ρ, which is exact for
the shallow-water-like law p = g ρ² / 2 and a reasonable proxy otherwise.
Users may overload this for more accurate thermodynamics.
"""
function sound_speed_from_state(state, parameters)
    p = pressure_from_state(state, parameters)
    ρ = state.ρ
    T = real(eltype(ρ))
    return sqrt(max(eps(T), (2 * p) / ρ))
end

function (::KineticEnergyPreservingNumericalFlux)(
    normal,
    (y⁻, p⁻),
    (y⁺, p⁺),
)
    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    u⁺ = ρu⁺ / ρ⁺

    θ⁻ = ρθ⁻ / ρ⁻
    θ⁺ = ρθ⁺ / ρ⁺

    uₙ⁻ = u⁻' * normal
    uₙ⁺ = u⁺' * normal

    # normal mass flux (symmetric average)
    mₙ⁻ = ρ⁻ * uₙ⁻
    mₙ⁺ = ρ⁺ * uₙ⁺
    m̂ₙ = (mₙ⁻ + mₙ⁺) / 2

    # averaged velocity and pressure
    û = (u⁻ + u⁺) / 2

    # pressure from equation of state (can be overloaded by users)
    pL = pressure_from_state(y⁻, p⁻)
    pR = pressure_from_state(y⁺, p⁺)
    p̄ = (pL + pR) / 2

    # averaged tracer
    θ̂ = (θ⁻ + θ⁺) / 2

    # fluxes already dotted with the normal (entropy-conservative core)
    flux_ρ  = m̂ₙ
    flux_ρu = m̂ₙ * û + p̄ * normal
    flux_ρθ = m̂ₙ * θ̂

    F_core = (ρ = flux_ρ, ρu = flux_ρu, ρθ = flux_ρθ)

    # entropy-stabilizing dissipation term (Rusanov-type, added to KE-preserving core)
    cL = sound_speed_from_state(y⁻, p⁻)
    cR = sound_speed_from_state(y⁺, p⁺)
    λL = abs(uₙ⁻) + cL
    λR = abs(uₙ⁺) + cR
    λ  = max(λL, λR)

    diss = (λ / 2) ⊠ (y⁻ ⊟ y⁺)

    return F_core ⊞ diss
end


function add_numerical_flux_boundary!(fn, dydt, args...)
    space = axes(dydt)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    topology = Spaces.topology(space)
    boundary_surface_geometries = Spaces.grid(space).boundary_surface_geometries

    for (iboundary, boundarytag) in
        enumerate(Topologies.boundary_tags(topology))
        for (iface, (elem⁻, face⁻)) in
            enumerate(Topologies.boundary_faces(topology, boundarytag))
            boundary_surface_geometry_slab =
                surface_geometry_slab =
                    slab(boundary_surface_geometries[iboundary], iface)

            arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), elem⁻), args)
            dydt_slab⁻ = slab(Fields.field_values(dydt), elem⁻)
            for q in 1:Nq
                sgeom⁻ = boundary_surface_geometry_slab[slab_index(q)]
                i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
                numflux⁻ = fn(
                    sgeom⁻.normal,
                    map(
                        slab ->
                            slab isa DataSlab2D ? slab[slab_index(i⁻, j⁻)] :
                            slab,
                        arg_slabs⁻,
                    ),
                )
                dydt_slab⁻[slab_index(i⁻, j⁻)] =
                    dydt_slab⁻[slab_index(i⁻, j⁻)] ⊟ (sgeom⁻.sWJ ⊠ numflux⁻)
            end
        end
    end
    return dydt
end
