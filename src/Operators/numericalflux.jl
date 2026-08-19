"""
    AbstractNumericalFlux

Abstract type for numerical flux functions used in DG methods.
"""
abstract type AbstractNumericalFlux end

"""
    AbstractBoundaryCondition

Abstract type for boundary conditions in DG methods.
"""
abstract type AbstractBoundaryCondition end

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
- [`RoeNumericalFlux`](@ref)
"""
@inline face_node_index_1d(face, Nq) = face == 1 ? 1 : Nq

# Surface Jacobian weight and outward unit normal for a 1D spectral-element
# endpoint. For extruded spaces, `local_geometry` should be the product
# geometry at that horizontal node and vertical level.
function compute_surface_geometry_1d(local_geometry, face)
    (; J, ∂ξ∂x) = local_geometry
    nvec = face == 1 ? (-J * ∂ξ∂x[1, :]) : (J * ∂ξ∂x[1, :])
    sWJ = LinearAlgebra.norm(nvec)
    n = nvec / sWJ
    # Project onto the horizontal orthonormal axis used by plane (x–z) states.
    n = Geometry.project(Geometry.UWAxis(), n)
    return Geometry.SurfaceGeometry(sWJ, Geometry.UVector(n.u))
end

# Surface Jacobian weight and outward unit horizontal normal for a face node
# (i, j) of a 2D spectral element within an extruded space. `local_geometry`
# is the product geometry at that horizontal node and vertical level, so `J`
# carries the vertical measure and `sWJ` is consistent with the 3D `WJ` of the
# mass-weighted volume residual. The normal is returned in the local
# orthonormal horizontal frame (`UVVector`): at a shared face node this frame
# is identical from both sides — including across cubed-sphere panel
# boundaries, where covariant components are discontinuous.
function compute_surface_geometry_extruded_2d(
    local_geometry,
    quad_weights,
    face,
    i,
    j,
    mstyle = Val{:horizontal}(),
)
    (; J, ∂ξ∂x) = local_geometry
    nvec = if face == 4
        -J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 2
        J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 1
        -J * ∂ξ∂x[2, :] * quad_weights[i]
    elseif face == 3
        J * ∂ξ∂x[2, :] * quad_weights[i]
    else
        error("invalid face index $face")
    end
    sWJ = LinearAlgebra.norm(nvec)
    n = Geometry.project(_metric_axis(mstyle), nvec / sWJ)
    return Geometry.SurfaceGeometry(sWJ, n)
end

# Device-dispatch seam (DSS-style): CPU methods live here; the
# `ClimaComms.CUDADevice` methods are provided by the ClimaCoreCUDAExt
# extension (ext/cuda/operators_dg.jl).
add_numerical_flux_internal!(fn, dydt, args...) = _add_numerical_flux_internal!(
    ClimaComms.device(axes(dydt)),
    fn,
    dydt,
    args...,
)

_add_numerical_flux_internal!(device, fn, dydt, args...) = error(
    "add_numerical_flux_internal! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

function _add_numerical_flux_internal!(
    ::ClimaComms.AbstractCPUDevice,
    fn,
    dydt,
    args...,
)
    space = axes(dydt)
    grid = Spaces.grid(space)
    if grid isa Grids.ExtrudedFiniteDifferenceGrid
        if grid.horizontal_grid isa Grids.SpectralElementGrid1D
            return add_numerical_flux_internal_extruded_1d!(fn, dydt, args...)
        elseif grid.horizontal_grid isa Grids.SpectralElementGrid2D
            return add_numerical_flux_internal_extruded_2d!(fn, dydt, args...)
        end
    end

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    topology = Spaces.topology(space)
    internal_surface_geometry = grid.internal_surface_geometry
    dydt_bc = Base.broadcastable(dydt)
    args_bc =
        map(arg -> arg isa Fields.Field ? Base.broadcastable(arg) : arg, args)

    for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
        enumerate(Topologies.interior_faces(topology))

        internal_surface_geometry_slab = slab(internal_surface_geometry, 1, iface)

        arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), 1, elem⁻), args_bc)
        arg_slabs⁺ = map(arg -> slab(Fields.todata(arg), 1, elem⁺), args_bc)

        dydt_slab⁻ = slab(Fields.field_values(dydt_bc), 1, elem⁻)
        dydt_slab⁺ = slab(Fields.field_values(dydt_bc), 1, elem⁺)

        for q in 1:Nq
            sgeom⁻ = internal_surface_geometry_slab[q]

            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

            argvals⁻ = map(
                slab -> slab isa DataLayouts.DataLayout ? slab[1, i⁻, j⁻, 1] : slab,
                arg_slabs⁻,
            )
            argvals⁺ = map(
                slab -> slab isa DataLayouts.DataLayout ? slab[1, i⁺, j⁺, 1] : slab,
                arg_slabs⁺,
            )
            numflux⁻ =
                add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻, argvals⁺))

            dydt_slab⁻[1, i⁻, j⁻, 1] =
                dydt_slab⁻[1, i⁻, j⁻, 1] - (sgeom⁻.sWJ * numflux⁻)
            dydt_slab⁺[1, i⁺, j⁺, 1] =
                dydt_slab⁺[1, i⁺, j⁺, 1] + (sgeom⁻.sWJ * numflux⁻)
        end
    end
    return dydt
end

"""
    add_numerical_flux_internal_extruded_1d!(fn, dydt, args...)

Add horizontal numerical fluxes on an extruded plane space
(`SpectralElementSpace1D` × finite-difference vertical).

Loops over vertical levels and 1D interval interior faces. Surface geometry is
built from the product local geometry (so ``sWJ`` carries the vertical measure).
`dydt` must already be stored in mass-weighted residual form (`WJ * ∂Y/∂t`),
matching the flat-DG convention used with [`WeakDivergence`](@ref).
"""
function add_numerical_flux_internal_extruded_1d!(fn, dydt, args...)
    space = axes(dydt)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nv = Spaces.nlevels(space)
    topology = Spaces.topology(space)
    local_geometry = Spaces.local_geometry_data(space)

    dydt_data = Fields.field_values(dydt)
    args_data = map(
        arg -> arg isa Fields.Field ? Fields.field_values(arg) : arg,
        args,
    )

    for v in 1:Nv
        for (elem⁻, face⁻, elem⁺, face⁺, _reversed) in
            Topologies.interior_faces(topology)

            i⁻ = face_node_index_1d(face⁻, Nq)
            i⁺ = face_node_index_1d(face⁺, Nq)

            lg⁻ = slab(local_geometry, v, elem⁻)[i⁻]
            sgeom⁻ = compute_surface_geometry_1d(lg⁻, face⁻)

            argvals⁻ = map(args_data) do arg
                val =
                    arg isa DataLayouts.AbstractData ?
                    slab(arg, v, elem⁻)[i⁻] : arg
                add_auto_broadcasters(val)
            end
            argvals⁺ = map(args_data) do arg
                val =
                    arg isa DataLayouts.AbstractData ?
                    slab(arg, v, elem⁺)[i⁺] : arg
                add_auto_broadcasters(val)
            end

            numflux⁻ =
                add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻, argvals⁺))

            dydt_slab⁻ = slab(dydt_data, v, elem⁻)
            dydt_slab⁺ = slab(dydt_data, v, elem⁺)
            dydt_slab⁻[i⁻] =
                dydt_slab⁻[i⁻] - (sgeom⁻.sWJ * numflux⁻)
            dydt_slab⁺[i⁺] =
                dydt_slab⁺[i⁺] + (sgeom⁻.sWJ * numflux⁻)
        end
    end
    return dydt
end

"""
    add_numerical_flux_internal_extruded_2d!(fn, dydt, args...)

Add horizontal numerical fluxes on an extruded 3D space
(`SpectralElementSpace2D` horizontal × finite-difference vertical), e.g. a
cubed-sphere shell. Loops over vertical levels, 2D interior faces, and face
nodes. Surface geometry is built from the product local geometry (so ``sWJ``
carries the vertical measure) and normals are in the local orthonormal
horizontal frame (`UVVector`). `dydt` must be stored in mass-weighted residual
form (`WJ * ∂Y/∂t`).
"""
function add_numerical_flux_internal_extruded_2d!(fn, dydt, args...)
    space = axes(dydt)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    Nv = Spaces.nlevels(space)
    topology = Spaces.topology(space)
    local_geometry = Spaces.local_geometry_data(space)
    FT = Spaces.undertype(space)
    (_, quad_weights) = Quadratures.quadrature_points(FT, quadrature_style)

    mstyle = _fd_metric_style(fn)
    dydt_data = Fields.field_values(dydt)
    args_data = map(
        arg -> arg isa Fields.Field ? Fields.field_values(arg) : arg,
        args,
    )

    for v in 1:Nv
        for (elem⁻, face⁻, elem⁺, face⁺, reversed) in
            Topologies.interior_faces(topology)

            dydt_slab⁻ = slab(dydt_data, v, elem⁻)
            dydt_slab⁺ = slab(dydt_data, v, elem⁺)

            for q in 1:Nq
                i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
                i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

                lg⁻ = slab(local_geometry, v, elem⁻)[1, i⁻, j⁻, 1]
                sgeom⁻ = compute_surface_geometry_extruded_2d(
                    lg⁻,
                    quad_weights,
                    face⁻,
                    i⁻,
                    j⁻,
                    mstyle,
                )

                argvals⁻ = map(args_data) do arg
                    val =
                        arg isa DataLayouts.AbstractData ?
                        slab(arg, v, elem⁻)[1, i⁻, j⁻, 1] : arg
                    add_auto_broadcasters(val)
                end
                argvals⁺ = map(args_data) do arg
                    val =
                        arg isa DataLayouts.AbstractData ?
                        slab(arg, v, elem⁺)[1, i⁺, j⁺, 1] : arg
                    add_auto_broadcasters(val)
                end

                numflux⁻ =
                    add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻, argvals⁺))

                dydt_slab⁻[1, i⁻, j⁻, 1] =
                    dydt_slab⁻[1, i⁻, j⁻, 1] - (sgeom⁻.sWJ * numflux⁻)
                dydt_slab⁺[1, i⁺, j⁺, 1] =
                    dydt_slab⁺[1, i⁺, j⁺, 1] + (sgeom⁻.sWJ * numflux⁻)
            end
        end
    end
    return dydt
end

"""
    CentralNumericalFlux(fluxfn)

Evaluates the central numerical flux using `fluxfn`.
"""
struct CentralNumericalFlux{F} <: AbstractNumericalFlux
    fluxfn::F
end

function (fn::CentralNumericalFlux)(normal, argvals⁻, argvals⁺)
    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    return ((F⁻ + F⁺) / 2)' * normal
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
    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    λ = max(fn.wavespeedfn(argvals⁻...), fn.wavespeedfn(argvals⁺...))
    Favg = ((F⁻ + F⁺) / 2)' * normal
    return Favg + (λ / 2) * (y⁻ - y⁺)
end

"""
    RoeNumericalFlux(fluxfn, roe_avg_fn)

Evaluates the Roe numerical flux using `fluxfn` and Roe-averaging function `roe_avg_fn`.

The Roe flux computes a central flux plus an entropy-stable dissipation term based on
the characteristic decomposition of the jump in conserved variables.
"""
struct RoeNumericalFlux{F, A} <: AbstractNumericalFlux
    fluxfn::F
    roe_avg_fn::A
end

function (fn::RoeNumericalFlux)(normal, argvals⁻, argvals⁺)
    y⁻ = argvals⁻[1]
    y⁺ = argvals⁺[1]
    params⁻ = argvals⁻[2]
    params⁺ = argvals⁺[2]

    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    Favg = (F⁻ + F⁺) / 2

    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * normal

    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * normal

    λ = sqrt(params⁻.g)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)

    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    ρ̄ = sqrt(ρ⁻ * ρ⁺)
    ū = fn.roe_avg_fn(ρ⁻, ρ⁺, u⁻, u⁺)
    θ̄ = fn.roe_avg_fn(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c̄ = fn.roe_avg_fn(ρ⁻, ρ⁺, c⁻, c⁺)

    ūₙ = ū' * normal

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * normal

    c̄⁻² = 1 / c̄^2
    w1 = abs(ūₙ - c̄) * (Δp - ρ̄ * c̄ * Δuₙ) * 0.5 * c̄⁻²
    w2 = abs(ūₙ + c̄) * (Δp + ρ̄ * c̄ * Δuₙ) * 0.5 * c̄⁻²
    w3 = abs(ūₙ) * (Δρ - Δp * c̄⁻²)
    w4 = abs(ūₙ) * ρ̄
    w5 = abs(ūₙ) * (Δρθ - θ̄ * Δp * c̄⁻²)

    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρu =
        (
            w1 * (ū - c̄ * normal) + w2 * (ū + c̄ * normal) + w3 * ū +
            w4 * (Δu - Δuₙ * normal)
        ) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ̄ + w5) * 0.5

    return (
        ρ = ((F⁻.ρ + F⁺.ρ) / 2)' * normal - fluxᵀn_ρ,
        ρu = ((F⁻.ρu + F⁺.ρu) / 2)' * normal - fluxᵀn_ρu,
        ρθ = ((F⁻.ρθ + F⁺.ρθ) / 2)' * normal - fluxᵀn_ρθ,
    )
end


"""
    PeriodicBC <: AbstractBoundaryCondition

Periodic boundary condition (handled by topology, no ghost state needed).
"""
struct PeriodicBC <: AbstractBoundaryCondition end

"""
    ReflectingWallBC <: AbstractBoundaryCondition

Reflecting wall boundary condition (no-normal-flow).
Reflects normal momentum component; preserves density and potential temperature.
"""
struct ReflectingWallBC <: AbstractBoundaryCondition end

"""
    ghost_state(bc::AbstractBoundaryCondition, normal, argvals⁻)

Construct the exterior-side argument tuple for the given BC.

Returns a tuple with the same length as `argvals⁻`, replacing only the
prognostic state `argvals⁻[1]` with the ghost state; remaining arguments
(e.g. equation parameters, coordinates) are forwarded unchanged.
"""
function ghost_state(::AbstractBoundaryCondition, normal, argvals⁻)
    error("ghost_state not implemented for this boundary condition")
end

function ghost_state(::ReflectingWallBC, normal, argvals⁻)
    y⁻ = argvals⁻[1]
    ρu⁺ = y⁻.ρu - 2 * LinearAlgebra.dot(y⁻.ρu, normal) * normal
    # y⁻ may arrive wrapped in an AutoBroadcaster at element boundaries;
    # unwrap before merge so we always work with a plain NamedTuple.
    y⁺ = merge(unwrap(y⁻), (ρu = ρu⁺,))
    return (y⁺, argvals⁻[2:end]...)
end

function add_numerical_flux_boundary!(fn, dydt, args...)
    space = axes(dydt)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    topology = Spaces.topology(space)
    boundary_surface_geometries = Spaces.grid(space).boundary_surface_geometries
    dydt_bc = Base.broadcastable(dydt)
    args_bc =
        map(arg -> arg isa Fields.Field ? Base.broadcastable(arg) : arg, args)

    for (iboundary, boundarytag) in
        enumerate(Topologies.boundary_tags(topology))
        for (iface, (elem⁻, face⁻)) in
            enumerate(Topologies.boundary_faces(topology, boundarytag))
            boundary_surface_geometry_slab =
                surface_geometry_slab =
                    slab(boundary_surface_geometries[iboundary], 1, iface)

            arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), 1, elem⁻), args_bc)
            dydt_slab⁻ = slab(Fields.field_values(dydt_bc), 1, elem⁻)
            for q in 1:Nq
                sgeom⁻ = boundary_surface_geometry_slab[q]
                i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
                argvals⁻ = map(
                    slab ->
                        slab isa DataLayouts.DataLayout ? slab[1, i⁻, j⁻, 1] : slab,
                    arg_slabs⁻,
                )
                numflux⁻ = add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻))
                dydt_slab⁻[1, i⁻, j⁻, 1] =
                    dydt_slab⁻[1, i⁻, j⁻, 1] - (sgeom⁻.sWJ * numflux⁻)
            end
        end
    end
    return dydt
end

"""
    add_numerical_flux_boundary!(numflux::AbstractNumericalFlux, bc::AbstractBoundaryCondition, dydt, args...)

Add numerical flux at boundaries using a typed boundary condition.
Constructs the ghost state via `ghost_state(bc, normal, argvals⁻)` and applies the numerical flux.
"""
function add_numerical_flux_boundary!(
    numflux::AbstractNumericalFlux,
    bc::AbstractBoundaryCondition,
    dydt,
    args...,
)
    add_numerical_flux_boundary!(dydt, args...) do normal, argvals⁻
        argvals⁺ = ghost_state(bc, normal, argvals⁻)
        numflux(normal, argvals⁻, argvals⁺)
    end
end

# Kinetic energy of a velocity AxisVector (2D UV / UW plane or 1D U).
@inline _specific_ke(u::Geometry.UVVector) = (u.u^2 + u.v^2) / 2
@inline _specific_ke(u::Geometry.UWVector) = (u.u^2 + u.w^2) / 2
@inline _specific_ke(u::Geometry.UVector) = (u.u^2) / 2

# Tangential unit for Roe shear wave: rotate n̂ 90° CCW in the horizontal
# plane. 1D `UVector` normals (extruded x–z hybrid) have no in-plane shear.
@inline _roe_tangent(n::Geometry.UVVector) = typeof(n)(-n.v, n.u)
@inline _roe_tangent(::Geometry.UVector) = nothing
@inline _roe_tangent(::Geometry.UWVector) = nothing

"""
    ideal_gas_pressure(state, params)

Default pressure for `EntropyConservingFlux`: `p = (γ-1)(ρe - ρKE)`.
"""
function ideal_gas_pressure(state, params)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    return (params.γ - 1) * (ρe - ρ * _specific_ke(u))
end

"""
    EntropyConservingFlux(fluxfn, entropy_var_fn, roe_avg_fn[; pressure_fn, momentum_pressure_fn, roe_pressure_fn, sound_speed_fn])

Kennedy-Gruber kinetic energy preserving (KEP) interface flux with Roe entropy-stable
dissipation for compressible Euler equations, following Souza et al. (2023, JAMES, Eqs 40-42).

The central part uses arithmetic averages of primitive variables {ρ}, {u}, {p}, {e}, giving
the KEP property. The Roe dissipation uses a full characteristic decomposition of the Roe-averaged
Jacobian (4 waves in 2D: two acoustics, one entropy/contact, one shear).

- `fluxfn(state, params...)`: physical flux tensor F(U)
- `entropy_var_fn(state, params...)`: entropy variables v = ∂η/∂U (stored, not used in dissipation)
- `roe_avg_fn(ρ⁻, ρ⁺, var⁻, var⁺)`: Roe-averaging function, e.g. density-weighted average
- `pressure_fn(state, params...)`: thermodynamic pressure for enthalpy / energy (defaults to [`ideal_gas_pressure`](@ref))
- `momentum_pressure_fn(state, params...)`: pressure in the K-G momentum flux (defaults to `pressure_fn`)
- `roe_pressure_fn(state, params...)`: pressure in Roe wave amplitudes α₁, α₂, α₄ (defaults to `momentum_pressure_fn`, so stratified p′ formulations stay consistent with the volume flux)
- `sound_speed_fn(state, params...)`: optional Roe sound speed; if `nothing`, uses `√((γ-1)(H̃-KẼ))`
"""
struct EntropyConservingFlux{F, V, A, P, MP, RP, S} <: AbstractNumericalFlux
    fluxfn::F
    entropy_var_fn::V
    roe_avg_fn::A
    pressure_fn::P
    momentum_pressure_fn::MP
    roe_pressure_fn::RP
    sound_speed_fn::S

    function EntropyConservingFlux(
        fluxfn,
        entropy_var_fn,
        roe_avg_fn;
        pressure_fn = ideal_gas_pressure,
        momentum_pressure_fn = nothing,
        roe_pressure_fn = nothing,
        sound_speed_fn = nothing,
    )
        F, V, A, P = typeof.((fluxfn, entropy_var_fn, roe_avg_fn, pressure_fn))
        MP = momentum_pressure_fn === nothing ? pressure_fn : momentum_pressure_fn
        # Roe Δp must use the same pressure as the K-G / volume momentum flux (p′ for stratified).
        RP = roe_pressure_fn === nothing ? MP : roe_pressure_fn
        S = sound_speed_fn
        return new{F, V, A, P, typeof(MP), typeof(RP), typeof(S)}(
            fluxfn,
            entropy_var_fn,
            roe_avg_fn,
            pressure_fn,
            MP,
            RP,
            S,
        )
    end
end

# Positional `pressure_fn` for backward compatibility (e.g. Compressible Euler).
function EntropyConservingFlux(fluxfn, entropy_var_fn, roe_avg_fn, pressure_fn)
    return EntropyConservingFlux(
        fluxfn,
        entropy_var_fn,
        roe_avg_fn;
        pressure_fn,
    )
end

function (fn::EntropyConservingFlux)(normal, argvals⁻, argvals⁺)
    y⁻ = argvals⁻[1]
    y⁺ = argvals⁺[1]
    params = argvals⁻[2]

    ρ⁻, ρu⁻, ρe⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρe
    ρ⁺, ρu⁺, ρe⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρe

    u⁻ = ρu⁻ / ρ⁻
    u⁺ = ρu⁺ / ρ⁺
    γ = params.γ

    KE⁻ = _specific_ke(u⁻)
    KE⁺ = _specific_ke(u⁺)
    p⁻ = fn.pressure_fn(argvals⁻...)
    p⁺ = fn.pressure_fn(argvals⁺...)
    pm⁻ = fn.momentum_pressure_fn(argvals⁻...)
    pm⁺ = fn.momentum_pressure_fn(argvals⁺...)
    p_roe⁻ = fn.roe_pressure_fn(argvals⁻...)
    p_roe⁺ = fn.roe_pressure_fn(argvals⁺...)

    # Kennedy-Gruber KEP interface flux: arithmetic averages (Souza et al. 2023,
    # JAMES, Eqs 40–42 / App. A). Uses {ρ}{u}{p}{e} for the central flux.
    ρ̄ = (ρ⁻ + ρ⁺) / 2
    ū = (u⁻ + u⁺) / 2
    p̄ = (p⁻ + p⁺) / 2
    p̄m = (pm⁻ + pm⁺) / 2
    ē = (ρe⁻ / ρ⁻ + ρe⁺ / ρ⁺) / 2  # arithmetic mean of specific total energy

    Fc_ρ = (ρ̄ * ū)' * normal
    Fc_ρu = (ρ̄ * (ū ⊗ ū) + p̄m * I)' * normal
    Fc_ρe = (ū * (ρ̄ * ē + p̄))' * normal  # {u}({ρ}{e} + {p})

    # Roe-averaged state for compressible Euler (Roe 1981)
    # Guard non-positive densities at the face (does not floor prognostics).
    pos = ρ⁻ > 0 && ρ⁺ > 0
    ρ̃ = pos ? sqrt(ρ⁻ * ρ⁺) : abs(ρ̄)
    ũ = pos ? fn.roe_avg_fn(ρ⁻, ρ⁺, u⁻, u⁺) : ū
    H⁻ = (ρe⁻ + p⁻) / ρ⁻  # specific total enthalpy
    H⁺ = (ρe⁺ + p⁺) / ρ⁺
    H̃ = pos ? fn.roe_avg_fn(ρ⁻, ρ⁺, H⁻, H⁺) : (H⁻ + H⁺) / 2
    KE_tilde = _specific_ke(ũ)
    c̃ = if fn.sound_speed_fn === nothing
        # Fall back to thermodynamic Roe c only when H̃ > KẼ.
        ΔH = H̃ - KE_tilde
        ΔH > 0 ? sqrt((γ - 1) * ΔH) : FT_zero(ΔH)
    else
        c⁻ = fn.sound_speed_fn(argvals⁻...)
        c⁺ = fn.sound_speed_fn(argvals⁺...)
        pos ? fn.roe_avg_fn(ρ⁻, ρ⁺, c⁻, c⁺) : max(c⁻, c⁺)
    end

    # Normal (and tangential, in 2D) directions.
    # Extruded 1D faces use `UVector` normals → no shear wave (Souza 1D Euler).
    ũₙ = ũ' * normal
    tang = _roe_tangent(normal)

    Δρ = ρ⁺ - ρ⁻
    Δu = u⁺ - u⁻
    Δuₙ = Δu' * normal
    # p′ jump for Roe amplitudes when momentum_pressure_fn = p′ (stratified)
    Δp = p_roe⁺ - p_roe⁻

    c̃⁻² = 1 / c̃^2
    α₁ = (Δp - ρ̃ * c̃ * Δuₙ) * 0.5 * c̃⁻²   # left-running acoustic
    α₂ = Δρ - Δp * c̃⁻²                        # entropy / contact
    α₄ = (Δp + ρ̃ * c̃ * Δuₙ) * 0.5 * c̃⁻²   # right-running acoustic

    λ₁ = abs(ũₙ - c̃)
    λ₂ = abs(ũₙ)
    λ₄ = abs(ũₙ + c̃)

    diss_ρ = λ₁ * α₁ + λ₂ * α₂ + λ₄ * α₄
    diss_ρu =
        (λ₁ * α₁) * (ũ - c̃ * normal) +
        (λ₂ * α₂) * ũ +
        (λ₄ * α₄) * (ũ + c̃ * normal)
    diss_ρe =
        λ₁ * α₁ * (H̃ - c̃ * ũₙ) +
        λ₂ * α₂ * KE_tilde +
        λ₄ * α₄ * (H̃ + c̃ * ũₙ)

    if tang !== nothing
        Δuₜ = Δu' * tang
        ũₜ = ũ' * tang
        α₃ = ρ̃ * Δuₜ                               # shear / vorticity
        diss_ρu = diss_ρu + (λ₂ * α₃) * tang
        diss_ρe = diss_ρe + λ₂ * α₃ * ũₜ
    end

    base = (
        ρ = Fc_ρ - diss_ρ / 2,
        ρu = Fc_ρu - diss_ρu / 2,
        ρe = Fc_ρe - diss_ρe / 2,
    )
    return merge(base, _passive_tracer_fluxes(y⁻, y⁺, ū, normal, λ₂))
end

@inline FT_zero(x) = zero(typeof(x))

# Handle passive tracer fields (ρθ) not part of the Euler entropy structure.
function _passive_tracer_fluxes(y⁻, y⁺, ū, normal, λ₂)
    nt⁻, nt⁺ = unwrap(y⁻), unwrap(y⁺)
    if !hasfield(typeof(nt⁻), :ρθ)
        return NamedTuple()
    end
    # Central advection + upwind dissipation for the passive tracer ρθ.
    Fc_ρθ = ((nt⁻.ρθ + nt⁺.ρθ) / 2 * ū)' * normal
    diss_ρθ = λ₂ * (nt⁺.ρθ - nt⁻.ρθ)
    return (ρθ = Fc_ρθ - diss_ρθ / 2,)
end

# ---------------------------------------------------------------------------
# LDG / interior-penalty Laplacian face fluxes
# Volume term: WJ-weighted κ∇² via (−WJ)·κ·(−wdiv(grad q)); face τ[[q]].
# ---------------------------------------------------------------------------

"""
    add_lifting_flux_internal!(fn, dydt, args...)

Add *symmetric* face lifting terms at internal faces — the DG correction for
non-conservative (gradient / curl) terms, where both sides of a face receive
their own correction rather than equal-and-opposite fluxes:

    dydt⁻ += sWJ * fn(n̂⁻, argvals⁻, argvals⁺)
    dydt⁺ += sWJ * fn(n̂⁺, argvals⁺, argvals⁻)

with `n̂⁻ = -n̂⁺` the outward unit normals. For example, the strong-form DG
gradient of a scalar `q` is completed by `fn(n̂, (q⁻,), (q⁺,)) = ((q⁺ − q⁻)/2) * n̂`
(the lifting of `(q* − q⁻) n̂` with a central interface value `q*`).

`dydt` must be in mass-weighted residual form (`WJ * ∂Y/∂t`), matching
[`add_numerical_flux_internal!`](@ref). Implemented for pure 2D spectral
element spaces and for extruded spaces with 1D (plane) or 2D (e.g.
cubed-sphere) horizontal spectral elements.
"""
add_lifting_flux_internal!(fn, dydt, args...) = _add_lifting_flux_internal!(
    ClimaComms.device(axes(dydt)),
    fn,
    dydt,
    args...,
)

_add_lifting_flux_internal!(device, fn, dydt, args...) = error(
    "add_lifting_flux_internal! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

function _add_lifting_flux_internal!(
    ::ClimaComms.AbstractCPUDevice,
    fn,
    dydt,
    args...,
)
    space = axes(dydt)
    grid = Spaces.grid(space)
    if !(grid isa Grids.ExtrudedFiniteDifferenceGrid)
        return add_lifting_flux_internal_2d!(fn, dydt, args...)
    end
    if grid.horizontal_grid isa Grids.SpectralElementGrid2D
        return add_lifting_flux_internal_extruded_2d!(fn, dydt, args...)
    end
    @assert grid.horizontal_grid isa Grids.SpectralElementGrid1D
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    Nv = Spaces.nlevels(space)
    topology = Spaces.topology(space)
    local_geometry = Spaces.local_geometry_data(space)

    dydt_data = Fields.field_values(dydt)
    args_data = map(
        arg -> arg isa Fields.Field ? Fields.field_values(arg) : arg,
        args,
    )

    for v in 1:Nv
        for (elem⁻, face⁻, elem⁺, face⁺, _reversed) in
            Topologies.interior_faces(topology)

            i⁻ = face_node_index_1d(face⁻, Nq)
            i⁺ = face_node_index_1d(face⁺, Nq)

            lg⁻ = slab(local_geometry, v, elem⁻)[i⁻]
            sgeom⁻ = compute_surface_geometry_1d(lg⁻, face⁻)

            argvals⁻ = map(args_data) do arg
                val =
                    arg isa DataLayouts.AbstractData ?
                    slab(arg, v, elem⁻)[i⁻] : arg
                add_auto_broadcasters(val)
            end
            argvals⁺ = map(args_data) do arg
                val =
                    arg isa DataLayouts.AbstractData ?
                    slab(arg, v, elem⁺)[i⁺] : arg
                add_auto_broadcasters(val)
            end

            lift⁻ = add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻, argvals⁺))
            lift⁺ = add_auto_broadcasters(fn(-sgeom⁻.normal, argvals⁺, argvals⁻))

            dydt_slab⁻ = slab(dydt_data, v, elem⁻)
            dydt_slab⁺ = slab(dydt_data, v, elem⁺)
            dydt_slab⁻[i⁻] =
                dydt_slab⁻[i⁻] + (sgeom⁻.sWJ * lift⁻)
            dydt_slab⁺[i⁺] =
                dydt_slab⁺[i⁺] + (sgeom⁻.sWJ * lift⁺)
        end
    end
    return dydt
end

# Symmetric face lifting on a pure 2D spectral element space (e.g. a
# cubed-sphere shell); see [`add_lifting_flux_internal!`](@ref). Uses the
# precomputed internal surface geometry, like the pure-2D numerical flux loop.
function add_lifting_flux_internal_2d!(fn, dydt, args...)
    space = axes(dydt)
    grid = Spaces.grid(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    topology = Spaces.topology(space)
    internal_surface_geometry = grid.internal_surface_geometry
    dydt_bc = Base.broadcastable(dydt)
    args_bc =
        map(arg -> arg isa Fields.Field ? Base.broadcastable(arg) : arg, args)

    for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
        enumerate(Topologies.interior_faces(topology))

        internal_surface_geometry_slab = slab(internal_surface_geometry, iface)

        arg_slabs⁻ = map(arg -> slab(Fields.todata(arg), elem⁻), args_bc)
        arg_slabs⁺ = map(arg -> slab(Fields.todata(arg), elem⁺), args_bc)
        dydt_slab⁻ = slab(Fields.field_values(dydt_bc), elem⁻)
        dydt_slab⁺ = slab(Fields.field_values(dydt_bc), elem⁺)

        for q in 1:Nq
            sgeom⁻ = internal_surface_geometry_slab[q]

            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

            argvals⁻ = map(
                slab_ -> slab_ isa DataLayouts.DataLayout ? slab_[1, i⁻, j⁻, 1] : slab_,
                arg_slabs⁻,
            )
            argvals⁺ = map(
                slab_ -> slab_ isa DataLayouts.DataLayout ? slab_[1, i⁺, j⁺, 1] : slab_,
                arg_slabs⁺,
            )

            lift⁻ = add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻, argvals⁺))
            lift⁺ = add_auto_broadcasters(fn(-sgeom⁻.normal, argvals⁺, argvals⁻))

            dydt_slab⁻[1, i⁻, j⁻, 1] =
                dydt_slab⁻[1, i⁻, j⁻, 1] + (sgeom⁻.sWJ * lift⁻)
            dydt_slab⁺[1, i⁺, j⁺, 1] =
                dydt_slab⁺[1, i⁺, j⁺, 1] + (sgeom⁻.sWJ * lift⁺)
        end
    end
    return dydt
end

# Symmetric face lifting on an extruded 3D space (2D horizontal spectral
# elements × finite-difference vertical); see [`add_lifting_flux_internal!`](@ref).
function add_lifting_flux_internal_extruded_2d!(fn, dydt, args...)
    space = axes(dydt)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    Nv = Spaces.nlevels(space)
    topology = Spaces.topology(space)
    local_geometry = Spaces.local_geometry_data(space)
    FT = Spaces.undertype(space)
    (_, quad_weights) = Quadratures.quadrature_points(FT, quadrature_style)

    mstyle = _fd_metric_style(fn)
    dydt_data = Fields.field_values(dydt)
    args_data = map(
        arg -> arg isa Fields.Field ? Fields.field_values(arg) : arg,
        args,
    )

    for v in 1:Nv
        for (elem⁻, face⁻, elem⁺, face⁺, reversed) in
            Topologies.interior_faces(topology)

            dydt_slab⁻ = slab(dydt_data, v, elem⁻)
            dydt_slab⁺ = slab(dydt_data, v, elem⁺)

            for q in 1:Nq
                i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
                i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

                lg⁻ = slab(local_geometry, v, elem⁻)[1, i⁻, j⁻, 1]
                sgeom⁻ = compute_surface_geometry_extruded_2d(
                    lg⁻,
                    quad_weights,
                    face⁻,
                    i⁻,
                    j⁻,
                    mstyle,
                )

                argvals⁻ = map(args_data) do arg
                    val =
                        arg isa DataLayouts.AbstractData ?
                        slab(arg, v, elem⁻)[1, i⁻, j⁻, 1] : arg
                    add_auto_broadcasters(val)
                end
                argvals⁺ = map(args_data) do arg
                    val =
                        arg isa DataLayouts.AbstractData ?
                        slab(arg, v, elem⁺)[1, i⁺, j⁺, 1] : arg
                    add_auto_broadcasters(val)
                end

                lift⁻ =
                    add_auto_broadcasters(fn(sgeom⁻.normal, argvals⁻, argvals⁺))
                lift⁺ =
                    add_auto_broadcasters(fn(-sgeom⁻.normal, argvals⁺, argvals⁻))

                dydt_slab⁻[1, i⁻, j⁻, 1] =
                    dydt_slab⁻[1, i⁻, j⁻, 1] + (sgeom⁻.sWJ * lift⁻)
                dydt_slab⁺[1, i⁺, j⁺, 1] =
                    dydt_slab⁺[1, i⁺, j⁺, 1] + (sgeom⁻.sWJ * lift⁺)
            end
        end
    end
    return dydt
end

# ---------------------------------------------------------------------------
# Flux-differencing (split-form / FDDG) volume divergence
# ---------------------------------------------------------------------------

@inline _fd_add(a::NamedTuple, b::NamedTuple) = map(_fd_add, a, b)
@inline _fd_add(a, b) = a + b

@inline _fd_scale(c, x::NamedTuple) = map(v -> _fd_scale(c, v), x)
@inline _fd_scale(c, x) = c * x

# ---------------------------------------------------------------------------
# Metric-style trait: controls whether the flux-differencing divergence and
# the interface surface-geometry use the projected horizontal (UV) metric
# vector or the full curvilinear (UVW) metric vector.
#
# Flux functions signal their requirement by overloading `_fd_metric_style`.
# The default is :horizontal (existing behaviour, no terrain cross-terms).
# Curvilinear flux functions return Val{:curvilinear}() so that
# `add_flux_differencing_divergence!` and `add_numerical_flux_internal!`
# automatically pass UVW metric vectors without any caller-side changes.
# ---------------------------------------------------------------------------

"""
    _fd_metric_style(fn) -> Val{:horizontal}() | Val{:curvilinear}()

Trait controlling the metric projection used by
[`add_flux_differencing_divergence!`](@ref) and
[`add_numerical_flux_internal_extruded_2d!`](@ref).

- `Val{:horizontal}()` (default): project `J ∂ξ/∂x` onto `UVAxis`, dropping
  the radial (W) cross-term.  Correct for flat or gently terrain-following
  grids where cross-terms are negligible.
- `Val{:curvilinear}()`: retain the full `UVWAxis` vector.  Required for
  terrain-following grids where the radial component `J ∂ξⁱ/∂z` is non-zero;
  the DG–FD splitting is otherwise unchanged.

Overload this for any two-point or interface flux function that needs the
full curvilinear metric.
"""
@inline _fd_metric_style(::F) where {F} = Val{:horizontal}()

# Axis selector derived from the metric style (avoids repeating if-else).
@inline _metric_axis(::Val{:horizontal}) = Geometry.UVAxis()
@inline _metric_axis(::Val{:curvilinear}) = Geometry.UVWAxis()

# Metric-scaled contravariant basis vector J ∂ξʳᵒʷ/∂x.
# Dispatches on the metric style so callers only pass `fn2pt` and `row`.
@inline _fd_metric_vector(mstyle, local_geometry, row) = Geometry.project(
    _metric_axis(mstyle),
    local_geometry.J * local_geometry.∂ξ∂x[row, :],
)

"""
    add_flux_differencing_divergence!(fn2pt, dydt, y)

Add the horizontal flux-differencing (split-form / FDDG) volume divergence to
the mass-weighted residual `dydt`, following Souza et al. (2023, JAMES,
Eqs. 25-30): the collocation derivative matrix acts on symmetric two-point
flux evaluations between node pairs along each coordinate direction, with
arithmetic averaging of the metric terms ``\\{J a^i\\}``.

`fn2pt(nvec_a, nvec_b, y_a, y_b)` must return the two-point flux contracted
with the (non-unit) metric vectors of the two nodes, given in the local
orthonormal horizontal frame; it must be jointly linear in `(nvec_a, nvec_b)`,
symmetric under the exchange `(nvec_a, y_a) ↔ (nvec_b, y_b)`, and consistent
(`fn2pt(n, n, y, y)` is the pointwise flux `F(y)⋅n`). The kinetic-energy
(or entropy) properties of the discretization are determined entirely by this
choice — e.g. the Kennedy-Gruber flux gives the KEP property.

Passing both nodal metric vectors (rather than their average) lets flux
implementations average *contravariant nodal fluxes*, e.g.
``\\{ρ\\}\\,\\{u ⋅ Ja\\}``: the metric terms are then never differentiated on
their own, so free-stream preservation does not require the discrete metric
identities (which ClimaCore's analytic cubed-sphere metrics do not satisfy);
averaging the metrics separately instead (``\\{Ja\\}⋅F``) makes the mean flux
multiply the raw metric-identity defect and visibly degrades smooth-state
accuracy on the sphere.

The result is stored in *weak-equivalent* form: the strong-form
flux-differencing sum plus the lifting of the consistent own-side flux at
element-boundary nodes, so it is a drop-in replacement for the weak-form
volume step `dydt = hwdiv(F) * (-WJ)` and composes with
[`add_numerical_flux_internal!`](@ref) unchanged (the combination yields the
standard FDDG SAT ``F^* - F(y^-)⋅n̂``).

By the SBP property, the volume sum and the own-side lifts telescope exactly,
so the node sum of this contribution vanishes per element (local
conservation), and total conservation follows from the antisymmetry of the
interface flux.

Implemented for pure 2D spectral element spaces and extruded spaces with 2D
horizontal spectral elements. Metric terms are the analytic ClimaCore metrics;
free-stream preservation therefore holds to truncation (not machine) accuracy
on curved meshes.
"""
add_flux_differencing_divergence!(fn2pt, dydt, y) =
    _add_flux_differencing_divergence!(
        ClimaComms.device(axes(dydt)),
        fn2pt,
        dydt,
        y,
    )

_add_flux_differencing_divergence!(device, fn2pt, dydt, y) = error(
    "add_flux_differencing_divergence! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

function _add_flux_differencing_divergence!(
    ::ClimaComms.AbstractCPUDevice,
    fn2pt,
    dydt,
    y,
)
    space = axes(dydt)
    grid = Spaces.grid(space)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    FT = Spaces.undertype(space)
    (_, w) = Quadratures.quadrature_points(FT, quadrature_style)
    D = Quadratures.differentiation_matrix(FT, quadrature_style)
    topology = Spaces.topology(space)
    Nh = Topologies.nlocalelems(topology)
    local_geometry = Spaces.local_geometry_data(space)
    dydt_data = Fields.field_values(dydt)
    y_data = Fields.field_values(y)

    if grid isa Grids.ExtrudedFiniteDifferenceGrid
        @assert grid.horizontal_grid isa Grids.SpectralElementGrid2D
        Nv = Spaces.nlevels(space)
        for h in 1:Nh, v in 1:Nv
            _fd_divergence_slab!(
                fn2pt,
                slab(dydt_data, v, h),
                slab(y_data, v, h),
                slab(local_geometry, v, h),
                D,
                w,
                Nq,
            )
        end
    else
        @assert grid isa Grids.SpectralElementGrid2D
        for h in 1:Nh
            _fd_divergence_slab!(
                fn2pt,
                slab(dydt_data, h),
                slab(y_data, h),
                slab(local_geometry, h),
                D,
                w,
                Nq,
            )
        end
    end
    return dydt
end

# Per-node flux-differencing body, shared verbatim by the CPU slab loop and
# the CUDA kernel: `y_at(i, j)` / `lg_at(i, j)` are element-local accessors
# (slab getindex on the CPU, CartesianIndex getindex on the GPU). Returns the
# mass-weighted contribution (strong-form FD sum with coefficient
# −2 wᵢ wⱼ D, plus the own-side consistent-flux boundary lifts of the
# weak-equivalent form; the outward sWJ·n̂ is ±(J a¹) wⱼ / ±(J a²) wᵢ,
# matching compute_surface_geometry).
@inline function _fd_volume_node_total(
    fn2pt::F,
    y_at::Y,
    lg_at::L,
    D,
    w,
    ::Val{Nq},
    i,
    j,
) where {F, Y, L, Nq}
    mstyle = _fd_metric_style(fn2pt)
    lg = lg_at(i, j)
    Ja1 = _fd_metric_vector(mstyle, lg, 1)
    Ja2 = _fd_metric_vector(mstyle, lg, 2)
    y_ij = y_at(i, j)

    c1 = -2 * w[i] * w[j] * D[i, 1]
    total = fn2pt(
        c1 * Ja1,
        c1 * _fd_metric_vector(mstyle, lg_at(1, j), 1),
        y_ij,
        y_at(1, j),
    )
    c2 = -2 * w[i] * w[j] * D[j, 1]
    total = _fd_add(
        total,
        fn2pt(
            c2 * Ja2,
            c2 * _fd_metric_vector(mstyle, lg_at(i, 1), 2),
            y_ij,
            y_at(i, 1),
        ),
    )
    for k in 2:Nq
        c1 = -2 * w[i] * w[j] * D[i, k]
        t1 = fn2pt(
            c1 * Ja1,
            c1 * _fd_metric_vector(mstyle, lg_at(k, j), 1),
            y_ij,
            y_at(k, j),
        )
        c2 = -2 * w[i] * w[j] * D[j, k]
        t2 = fn2pt(
            c2 * Ja2,
            c2 * _fd_metric_vector(mstyle, lg_at(i, k), 2),
            y_ij,
            y_at(i, k),
        )
        total = _fd_add(total, _fd_add(t1, t2))
    end

    i == 1 &&
        (total = _fd_add(total, fn2pt(-w[j] * Ja1, -w[j] * Ja1, y_ij, y_ij)))
    i == Nq &&
        (total = _fd_add(total, fn2pt(w[j] * Ja1, w[j] * Ja1, y_ij, y_ij)))
    j == 1 &&
        (total = _fd_add(total, fn2pt(-w[i] * Ja2, -w[i] * Ja2, y_ij, y_ij)))
    j == Nq &&
        (total = _fd_add(total, fn2pt(w[i] * Ja2, w[i] * Ja2, y_ij, y_ij)))
    return total
end

function _fd_divergence_slab!(fn2pt, dydt_slab, y_slab, lg_slab, D, w, Nq)
    y_at = (a, b) -> y_slab[1, a, b, 1]
    lg_at = (a, b) -> lg_slab[1, a, b, 1]
    vNq = Val(Nq)
    for j in 1:Nq, i in 1:Nq
        total = _fd_volume_node_total(fn2pt, y_at, lg_at, D, w, vNq, i, j)
        dydt_slab[1, i, j, 1] =
            dydt_slab[1, i, j, 1] + add_auto_broadcasters(total)
    end
    return dydt_slab
end

# ---------------------------------------------------------------------------
# DG connectivity buffer (device-resident; used by the GPU face kernels)
# ---------------------------------------------------------------------------

"""
    DGConnectivity

Cached, device-resident connectivity and face geometry for the DG
internal-face operators (the DSS-buffer analog for DG):

- `faces`: `5 × nfaces` `Int32` matrix of interior faces
  `(elem⁻, face⁻, elem⁺, face⁺, reversed)`;
- `sgeom`: precomputed [`Geometry.SurfaceGeometry`](@ref) per
  `(q, level, face)` (level = 1 for pure 2D spaces), evaluated from the
  minus side exactly as the CPU loops do;
- a deterministic gather map from element boundary nodes to their face
  contributions, in ragged-array form (`node_*`, `node_offset`,
  `contrib_*`): each boundary node `(elem, i, j)` lists the
  `(face, side, q)` face-node slots that accumulate into it (2 entries at
  element corners, 1 elsewhere), sorted at construction so the GPU gather
  is bitwise deterministic.

Built once per space by [`dg_connectivity`](@ref) and stored with the array
type of the space's device (`ClimaComms.array_type`).
"""
struct DGConnectivity{FA, SG, IV}
    nfaces::Int
    nbnodes::Int
    faces::FA
    sgeom::SG
    node_elem::IV
    node_i::IV
    node_j::IV
    node_offset::IV
    contrib_face::IV
    contrib_side::IV
    contrib_q::IV
end

const _dg_connectivity_cache = IdDict{Any, Any}()
const _dg_connectivity_lock = ReentrantLock()

"""
    dg_connectivity(space)

Memoized [`DGConnectivity`](@ref) for `space` (keyed on the underlying grid
and the space type, so center/face extruded spaces get separate buffers).
"""
function dg_connectivity(space)
    key = (Spaces.grid(space), typeof(space))
    return lock(_dg_connectivity_lock) do
        get!(() -> build_dg_connectivity(space), _dg_connectivity_cache, key)
    end
end

function build_dg_connectivity(space)
    topology = Spaces.topology(space)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    FT = Spaces.undertype(space)
    grid = Spaces.grid(space)
    extruded = grid isa Grids.ExtrudedFiniteDifferenceGrid
    Nv = extruded ? Spaces.nlevels(space) : 1
    (_, w) = Quadratures.quadrature_points(FT, quadrature_style)
    DA = ClimaComms.array_type(topology)

    ifaces = collect(Topologies.interior_faces(topology))
    nfaces = length(ifaces)
    faces = Matrix{Int32}(undef, 5, nfaces)

    lg_host = Adapt.adapt(Array, Spaces.local_geometry_data(space))
    SG = Geometry.SurfaceGeometry{FT, Geometry.UVVector{FT}}
    sgeom = Array{SG}(undef, Nq, Nv, nfaces)

    # (elem, i, j) → list of (face, side, q); side 1 = minus, 2 = plus
    contrib = Dict{NTuple{3, Int}, Vector{NTuple{3, Int32}}}()
    for (f, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in enumerate(ifaces)
        faces[:, f] .=
            (elem⁻, face⁻, elem⁺, face⁺, reversed ? Int32(1) : Int32(0))
        for q in 1:Nq
            i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
            i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)
            push!(
                get!(() -> NTuple{3, Int32}[], contrib, (elem⁻, i⁻, j⁻)),
                (Int32(f), Int32(1), Int32(q)),
            )
            push!(
                get!(() -> NTuple{3, Int32}[], contrib, (elem⁺, i⁺, j⁺)),
                (Int32(f), Int32(2), Int32(q)),
            )
            for v in 1:Nv
                lg =
                    extruded ?
                    slab(lg_host, v, elem⁻)[1, i⁻, j⁻, 1] :
                    slab(lg_host, elem⁻)[1, i⁻, j⁻, 1]
                sgeom[q, v, f] = compute_surface_geometry_extruded_2d(
                    lg,
                    w,
                    face⁻,
                    i⁻,
                    j⁻,
                )
            end
        end
    end

    bnodes = sort!(collect(keys(contrib)))
    nbnodes = length(bnodes)
    node_elem = Vector{Int32}(undef, nbnodes)
    node_i = Vector{Int32}(undef, nbnodes)
    node_j = Vector{Int32}(undef, nbnodes)
    node_offset = Vector{Int32}(undef, nbnodes + 1)
    contrib_face = Int32[]
    contrib_side = Int32[]
    contrib_q = Int32[]
    node_offset[1] = 1
    for (n, key) in enumerate(bnodes)
        (elem, i, j) = key
        node_elem[n] = elem
        node_i[n] = i
        node_j[n] = j
        entries = sort!(contrib[key])
        for (f, side, q) in entries
            push!(contrib_face, f)
            push!(contrib_side, side)
            push!(contrib_q, q)
        end
        node_offset[n + 1] = node_offset[n] + length(entries)
    end

    return DGConnectivity(
        nfaces,
        nbnodes,
        DA(faces),
        DA(sgeom),
        DA(node_elem),
        DA(node_i),
        DA(node_j),
        DA(node_offset),
        DA(contrib_face),
        DA(contrib_side),
        DA(contrib_q),
    )
end

# ---------------------------------------------------------------------------
# Two-point (volume) and interface fluxes
# ---------------------------------------------------------------------------

"""
    kennedy_gruber_scalars_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux for the flux-form (ρ, ρe) subsystem (Souza et
al. 2023, JAMES, Eqs. 39 & 41): ``F_ρ = \\{ρ\\}\\{ũ\\}``,
``F_{ρe} = \\{ũ\\}(\\{ρ\\}\\{e\\} + \\{p\\})``, with `e` the specific total
energy and ``\\{ũ\\} = \\{u ⋅ nvec\\}`` the average of the **contravariant
nodal fluxes** (each node's velocity contracted with its own metric vector —
see [`add_flux_differencing_divergence!`](@ref) for why). Symmetric,
consistent, jointly linear in `(nvec_a, nvec_b)`.

State fields required: `ρ`, `ρe`, `e`, `p`, and `uv` (velocity in the local
orthonormal horizontal frame).
"""
function kennedy_gruber_scalars_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    return (ρ = ρ̄ * ūn, ρe = (ρ̄ * ē + p̄) * ūn)
end

"""
    kennedy_gruber_rusanov_scalars(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe) subsystem: [`kennedy_gruber_scalars_flux`](@ref)
as the central part plus a Rusanov penalty scaled by the state field `λ`
(the paper's interface choice, Souza et al. 2023).
"""
function kennedy_gruber_rusanov_scalars(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_scalars_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
    )
end

"""
    vi_kep_scalars_flux(nvec_a, nvec_b, y_a, y_b)

Two-point volume flux for the (ρ, ρe) subsystem, kinetic-energy compatible
with the vector invariant momentum pairing similar to (strong ``∇K`` + central K
lifting): ``F_ρ = \\{ρũ\\}`` (average of nodal contravariant mass fluxes —
NOT the Kennedy-Gruber ``\\{ρ\\}\\{ũ\\}``, which is KEP only for the
flux-form/fluctuation pairings) and ``F_{ρe} = \\{e\\}F_ρ + \\{p\\}\\{ũ\\}``
(advective part on ``F_ρ``, so constant-`e` states transport consistently).
The volume KE production then telescopes to the physical boundary flux
``ρ(u·n̂)K``, cancelled exactly by [`vi_kep_interface_scalars`](@ref) —
metric-transparent (flat and terrain-warped alike).

State fields required: `ρ`, `e`, `p`, `uv`.
"""
function vi_kep_scalars_flux(nvec_a, nvec_b, y_a, y_b)
    ũ_a = y_a.uv' * nvec_a
    ũ_b = y_b.uv' * nvec_b
    Fρ = (y_a.ρ * ũ_a + y_b.ρ * ũ_b) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (ũ_a + ũ_b) / 2
    return (ρ = Fρ, ρe = ē * Fρ + p̄ * ūn)
end

"""
    vi_kep_interface_scalars(normal, argvals⁻, argvals⁺)

Interface flux completing [`vi_kep_scalars_flux`](@ref): the same central
flux at the face plus a λ-Rusanov penalty on ``[\\![ρe]\\!]`` ONLY — a
density-jump penalty produces sign-indefinite face kinetic energy
(``+λ/2[\\![ρ]\\!][\\![K]\\!]``) under the vector-invariant pairing, so
exact KEP requires a central mass flux. Velocity dissipation:
[`rho_weighted_jump_penalty_lift`](@ref).
"""
function vi_kep_interface_scalars(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = vi_kep_scalars_flux(normal, normal, y⁻, y⁺)
    return (ρ = F.ρ, ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe))
end

"""
    VIESInterfaceScalars(γm1)

Entropy-dissipative (ρ, ρe) interface flux (callable, `γm1 = γ − 1`):
the [`vi_kep_scalars_flux`](@ref) central part (mass stays central — the
exact VI kinetic-energy budget is preserved) plus ρe dissipation in the entropy
variable ``v = ∂S/∂ρe|_ρ = −ρ/p``:

    F*_ρe = F_central − (λ/2)\\,w̄\\,(v⁺ − v⁻),   w̄ = p̄²/((γ−1)ρ̄)

Entropy production ``−(λ/2)w̄[\\![v]\\!]² ≤ 0`` by construction; KE-inert;
reduces to internal-energy Rusanov near constant states (replacing the
entropy-indefinite ``[\\![ρe]\\!]`` Rusanov of
[`vi_kep_interface_scalars`](@ref)).

State fields required: `ρ`, `e`, `p`, `λ`, `uv`.
"""
struct VIESInterfaceScalars{T} <: AbstractNumericalFlux
    γm1::T
end

function (fn::VIESInterfaceScalars)(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = vi_kep_scalars_flux(normal, normal, y⁻, y⁺)
    p̄ = (y⁻.p + y⁺.p) / 2
    ρ̄ = (y⁻.ρ + y⁺.ρ) / 2
    w̄ = p̄^2 / (fn.γm1 * ρ̄)
    v⁻ = -y⁻.ρ / y⁻.p
    v⁺ = -y⁺.ρ / y⁺.p
    return (ρ = F.ρ, ρe = F.ρe - λ / 2 * w̄ * (v⁺ - v⁻))
end

"""
    VIES2InterfaceScalars(γm1)

[`VIESInterfaceScalars`](@ref) with acoustic Roe dissipation -
the same central flux and entropy-variable ρe
dissipation, with the two acoustic Roe waves additionally damped at
``|û_n ± ĉ|``:

    α± = ([[p′]] ± ρ̄ĉ[[u_n]])/(2ĉ²)
    F*_ρ  −= (s₊α₊ + s₋α₋)/2
    F*_ρe −= (s₊α₊(h̄ + ĉû_n) + s₋α₋(h̄ − ĉû_n))/2

The contact discontinuity (``α₀ = [[ρ]] − [[p]]/ĉ²``) stays exactly central — a
density-jump penalty is sign-indefinite in face kinetic energy under the
vector-invariant pairing.

State fields required: `ρ`, `e`, `p`, `p′`, `λ`, `uv`.
"""
struct VIES2InterfaceScalars{T} <: AbstractNumericalFlux
    γm1::T
end

function (fn::VIES2InterfaceScalars)(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = vi_kep_scalars_flux(normal, normal, y⁻, y⁺)
    p̄ = (y⁻.p + y⁺.p) / 2
    ρ̄ = (y⁻.ρ + y⁺.ρ) / 2
    # thermal channel: entropy-variable ρe dissipation (as VIES)
    w̄ = p̄^2 / (fn.γm1 * ρ̄)
    Δv = y⁻.ρ / y⁻.p - y⁺.ρ / y⁺.p
    # acoustic channels: Roe α± from ([[p′]], [[u_n]]); contact central
    ĉ = sqrt((fn.γm1 + 1) * p̄ / ρ̄)
    ûₙ = ((y⁻.uv + y⁺.uv)' * normal) / 2
    Δuₙ = (y⁺.uv - y⁻.uv)' * normal
    Δp′ = y⁺.p′ - y⁻.p′
    α₊ = (Δp′ + ρ̄ * ĉ * Δuₙ) / (2 * ĉ^2)
    α₋ = (Δp′ - ρ̄ * ĉ * Δuₙ) / (2 * ĉ^2)
    s₊ = abs(ûₙ + ĉ)
    s₋ = abs(ûₙ - ĉ)
    h̄ = (y⁻.e + y⁻.p / y⁻.ρ + y⁺.e + y⁺.p / y⁺.ρ) / 2
    Dρ = s₊ * α₊ + s₋ * α₋
    Dρe = s₊ * α₊ * (h̄ + ĉ * ûₙ) + s₋ * α₋ * (h̄ - ĉ * ûₙ)
    # Δv = v⁺ − v⁻ (v = −ρ/p), matching VIESInterfaceScalars' −(λ/2)w̄[[v]]
    return (ρ = F.ρ - Dρ / 2, ρe = F.ρe - λ / 2 * w̄ * Δv - Dρe / 2)
end

"""
    rho_weighted_jump_penalty_lift(normal, (q⁻, ρ⁻, λ⁻), (q⁺, ρ⁺, λ⁺))

ρ-weighted λ jump penalty for velocity components (rate
``\\max(λ)/2·\\{ρ\\}/ρ_{own}``): the face kinetic-energy tendency is
``−\\max(λ)/2\\,\\{ρ\\}|[\\![u]\\!]|² ≤ 0`` EXACTLY (the plain
[`jump_penalty_lift`](@ref) only to ``O([\\![ρ]\\!])``). Use through
[`lifting_correction`](@ref) with argument fields `(u_c, ρ, λ)`.
"""
rho_weighted_jump_penalty_lift(normal, (q⁻, ρ⁻, λ⁻), (q⁺, ρ⁺, λ⁺)) =
    max(λ⁻, λ⁺) / 2 * ((ρ⁻ + ρ⁺) / 2 / ρ⁻) * (q⁺ - q⁻)

"""
    kennedy_gruber_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux for the full (ρ, ρe, ρu⃗) system with momentum
carried in GLOBAL CARTESIAN components (Souza et al. 2023): the basis is
constant, so component-wise flux differencing retains the KEP property with
no curvature source terms. Contravariant nodal fluxes are averaged (each
node's own metric vector).

State fields required: `ρ`, `e`, `p`, `uv` (velocity, local orthonormal
horizontal frame), `u1`, `u2`, `u3` (Cartesian velocity components), and
`E1`, `E2`, `E3` (the tangential projections of the Cartesian unit vectors
ê₁, ê₂, ê₃, each as a `UVVector` — position-dependent on the sphere but
state-independent). The pressure flux for component ``c`` is
``\\{p\\}\\{ê_c ⋅ nvec\\}``.
"""
function kennedy_gruber_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    return (
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn + p̄ * Ē1n,
        ρu2 = ρ̄ * ū2 * ūn + p̄ * Ē2n,
        ρu3 = ρ̄ * ū3 * ūn + p̄ * Ē3n,
    )
end

"""
    kennedy_gruber_rusanov_cartesian(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe, ρu⃗-Cartesian) system:
[`kennedy_gruber_cartesian_flux`](@ref) as the central part plus a Rusanov
penalty scaled by the state field `λ` (jumps of the conserved variables;
momentum jumps via `ρ * u_c`). Additional state fields: `ρe`, `λ`.
"""
function kennedy_gruber_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
        ρu1 = F.ρu1 - λ / 2 * (y⁺.ρ * y⁺.u1 - y⁻.ρ * y⁻.u1),
        ρu2 = F.ρu2 - λ / 2 * (y⁺.ρ * y⁺.u2 - y⁻.ρ * y⁻.u2),
        ρu3 = F.ρu3 - λ / 2 * (y⁺.ρ * y⁺.u3 - y⁻.ρ * y⁻.u3),
    )
end

"""
    kennedy_gruber_roe_cartesian(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe, ρu⃗-Cartesian) system:
[`kennedy_gruber_cartesian_flux`](@ref) as the central part plus ROE-TYPE
wave-selective dissipation (Souza et al. 2023 interface choice): acoustic
waves are damped at ``|û_n ± ĉ|`` but entropy and shear jumps at
``max(|û_n|, ĉ/20)`` — so stationary balanced structure (contact/shear
jumps with ``u_n ≈ 0``) receives ~5% of Rusanov's uniform ``|u| + c``
dissipation (the Harten-type floor is required: see inline comment).
The energy eigen-component uses ``B = Ĥ - ĉ²/(γ-1)``, which absorbs the
geopotential and vertical-kinetic contributions of ``ρe`` without needing
them separately (Φ is single-valued at the face). Same state fields as
[`kennedy_gruber_rusanov_cartesian`](@ref); requires `γ` jumps consistent
with `p`/`e` (dry ideal gas).
"""
function kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    # keep the working precision of the state (Float32 fields stay Float32)
    γd = oftype(y⁻.ρ, γ_dry)
    # face normal in Cartesian components (E_c single-valued at the node)
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    # Roe-averaged state
    s⁻ = sqrt(y⁻.ρ)
    s⁺ = sqrt(y⁺.ρ)
    ρ̂ = s⁻ * s⁺
    a⁻ = s⁻ / (s⁻ + s⁺)
    a⁺ = 1 - a⁻
    û1 = a⁻ * y⁻.u1 + a⁺ * y⁺.u1
    û2 = a⁻ * y⁻.u2 + a⁺ * y⁺.u2
    û3 = a⁻ * y⁻.u3 + a⁺ * y⁺.u3
    Ĥ = a⁻ * (y⁻.e + y⁻.p / y⁻.ρ) + a⁺ * (y⁺.e + y⁺.p / y⁺.ρ)
    ĉ = a⁻ * sqrt(γd * y⁻.p / y⁻.ρ) + a⁺ * sqrt(γd * y⁺.p / y⁺.ρ)
    ûn = û1 * n1 + û2 * n2 + û3 * n3
    # jumps and wave amplitudes
    Δρ = y⁺.ρ - y⁻.ρ
    Δp = y⁺.p - y⁻.p
    Δu1 = y⁺.u1 - y⁻.u1
    Δu2 = y⁺.u2 - y⁻.u2
    Δu3 = y⁺.u3 - y⁻.u3
    Δun = Δu1 * n1 + Δu2 * n2 + Δu3 * n3
    α₊ = (Δp + ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₋ = (Δp - ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₀ = Δρ - Δp / ĉ^2
    s₊ = abs(ûn + ĉ)
    s₋ = abs(ûn - ĉ)
    # Harten-type entropy floor on the contact/shear speed: pure |û_n|
    # leaves density jumps in near-stagnant columns (e.g. the model top)
    # undamped, and the min-ρ cell can drain unchecked (observed: secular
    # min-ρ collapse from day ~2.3 of a perturbed baroclinic wave at
    # zelem = 30). ε = 0.05 retains 5% of the Rusanov ρ-jump dissipation
    # while keeping the spurious forcing of balanced jets ~20× below
    # Rusanov. The acoustic speeds need no floor (|û_n| ≪ ĉ here).
    s₀ = max(abs(ûn), ĉ / 20)
    Δut1 = Δu1 - Δun * n1
    Δut2 = Δu2 - Δun * n2
    Δut3 = Δu3 - Δun * n3
    B = Ĥ - ĉ^2 / (γd - 1)
    Dρ = s₊ * α₊ + s₋ * α₋ + s₀ * α₀
    Dρu1 =
        s₊ * α₊ * (û1 + ĉ * n1) + s₋ * α₋ * (û1 - ĉ * n1) +
        s₀ * (α₀ * û1 + ρ̂ * Δut1)
    Dρu2 =
        s₊ * α₊ * (û2 + ĉ * n2) + s₋ * α₋ * (û2 - ĉ * n2) +
        s₀ * (α₀ * û2 + ρ̂ * Δut2)
    Dρu3 =
        s₊ * α₊ * (û3 + ĉ * n3) + s₋ * α₋ * (û3 - ĉ * n3) +
        s₀ * (α₀ * û3 + ρ̂ * Δut3)
    Dρe =
        s₊ * α₊ * (Ĥ + ĉ * ûn) + s₋ * α₋ * (Ĥ - ĉ * ûn) +
        s₀ * (α₀ * B + ρ̂ * (û1 * Δut1 + û2 * Δut2 + û3 * Δut3))
    return (
        ρ = F.ρ - Dρ / 2,
        ρe = F.ρe - Dρe / 2,
        ρu1 = F.ρu1 - Dρu1 / 2,
        ρu2 = F.ρu2 - Dρu2 / 2,
        ρu3 = F.ρu3 - Dρu3 / 2,
    )
end
# dry-air ratio of specific heats used by the Roe linearization
const γ_dry = 7 / 5

# ---------------------------------------------------------------------------
# Curvilinear (full UVW metric) variants of the KG and Roe Cartesian fluxes.
#
# These extend the flat-sphere formulas to terrain-following grids by retaining
# the W (radial) component of the face-normal metric vector `Ja^i`.  The
# DG–FD split is unchanged; only the horizontal metric projection is widened.
#
# State fields required (in addition to `ρ`, `e`, `p`, `u1`, `u2`, `u3`, `λ`):
#   uvw  :: UVWVector  — full velocity (uE, uN, w_c) in local orthonormal frame
#   Ec1/Ec2/Ec3 :: UVWVector — full projections of ê₁/ê₂/ê₃ including eR_c
#
# `_fd_metric_style` is overloaded so that `add_flux_differencing_divergence!`
# and `add_numerical_flux_internal_extruded_2d!` automatically pass UVW metric
# vectors — no caller-side changes needed.
# ---------------------------------------------------------------------------

"""
    kennedy_gruber_cartesian_flux_curvilinear(nvec_a, nvec_b, y_a, y_b)

Curvilinear extension of [`kennedy_gruber_cartesian_flux`](@ref): the metric
vectors `nvec_a`, `nvec_b` and the state velocity `uvw` / basis projections
`Ec1/Ec2/Ec3` are `UVWVector`s, so the terrain metric cross-terms
``Ja^i_W = J \\partial ξ^i / \\partial z`` are retained.

On a flat grid ``Ja^i_W = 0`` and the formula reduces bitwise to
[`kennedy_gruber_cartesian_flux`](@ref).
"""
function kennedy_gruber_cartesian_flux_curvilinear(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uvw' * nvec_a + y_b.uvw' * nvec_b) / 2
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    Ē1n = (y_a.Ec1' * nvec_a + y_b.Ec1' * nvec_b) / 2
    Ē2n = (y_a.Ec2' * nvec_a + y_b.Ec2' * nvec_b) / 2
    Ē3n = (y_a.Ec3' * nvec_a + y_b.Ec3' * nvec_b) / 2
    return (
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn + p̄ * Ē1n,
        ρu2 = ρ̄ * ū2 * ūn + p̄ * Ē2n,
        ρu3 = ρ̄ * ū3 * ūn + p̄ * Ē3n,
    )
end
@inline _fd_metric_style(::typeof(kennedy_gruber_cartesian_flux_curvilinear)) =
    Val{:curvilinear}()

"""
    kennedy_gruber_roe_cartesian_curvilinear(normal, argvals⁻, argvals⁺)

Curvilinear extension of [`kennedy_gruber_roe_cartesian`](@ref): the face
`normal` is a `UVWVector` so the terrain metric cross-term is included in the
normal-velocity computation and the Roe wave amplitudes.  The shear energy
term is computed as ``\\hat{u}_{uvw} \\cdot \\Delta u_{\\perp}`` in the full
UVW frame, capturing the radial-shear contribution through tilted faces.

On a flat grid the W component of `normal` is zero and the Roe wave speeds
and amplitudes reduce to those of [`kennedy_gruber_roe_cartesian`](@ref);
the only flat-grid difference is the inclusion of the vertical-velocity shear
``\\hat{w}_r \\, \\Delta w_r`` in the energy dissipation (a correction previously
absent because ``w_r`` was not in the state).

Additional state fields beyond [`kennedy_gruber_roe_cartesian`](@ref):
`uvw` (`UVWVector`) and `Ec1/Ec2/Ec3` (`UVWVector`).
"""
function kennedy_gruber_roe_cartesian_curvilinear(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_cartesian_flux_curvilinear(normal, normal, y⁻, y⁺)
    γd = oftype(y⁻.ρ, γ_dry)
    # face normal in Cartesian components (includes W terrain cross-term)
    n1 = y⁻.Ec1' * normal
    n2 = y⁻.Ec2' * normal
    n3 = y⁻.Ec3' * normal
    # Roe-averaged state
    s⁻ = sqrt(y⁻.ρ)
    s⁺ = sqrt(y⁺.ρ)
    ρ̂ = s⁻ * s⁺
    a⁻ = s⁻ / (s⁻ + s⁺)
    a⁺ = 1 - a⁻
    û1 = a⁻ * y⁻.u1 + a⁺ * y⁺.u1
    û2 = a⁻ * y⁻.u2 + a⁺ * y⁺.u2
    û3 = a⁻ * y⁻.u3 + a⁺ * y⁺.u3
    ûuvw = a⁻ * y⁻.uvw + a⁺ * y⁺.uvw   # Roe-avg full velocity in UVW frame
    Ĥ = a⁻ * (y⁻.e + y⁻.p / y⁻.ρ) + a⁺ * (y⁺.e + y⁺.p / y⁺.ρ)
    ĉ = a⁻ * sqrt(γd * y⁻.p / y⁻.ρ) + a⁺ * sqrt(γd * y⁺.p / y⁺.ρ)
    # full normal velocity: tangential Cartesian + radial through terrain normal
    ûn = ûuvw' * normal
    # jumps
    Δρ = y⁺.ρ - y⁻.ρ
    Δp = y⁺.p - y⁻.p
    Δu1 = y⁺.u1 - y⁻.u1
    Δu2 = y⁺.u2 - y⁻.u2
    Δu3 = y⁺.u3 - y⁻.u3
    Δuvw = y⁺.uvw - y⁻.uvw
    Δun = Δuvw' * normal  # full normal-velocity jump (includes radial)
    α₊ = (Δp + ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₋ = (Δp - ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₀ = Δρ - Δp / ĉ^2
    s₊ = abs(ûn + ĉ)
    s₋ = abs(ûn - ĉ)
    s₀ = max(abs(ûn), ĉ / 20)
    # transverse jumps of tangential Cartesian components
    Δut1 = Δu1 - Δun * n1
    Δut2 = Δu2 - Δun * n2
    Δut3 = Δu3 - Δun * n3
    B = Ĥ - ĉ^2 / (γd - 1)
    # full shear energy in UVW frame: û_uvw · Δu_transverse
    # = û_uvw · Δuvw − ûn · Δun  (Pythagoras on the decomposition)
    shear_E = ûuvw' * Δuvw - ûn * Δun
    Dρ = s₊ * α₊ + s₋ * α₋ + s₀ * α₀
    Dρu1 =
        s₊ * α₊ * (û1 + ĉ * n1) + s₋ * α₋ * (û1 - ĉ * n1) +
        s₀ * (α₀ * û1 + ρ̂ * Δut1)
    Dρu2 =
        s₊ * α₊ * (û2 + ĉ * n2) + s₋ * α₋ * (û2 - ĉ * n2) +
        s₀ * (α₀ * û2 + ρ̂ * Δut2)
    Dρu3 =
        s₊ * α₊ * (û3 + ĉ * n3) + s₋ * α₋ * (û3 - ĉ * n3) +
        s₀ * (α₀ * û3 + ρ̂ * Δut3)
    Dρe =
        s₊ * α₊ * (Ĥ + ĉ * ûn) + s₋ * α₋ * (Ĥ - ĉ * ûn) +
        s₀ * (α₀ * B + ρ̂ * shear_E)
    return (
        ρ = F.ρ - Dρ / 2,
        ρe = F.ρe - Dρe / 2,
        ρu1 = F.ρu1 - Dρu1 / 2,
        ρu2 = F.ρu2 - Dρu2 / 2,
        ρu3 = F.ρu3 - Dρu3 / 2,
    )
end
# Interface flux: `add_numerical_flux_internal!` derives the face-normal style
# from `fn`, so this ensures the normal is a `UVWVector`. Must appear AFTER
# the function definition so `typeof(...)` resolves correctly.
@inline _fd_metric_style(::typeof(kennedy_gruber_roe_cartesian_curvilinear)) =
    Val{:curvilinear}()

"""
    logmean(a, b)

Numerically stable logarithmic mean ``(a − b)/(\\log a − \\log b)`` for
positive arguments (Ismail & Roe 2009, Appendix B): with ``ζ = a/b`` and
``f = (ζ−1)/(ζ+1)``, ``\\log ζ = 2f(1 + f²/3 + f⁴/5 + …)``, so the mean is
``(a+b)/(2F)`` with the series used for ``f² < 10⁻²`` (exact limit
``(a+b)/2`` at ``a = b``) and the closed form elsewhere.
"""
@inline function logmean(a, b)
    ζ = a / b
    f = (ζ - 1) / (ζ + 1)
    u = f * f
    F = if u < oftype(u, 1e-2)
        1 + u / 3 + u * u / 5 + u * u * u / 7
    else
        log(ζ) / (2 * f)
    end
    return (a + b) / (2 * F)
end

"""
    wb_gravity_cartesian_increment(nvec_a, nvec_b, y_a, y_b)

Non-symmetric two-point FLUCTUATION term for the geopotential (gravity)
source ``ρ ∂Φ/∂x_c`` of the (ρ, ρe, ρu⃗-Cartesian) system, following
Waruszewski et al. (2022, JCP 468:111507, Eqs. 73–76): each Cartesian
momentum component ``c`` receives

    (1/2) ρ̂ (Φ_b − Φ_a) \\{ê_c ⋅ nvec\\},
    ρ̂ = \\{b\\} \\, \\mathrm{logmean}(ρ_a, ρ_b) / b_a,

with ``b = ρ/(2p)`` the inverse temperature and the OWN node first (the
[`add_flux_differencing_divergence!`](@ref) row convention — the term is
genuinely non-symmetric through ``1/b_a``). The geopotential is
single-valued at element faces, so the fluctuation vanishes there: the term
is VOLUME-ONLY and the interface fluxes need no change. It also vanishes
for `y_a == y_b`, so the kernel's own-side boundary lifts contribute
nothing (pure strong-form fluctuation, like
[`kg_massflux_fluctuation`](@ref)).

Well-balance: added to a volume flux whose momentum pressure term is the
arithmetic ``\\{p\\}``, the combined PGF + gravity operator cancels
PAIRWISE (any node pair) on isothermal hydrostatic states
``p ∝ \\exp(−Φ/(R_d T₀))``, for which
``\\mathrm{logmean}(ρ_a, ρ_b)(Φ_b − Φ_a) = −(p_b − p_a)`` is an identity.
The pair sum then telescopes to ``p_n\\,D(ê_c ⋅ Ja)`` — so the rest state
is an EXACT discrete steady state wherever the discrete metric identity
closes for the directions this kernel spans: machine zero on affine 2D
meshes; the free-stream-level GCL defect on smooth curved 2D meshes. On
WARPED EXTRUDED grids the horizontal identity alone does not close
(``Σ_{k=1,2} ∂_{ξ^k}(Ja^k) = −∂_{ξ^3}(Ja³) ≠ 0``, O(slope)): the
fluctuation removes the pairwise thermodynamic imbalance but the geometric
remainder requires the vertical (cross-term) direction — supply it in the
staggered/HEVI vertical discretization to complete 3D well-balance
(Waruszewski et al.'s scheme is exact because its flux differencing spans
all three directions). General hydrostatic profiles reduce to the same
accuracy via the ``p′ = p − p_{ref}`` splitting (isothermal reference in
the flux, deviation elsewhere).

State fields required: `ρ`, `p`, `Φ`, `E1`, `E2`, `E3`.
"""
@inline function wb_gravity_cartesian_increment(nvec_a, nvec_b, y_a, y_b)
    b_a = y_a.ρ / (2 * y_a.p)
    b_b = y_b.ρ / (2 * y_b.p)
    ρ̂ = (b_a + b_b) / 2 * logmean(y_a.ρ, y_b.ρ) / b_a
    gΔΦ = ρ̂ * (y_b.Φ - y_a.Φ) / 2
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    return (ρu1 = gΔΦ * Ē1n, ρu2 = gΔΦ * Ē2n, ρu3 = gΔΦ * Ē3n)
end

"""
    kennedy_gruber_gravity_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

[`kennedy_gruber_cartesian_flux`](@ref) plus the well-balanced geopotential
fluctuation [`wb_gravity_cartesian_increment`](@ref) in the momentum
components (the gravity term rides the same ``\\{ê_c ⋅ nvec\\}`` direction
as the pressure, exactly as ``δ_{ik}(p^* + \\tfrac12 ρ̂ [\\![Φ]\\!])`` in
Waruszewski et al. 2022, Eq. 76). Retains the KG kinetic-energy property —
the added term contributes only the physical ``ρ u ⋅ ∇Φ`` KE ↔ PE
exchange — and total energy remains conservative (the ρe flux is
untouched; Φ enters it through the specific total energy `e`).

REQUIREMENTS: the specific total energy `e` must INCLUDE the geopotential
(``ρe ⊃ ρΦ``, so `p` consistently subtracts it), and the caller must NOT
apply a separate pointwise horizontal gravity source — the fluctuation IS
the horizontal ``ρ∇Φ`` term, in discretely balanced form. State fields:
those of [`kennedy_gruber_cartesian_flux`](@ref) plus `Φ`.
"""
function kennedy_gruber_gravity_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    F = kennedy_gruber_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    G = wb_gravity_cartesian_increment(nvec_a, nvec_b, y_a, y_b)
    return (
        ρ = F.ρ,
        ρe = F.ρe,
        ρu1 = F.ρu1 + G.ρu1,
        ρu2 = F.ρu2 + G.ρu2,
        ρu3 = F.ρu3 + G.ρu3,
    )
end

"""
    kg_massflux_fluctuation(nvec_a, nvec_b, y_a, y_b)

Non-symmetric two-point FLUCTUATION form for the advective operator
``(u·∇_h)u_c`` acting on velocity components in velocity (non-conservative)
form, driven by the Kennedy-Gruber mass flux:
``P^\\#_c(a, b) = F^\\#_ρ(a, b)\\,(u_{c,b} - u_{c,a})/2`` with
``F^\\#_ρ = \\{ρ\\}\\{u ⋅ nvec\\}`` (contravariant nodal fluxes averaged).

Pass to [`add_flux_differencing_divergence!`](@ref); the own-side boundary
lifts evaluate to zero (the jump vanishes for `y_a == y_b`), so the kernel
degenerates to the pure strong-form fluctuation sum. The mass-weighted
result divided by ``ρ\\,WJ`` is ``-(u·∇_h)u_c``, replacing BOTH the
relative-vorticity cross product and the horizontal-KE gradient of the
vector-invariant form.

KE compatibility with the KG mass flux (the fluctuation-form analog of the
KEP property): ``K_i F^\\#_ρ(i,j) + u_{c,i} P^\\#_c(i,j)`` (summed over
components) equals ``F^\\#_ρ(i,j)\\,(u_i · u_j)/2``, which is symmetric, so
the volume kinetic-energy production telescopes to face terms; complete them
with [`advective_fluctuation_lift`](@ref). The advected components must be
in a globally constant frame (e.g. Cartesian) — position-dependent frames
reintroduce curvature terms the jumps cannot see.

State fields required: `ρ`, `uv`, `u1`, `u2`, `u3`.
"""
function kg_massflux_fluctuation(nvec_a, nvec_b, y_a, y_b)
    F = ((y_a.ρ + y_b.ρ) / 2) * ((y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2)
    return (
        u1 = F * (y_b.u1 - y_a.u1) / 2,
        u2 = F * (y_b.u2 - y_a.u2) / 2,
        u3 = F * (y_b.u3 - y_a.u3) / 2,
    )
end

"""
    advective_fluctuation_lift(normal, argvals⁻, argvals⁺)

Per-component face SAT completing [`kg_massflux_fluctuation`](@ref): the
EXACT velocity-variables transform of the flux-form KG central face
treatment, ``δu_c = (δ(ρ u_c) - u_c\\,δρ)/ρ`` applied to the central
interface fluxes — the own-flux terms cancel, leaving each side
``-\\{ρ\\}(\\{uv\\} ⋅ n̂_{side})\\,(u_c^{other} - u_c^{side})/2``.
Because it is the exact transform of a KE-consistent face treatment, the
face kinetic-energy bookkeeping is identical to the flux form's. (The same
transform of the Rusanov jumps reproduces the λ velocity-jump penalties
with a ``ρ^{other}/ρ^{side}`` weight; the plain penalties are their
constant-ρ limit and provide the face dissipation.) NOTE the sign: the
naive "central lifting" sign (+) is anti-consistent and exponentially
unstable at O(jump). Use through [`lifting_correction`](@ref) with argument
fields `(u_c, ρ, uv)`; divide the result by ``ρ`` for the velocity
tendency.
"""
function advective_fluctuation_lift(normal, argvals⁻, argvals⁺)
    u_c⁻, ρ⁻, uv⁻ = argvals⁻
    u_c⁺, ρ⁺, uv⁺ = argvals⁺
    F = ((ρ⁻ + ρ⁺) / 2) * (((uv⁻ + uv⁺) / 2)' * normal)
    return -F * (u_c⁺ - u_c⁻) / 2
end

"""
    kennedy_gruber_height_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber-style two-point mass flux ``\\{h\\}\\{u ⋅ nvec\\}`` for the
shallow-water height equation (contravariant nodal fluxes averaged). State
fields required: `h`, `uv`.
"""
kennedy_gruber_height_flux(nvec_a, nvec_b, y_a, y_b) =
    ((y_a.h + y_b.h) / 2) * ((y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2)

"""
    kennedy_gruber_rusanov_height(normal, argvals⁻, argvals⁺)

Interface flux for the shallow-water height equation:
[`kennedy_gruber_height_flux`](@ref) central part plus a Rusanov penalty
scaled by the state field `λ`.
"""
function kennedy_gruber_rusanov_height(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    return kennedy_gruber_height_flux(normal, normal, y⁻, y⁺) -
           λ / 2 * (y⁺.h - y⁻.h)
end

# ---------------------------------------------------------------------------
# DG face-function library for non-conservative (vector-invariant) terms
# ---------------------------------------------------------------------------

"""
    central_gradient_lift(normal, (q⁻,), (q⁺,))

Symmetric central lifting completing the strong-form DG gradient of a scalar:
each side adds ``(q^* - q_{side}) n̂_{side}`` with central ``q^*``, i.e.
``((q⁺ - q⁻)/2)\\,n̂`` on the minus side. Use with
[`add_lifting_flux_internal!`](@ref) / [`lifting_correction`](@ref).
"""
central_gradient_lift(normal, (q⁻,), (q⁺,)) = ((q⁺ - q⁻) / 2) * normal

"""
    central_curl3_lift(normal, (u⁻, v⁻), (u⁺, v⁺))

Central lifting for the radial component of the horizontal curl:
``r̂ ⋅ (n̂ × (u^* - u_{side}))`` from the tangential jumps of the orthonormal
velocity components `(u, v)`.
"""
central_curl3_lift(normal, (u⁻, v⁻), (u⁺, v⁺)) =
    (
        normal.components.data.:1 * (v⁺ - v⁻) -
        normal.components.data.:2 * (u⁺ - u⁻)
    ) / 2

"""
    central_curl12_lift(normal, (w⁻,), (w⁺,))

Central lifting for the horizontal components of ``∇ × (w r̂)``:
``n̂ × r̂\\,(w^* - w_{side})``, returned as a `UVVector`.
"""
central_curl12_lift(normal, (w⁻,), (w⁺,)) =
    ((w⁺ - w⁻) / 2) *
    Geometry.UVVector(normal.components.data.:2, -normal.components.data.:1)

"""
    jump_penalty_lift(normal, (q⁻, λ⁻), (q⁺, λ⁺))

λ-scaled interface penalty: each side relaxes toward its neighbor at rate
``\\max(λ⁻, λ⁺)/2``.
"""
jump_penalty_lift(normal, (q⁻, λ⁻), (q⁺, λ⁺)) = max(λ⁻, λ⁺) / 2 * (q⁺ - q⁻)

"""
    lifting_correction(fn, ::Type{T}, args...)

WJ-normalized DG face-lifting correction field of element type `T`: applies
[`add_lifting_flux_internal!`](@ref) with face function `fn` to a zero
residual on the space of `args[1]` and divides by `WJ`. The result is the
correction to the corresponding element-local strong-form operator.
"""
function lifting_correction(fn, ::Type{T}, args...) where {T}
    space = axes(args[1])
    lgeom = Fields.local_geometry_field(space)
    r = similar(args[1], T)
    fill!(parent(r), 0)
    add_lifting_flux_internal!(fn, r, args...)
    return r ./ lgeom.WJ
end

"""
    ldg_laplacian_tendency(q, ρ_weight, κ, τ)

WJ-normalized LDG / SIPG Laplacian tendency approximating
``κ ∇⋅(ρ_{weight} ∇q)`` (or ``κ ∇²q`` when `ρ_weight === nothing`): weak-form
volume term plus the interior-penalty face flux ``τ [\\![q]\\!]`` (see
[`LDGLaplacianFlux`](@ref) and [`ldg_penalty_parameter`](@ref)).
"""
function ldg_laplacian_tendency(q, ρ_weight, κ, τ)
    wdiv = WeakDivergence()
    grad = Gradient()
    lgeom = Fields.local_geometry_field(axes(q))
    residual = similar(q)
    if ρ_weight === nothing
        @. residual = (-lgeom.WJ) * κ * (-wdiv(grad(q)))
    else
        @. residual = (-lgeom.WJ) * κ * (-wdiv(ρ_weight * grad(q)))
    end
    add_ldg_laplacian_flux_internal!(residual, q, τ)
    return residual ./ lgeom.WJ
end

"""
    ldg_penalty_parameter(κ, space)

Interior-penalty scaling ``τ = κ (2N_q − 1)^2 / h`` using the horizontal
spectral-element length scale (works for extruded hybrid spaces).
"""
function ldg_penalty_parameter(κ, space)
    hspace =
        space isa Spaces.ExtrudedFiniteDifferenceSpace ?
        Spaces.horizontal_space(space) : space
    h = Spaces.node_horizontal_length_scale(hspace)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(hspace))
    return κ * (2 * Nq - 1)^2 / h
end

"""
    LDGLaplacianFlux(τ)

Scalar interior-penalty flux for the LDG/SIPG Laplacian. Called through
[`add_numerical_flux_internal!`](@ref) on a WJ-weighted residual of
``−∇·F`` with ``F = −κ∇q``. Returns ``τ[[q]]`` with ``[[q]] = q⁻ − q⁺``.
"""
struct LDGLaplacianFlux{T} <: AbstractNumericalFlux
    τ::T
end

function (fn::LDGLaplacianFlux)(normal, argvals⁻, argvals⁺)
    q⁻, q⁺ = argvals⁻[1], argvals⁺[1]
    return fn.τ * (q⁻ - q⁺)
end

"""
    add_ldg_laplacian_flux_internal!(dydt, q, τ)

Add LDG interior-penalty face coupling to a WJ-weighted Laplacian residual.
"""
add_ldg_laplacian_flux_internal!(dydt, q, τ) =
    add_numerical_flux_internal!(LDGLaplacianFlux(τ), dydt, q)
