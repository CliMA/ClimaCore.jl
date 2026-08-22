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
    n = Geometry.project(Geometry.UVAxis(), nvec / sWJ)
    return Geometry.SurfaceGeometry(sWJ, n)
end

# Device-dispatch seam (DSS-style): CPU methods live here; the
# `ClimaComms.CUDADevice` methods are provided by the ClimaCoreCUDAExt
# extension (ext/cuda/operators_dg.jl).
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
add_numerical_flux_internal!(fn::F, dydt, args...) where {F} =
    _add_numerical_flux_internal!(
        ClimaComms.device(axes(dydt)),
        fn,
        dydt,
        args...,
    )

_add_numerical_flux_internal!(device, fn::F, dydt, args...) where {F} = error(
    "add_numerical_flux_internal! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

function _add_numerical_flux_internal!(
    ::ClimaComms.AbstractCPUDevice,
    fn::F,
    dydt,
    args...,
) where {F}
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
function add_numerical_flux_internal_extruded_1d!(fn::F, dydt, args...) where {F}
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
function add_numerical_flux_internal_extruded_2d!(fn::F, dydt, args...) where {F}
    space = axes(dydt)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    Nv = Spaces.nlevels(space)
    topology = Spaces.topology(space)
    local_geometry = Spaces.local_geometry_data(space)
    FT = Spaces.undertype(space)
    (_, quad_weights) = Quadratures.quadrature_points(FT, quadrature_style)

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

function add_numerical_flux_boundary!(fn::F, dydt, args...) where {F}
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

# ---------------------------------------------------------------------------
# Symmetric face lifting for non-conservative (gradient / curl) terms
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
add_lifting_flux_internal!(fn::F, dydt, args...) where {F} =
    _add_lifting_flux_internal!(
        ClimaComms.device(axes(dydt)),
        fn,
        dydt,
        args...,
    )

_add_lifting_flux_internal!(device, fn::F, dydt, args...) where {F} = error(
    "add_lifting_flux_internal! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

function _add_lifting_flux_internal!(
    ::ClimaComms.AbstractCPUDevice,
    fn::F,
    dydt,
    args...,
) where {F}
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
function add_lifting_flux_internal_2d!(fn::F, dydt, args...) where {F}
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
function add_lifting_flux_internal_extruded_2d!(fn::F, dydt, args...) where {F}
    space = axes(dydt)
    quadrature_style = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    Nv = Spaces.nlevels(space)
    topology = Spaces.topology(space)
    local_geometry = Spaces.local_geometry_data(space)
    FT = Spaces.undertype(space)
    (_, quad_weights) = Quadratures.quadrature_points(FT, quadrature_style)

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

"""
    lifting_correction(fn, ::Type{T}, args...)

WJ-normalized DG face-lifting correction field of element type `T`: applies
[`add_lifting_flux_internal!`](@ref) with face function `fn` to a zero
residual on the space of `args[1]` and divides by `WJ`. The result is the
correction to the corresponding element-local strong-form operator.
"""
function lifting_correction(fn::F, ::Type{T}, args...) where {F, T}
    space = axes(args[1])
    lgeom = Fields.local_geometry_field(space)
    r = similar(args[1], T)
    fill!(parent(r), 0)
    add_lifting_flux_internal!(fn, r, args...)
    return r ./ lgeom.WJ
end

# ---------------------------------------------------------------------------
# Flux-differencing (split-form / FDDG) volume divergence
# ---------------------------------------------------------------------------

@inline _fd_add(a::NamedTuple, b::NamedTuple) = map(_fd_add, a, b)
@inline _fd_add(a, b) = a + b

@inline _fd_scale(c, x::NamedTuple) = map(v -> _fd_scale(c, v), x)
@inline _fd_scale(c, x) = c * x

# Metric-scaled contravariant basis vector J ∂ξʳᵒʷ/∂x, projected onto the
# local orthonormal horizontal frame (single-valued at shared nodes, including
# across cubed-sphere panel edges).
@inline _fd_metric_vector(local_geometry, row) = Geometry.project(
    Geometry.UVAxis(),
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
add_flux_differencing_divergence!(fn2pt::F, dydt, y) where {F} =
    _add_flux_differencing_divergence!(
        ClimaComms.device(axes(dydt)),
        fn2pt,
        dydt,
        y,
    )

_add_flux_differencing_divergence!(device, fn2pt::F, dydt, y) where {F} = error(
    "add_flux_differencing_divergence! is not implemented for $device; load CUDA.jl for CUDADevice support",
)

function _add_flux_differencing_divergence!(
    ::ClimaComms.AbstractCPUDevice,
    fn2pt::F,
    dydt,
    y,
) where {F}
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
    lg = lg_at(i, j)
    Ja1 = _fd_metric_vector(lg, 1)
    Ja2 = _fd_metric_vector(lg, 2)
    y_ij = y_at(i, j)

    c1 = -2 * w[i] * w[j] * D[i, 1]
    total = fn2pt(
        c1 * Ja1,
        c1 * _fd_metric_vector(lg_at(1, j), 1),
        y_ij,
        y_at(1, j),
    )
    c2 = -2 * w[i] * w[j] * D[j, 1]
    total = _fd_add(
        total,
        fn2pt(
            c2 * Ja2,
            c2 * _fd_metric_vector(lg_at(i, 1), 2),
            y_ij,
            y_at(i, 1),
        ),
    )
    for k in 2:Nq
        c1 = -2 * w[i] * w[j] * D[i, k]
        t1 = fn2pt(
            c1 * Ja1,
            c1 * _fd_metric_vector(lg_at(k, j), 1),
            y_ij,
            y_at(k, j),
        )
        c2 = -2 * w[i] * w[j] * D[j, k]
        t2 = fn2pt(
            c2 * Ja2,
            c2 * _fd_metric_vector(lg_at(i, k), 2),
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

function _fd_divergence_slab!(fn2pt::F, dydt_slab, y_slab, lg_slab, D, w, Nq) where {F}
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

"""
    dg_connectivity(space)

Memoized [`DGConnectivity`](@ref) for `space`, keyed on the underlying grid
and the space type (so center/face extruded spaces get separate buffers).
Stored in `Utilities.Cache.OBJECT_CACHE` alongside the grid objects, so
`Utilities.Cache.clean_cache!` releases it (the buffer holds device arrays).
"""
function dg_connectivity(space)
    key = (DGConnectivity, Spaces.grid(space), typeof(space))
    return get!(() -> build_dg_connectivity(space), Cache.OBJECT_CACHE, key)
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
