

abstract type AbstractSpectralElementGrid <: AbstractGrid end

"""
    Discretization

Supertype of the singleton types [`CG`](@ref) and [`DG`](@ref), which
distinguish the Galerkin discretization of a spectral-element grid. Select it
with the `discretization` keyword of [`SpectralElementGrid1D`](@ref) /
[`SpectralElementGrid2D`](@ref) (and the corresponding `Spaces` constructors);
read it back with [`discretization`](@ref). Omitting the keyword follows the
quadrature: `CG()` when its nodes are shared across element boundaries
(`Quadratures.requires_dss`, e.g. `Quadratures.GLL`) and `DG()` otherwise.
Passing `CG()` explicitly with a quadrature that cannot represent a continuous
space (e.g. `Quadratures.GL`) is an `ArgumentError`.
"""
abstract type Discretization end

"""
    CG()

The continuous-Galerkin [`Discretization`](@ref): functions are
single-valued at element boundaries, and element-local weak operators are
completed by [`Spaces.weighted_dss!`](@ref).
"""
struct CG <: Discretization end

"""
    DG()

The discontinuous-Galerkin [`Discretization`](@ref): functions are
element-local (multi-valued at element boundaries), `Spaces.weighted_dss!` is
a no-op, and element coupling enters through interface numerical fluxes (see
`Operators.add_numerical_flux_interior!`).
"""
struct DG <: Discretization end

"""
    SpectralElementGrid1D(
        topology::Topologies.IntervalTopology,
        quadrature_style::Quadratures.QuadratureStyle;
        VIJH,
        discretization = nothing,
    )

A one-dimensional grid: within each element the space is represented as a
polynomial. `discretization` selects continuous ([`CG`](@ref)) or
discontinuous ([`DG`](@ref)) Galerkin, and follows the quadrature when omitted;
see [`SpectralElementGrid2D`](@ref).
"""
mutable struct SpectralElementGrid1D{
    T,
    Q,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
    D,
    Disc,
} <: AbstractSpectralElementGrid
    topology::T
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
    dss_weights::D
    discretization::Disc
end

Adapt.@adapt_structure SpectralElementGrid1D

local_geometry_type(
    ::Type{SpectralElementGrid1D{<:Any, <:Any, <:Any, LG}},
) where {LG} = eltype(LG) # calls eltype from DataLayouts

# non-view grids are cached based on their input arguments
# this means that if data is saved in two different files, reloading will give fields which live on the same grid
function SpectralElementGrid1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle;
    VIJH::Type{<:DataLayouts.VIJHWithF} = DataLayouts.VIJFH,
    discretization::Union{Discretization, Nothing} = nothing,
)
    discretization = resolve_discretization(discretization, quadrature_style)
    get!(
        Cache.OBJECT_CACHE,
        (SpectralElementGrid1D, topology, quadrature_style, discretization),
    ) do
        _SpectralElementGrid1D(
            topology,
            quadrature_style,
            VIJH;
            discretization,
        )
    end
end

# A continuous space needs nodes shared between elements, which exist only for
# `requires_dss` quadratures (e.g. GLL). An omitted discretization follows the
# quadrature, so a single-node horizontal space (`GL{1}` under a column) is
# `DG()`; an explicit `CG()` that the quadrature cannot represent is rejected
# rather than silently downgraded.
function resolve_discretization(discretization, quadrature_style)
    requires_dss = Quadratures.requires_dss(quadrature_style)
    isnothing(discretization) && return requires_dss ? CG() : DG()
    requires_dss ||
        discretization isa DG ||
        throw(
            ArgumentError(
                "$(typeof(quadrature_style)) does not share nodes between \
                 elements, so it cannot represent a continuous space; pass \
                 discretization = Grids.DG() or omit the keyword",
            ),
        )
    return discretization
end

function _SpectralElementGrid1D(
    topology,
    quadrature_style,
    ::Type{VIJH};
    discretization,
) where {VIJH}
    DA = ClimaComms.array_type(topology)
    global_geometry = Geometry.CartesianGlobalGeometry()
    CoordType = Topologies.coordinate_type(topology)
    AIdx = Geometry.coordinate_axis(CoordType)
    FT = eltype(CoordType)
    Nh = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    ∂x∂ξ_bases = (
        Geometry.Components{Geometry.Orthonormal, AIdx}(),
        Geometry.Components{Geometry.Covariant, AIdx}(),
    )
    LG = Geometry.LocalGeometryType(CoordType, FT, AIdx)
    local_geometry = VIJH{LG, 1, Nq, 1, nothing}(Array{FT}, Nh)
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

    for h in 1:Nh, i in 1:Nq
        ξ = quad_points[i]
        # TODO: we need to massage the coordinate points because the grid is assumed 2D
        vcoords = Topologies.vertex_coordinates(topology, h)
        x = Geometry.linear_interpolate(vcoords, ξ)
        ∂x∂ξ =
            (
                Geometry.component(vcoords[2], 1) -
                Geometry.component(vcoords[1], 1)
            ) / 2
        J = abs(∂x∂ξ)
        WJ = J * quad_weights[i]
        local_geometry[1, i, 1, h] = Geometry.LocalGeometry(
            x,
            J,
            WJ,
            Geometry.Tensor(SMatrix{1, 1}(∂x∂ξ), ∂x∂ξ_bases),
        )
    end

    device_local_geometry = DataLayouts.rebuild(local_geometry, DA)
    return SpectralElementGrid1D(
        topology,
        quadrature_style,
        global_geometry,
        device_local_geometry,
        compute_dss_weights(device_local_geometry, topology, discretization),
        discretization,
    )
end



"""
    SpectralElementGrid2D <: AbstractSpectralElementGrid

A two-dimensional grid: within each element the space is represented as a polynomial.
"""
mutable struct SpectralElementGrid2D{
    T,
    Q,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
    D,
    IS,
    BS,
    M,
    Disc,
} <: AbstractSpectralElementGrid
    topology::T
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
    dss_weights::D
    interior_surface_geometry::IS
    boundary_surface_geometries::BS
    mask::M
    enable_bubble::Bool
    autodiff_metric::Bool
    discretization::Disc
end

Adapt.@adapt_structure SpectralElementGrid2D

local_geometry_type(
    ::Type{SpectralElementGrid2D{<:Any, <:Any, <:Any, LG}},
) where {LG} = eltype(LG) # calls eltype from DataLayouts

"""
    SpectralElementGrid2D(
        topology,
        quadrature_style;
        enable_bubble,
        autodiff_metric,
        VIJH,
        enable_mask::Bool,
        discretization = nothing,
    )

Construct a `SpectralElementGrid2D` instance given a `topology` and `quadrature`. The
flag `enable_bubble` enables the `bubble correction` for more accurate element areas.
The flag `autodiff_metric` enables the use of automatic differentiation instead of the
SEM for computing metric terms.

# Input arguments:

  - topology: Topology2D
  - quadrature_style: QuadratureStyle
  - enable_bubble: Bool
  - autodiff_metric: Bool
  - VIJH: subtype of DataLayouts.VIJHWithF with a specific F axis
  - enable_mask: Boolean used to skip operations where the space's mask is 0
  - discretization: continuous ([`CG`](@ref)) or discontinuous
    ([`DG`](@ref)) Galerkin, following the quadrature when omitted. On a `DG()` grid no continuity is maintained
    across element boundaries, so [`Spaces.weighted_dss!`](@ref) is a no-op on
    fields over this grid and inter-element coupling is instead supplied by DG
    numerical fluxes (see `Operators.add_numerical_flux_interior!`). No DSS
    weights are computed. `InputOutput` serializes the discretization; grids in
    files written before it existed read back as continuous.

The idea behind the so-called `bubble_correction` is that the numerical area
of the domain (e.g., the sphere) is given by the sum of nodal integration weights
times their corresponding Jacobians. However, this discrete sum is not exactly
equal to the exact geometric area  (4pi*radius^2 for the sphere). To make these equal,
the "epsilon bubble" approach modifies the inner weights in each element so that
geometric and numerical areas of each element match.

Let ``\\Delta A^e := A^e_{exact} - A^e_{approx}``, then, in
the case of linear elements, we correct ``W_{i,j} J^e_{i,j}`` by:

```math
\\widehat{W_{i,j} J^e}_{i,j} = W_{i,j} J^e_{i,j} + \\Delta A^e * W_{i,j} / Nq^2 .
```

and the case of non linear elements, by

```math
\\widehat{W_{i,j} J^e}_{i,j} = W_{i,j} J^e_{i,j} \\left( 1 + \\tilde{A}^e \\right) ,
```

where ``\\tilde{A}^e`` is the approximated area given by the sum of the interior nodal integration weights.

Note: This is accurate only for cubed-spheres of the [`Meshes.EquiangularCubedSphere`](@ref) and
[`Meshes.EquidistantCubedSphere`](@ref) type, not for [`Meshes.ConformalCubedSphere`](@ref).
"""
function SpectralElementGrid2D(
    topology::Topologies.Topology2D,
    quadrature_style::Quadratures.QuadratureStyle;
    VIJH::Type{<:DataLayouts.VIJHWithF} = DataLayouts.VIJFH,
    enable_bubble::Bool = false,
    autodiff_metric::Bool = true,
    enable_mask::Bool = false,
    discretization::Union{Discretization, Nothing} = nothing,
)
    discretization = resolve_discretization(discretization, quadrature_style)
    get!(
        Cache.OBJECT_CACHE,
        (
            SpectralElementGrid2D,
            topology,
            quadrature_style,
            enable_bubble,
            autodiff_metric,
            VIJH,
            enable_mask,
            discretization,
        ),
    ) do
        _SpectralElementGrid2D(
            topology,
            quadrature_style,
            VIJH;
            enable_bubble,
            autodiff_metric,
            enable_mask,
            discretization,
        )
    end
end

function get_CoordType2D(topology)
    domain = Topologies.domain(topology)
    return if domain isa Domains.SphereDomain
        FT = Domains.float_type(domain)
        Geometry.LatLongPoint{FT} # Domains.coordinate_type(topology)
    else
        Topologies.coordinate_type(topology)
    end
end

# The "epsilon bubble" correction: the numerical area of the domain (the sum
# of nodal integration weights times their Jacobians) is not exactly equal to
# the geometric area (e.g. 4π radius² for the sphere), but the two are
# required to match. The correction modifies the interior weights of each
# element so that its numerical area equals the geometric area, approximated
# by `high_order_elem_area` from a quadrature of twice the order. Linear
# elements (Nq == 2) have no interior nodes, so the deficit is spread
# uniformly over all nodes; higher-order elements use the HOMME bubble
# correction, scaling J at interior nodes only. `@noinline` keeps this branchy
# block a separately inferred and cached unit of the grid constructor.
@noinline function apply_bubble_correction!(
    local_geometry,
    topology,
    quadrature_style,
    global_geometry,
    autodiff_metric,
    lidx,
    elem,
    elem_area::FT,
    high_order_elem_area::FT,
    quad_weights,
) where {FT}
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    lg_args =
        (global_geometry, topology, quadrature_style, autodiff_metric, elem)
    if abs(elem_area - high_order_elem_area) ≤ eps(FT)
        for i in 1:Nq, j in 1:Nq
            u, ∂u∂ξ = local_geometry_at_nodal_point(lg_args..., i, j)
            J = det(parent(∂u∂ξ))
            WJ = J * quad_weights[i] * quad_weights[j]
            local_geometry[1, i, j, lidx] = Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
        end
    else
        Δarea = high_order_elem_area - elem_area
        if Nq == 2
            for i in 1:Nq, j in 1:Nq
                u, ∂u∂ξ = local_geometry_at_nodal_point(lg_args..., i, j)
                J = det(parent(∂u∂ξ)) + Δarea / Nq^2
                WJ = J * quad_weights[i] * quad_weights[j]
                local_geometry[1, i, j, lidx] =
                    Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
            end
        else
            interior_elem_area = zero(FT)
            for i in 2:(Nq - 1), j in 2:(Nq - 1)
                u, ∂u∂ξ = local_geometry_at_nodal_point(lg_args..., i, j)
                J = det(parent(∂u∂ξ))
                WJ = J * quad_weights[i] * quad_weights[j]
                interior_elem_area += WJ
            end
            if abs(interior_elem_area) ≤ sqrt(eps(FT))
                error(
                    "Bubble correction cannot be performed; sum of inner weights is too small.",
                )
            end
            rel_interior_elem_area_Δ = Δarea / interior_elem_area

            for i in 1:Nq, j in 1:Nq
                u, ∂u∂ξ = local_geometry_at_nodal_point(lg_args..., i, j)
                J = det(parent(∂u∂ξ))
                # Modify J only for interior nodes
                if i != 1 && j != 1 && i != Nq && j != Nq
                    J *= (1 + rel_interior_elem_area_Δ)
                end
                WJ = J * quad_weights[i] * quad_weights[j]
                local_geometry[1, i, j, lidx] =
                    Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
            end
        end
    end
end

@noinline function compute_nodal_local_geometries!(
    local_geometry,
    topology,
    quadrature_style,
    global_geometry,
    autodiff_metric,
    enable_bubble,
)
    domain = Topologies.domain(topology)
    FT = Domains.float_type(domain)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    _, quad_weights = Quadratures.quadrature_points(FT, quadrature_style)

    if enable_bubble
        high_order_quadrature_style = Quadratures.GLL{Nq * 2}()
        high_order_Nq =
            Quadratures.degrees_of_freedom(high_order_quadrature_style)
        _, high_order_quad_weights =
            Quadratures.quadrature_points(FT, high_order_quadrature_style)
        for (lidx, elem) in enumerate(Topologies.localelems(topology))
            elem_area = zero(FT)
            high_order_elem_area = zero(FT)
            lg_args = (
                global_geometry,
                topology,
                quadrature_style,
                autodiff_metric,
                elem,
            )
            high_order_lg_args = (
                global_geometry,
                topology,
                high_order_quadrature_style,
                autodiff_metric,
                elem,
            )
            # high-order quadrature loop for computing the geometric element
            # face area
            for i in 1:high_order_Nq, j in 1:high_order_Nq
                u, ∂u∂ξ =
                    local_geometry_at_nodal_point(high_order_lg_args..., i, j)
                J_high_order = det(parent(∂u∂ξ))
                WJ_high_order =
                    J_high_order *
                    high_order_quad_weights[i] *
                    high_order_quad_weights[j]
                high_order_elem_area += WJ_high_order
            end
            # low-order quadrature loop for computing the numerical element
            # face area
            for i in 1:Nq, j in 1:Nq
                u, ∂u∂ξ = local_geometry_at_nodal_point(lg_args..., i, j)
                J = det(parent(∂u∂ξ))
                WJ = J * quad_weights[i] * quad_weights[j]
                elem_area += WJ
            end
            apply_bubble_correction!(
                local_geometry,
                topology,
                quadrature_style,
                global_geometry,
                autodiff_metric,
                lidx,
                elem,
                elem_area,
                high_order_elem_area,
                quad_weights,
            )
        end
    else
        for (lidx, elem) in enumerate(Topologies.localelems(topology))
            lg_args = (
                global_geometry,
                topology,
                quadrature_style,
                autodiff_metric,
                elem,
            )
            for i in 1:Nq, j in 1:Nq
                u, ∂u∂ξ = local_geometry_at_nodal_point(lg_args..., i, j)
                J = det(parent(∂u∂ξ))
                WJ = J * quad_weights[i] * quad_weights[j]
                local_geometry[1, i, j, lidx] =
                    Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
            end
        end
    end
end

@noinline function compute_surface_geometries(
    ::Type{VIJH},
    ::Type{SG},
    ::Type{FT},
    DA,
    local_geometry,
    topology,
    quad_weights,
    Nq,
) where {VIJH, SG, FT}
    interior_faces = Array(Topologies.interior_faces(topology))
    interior_surface_geometry =
        VIJH{SG, 1, Nq, 1, nothing}(Array{FT}, length(interior_faces))
    for (iface, (lidx⁻, face⁻, lidx⁺, face⁺, reversed)) in
        enumerate(interior_faces)
        local_geometry_slab⁻ = slab(local_geometry, 1, lidx⁻)
        local_geometry_slab⁺ = slab(local_geometry, 1, lidx⁺)

        for q in 1:Nq
            sgeom⁻ = compute_surface_geometry(
                local_geometry_slab⁻,
                quad_weights,
                face⁻,
                q,
                false,
            )
            sgeom⁺ = compute_surface_geometry(
                local_geometry_slab⁺,
                quad_weights,
                face⁺,
                q,
                reversed,
            )

            @assert sgeom⁻.sWJ ≈ sgeom⁺.sWJ
            @assert sgeom⁻.normal ≈ -sgeom⁺.normal

            interior_surface_geometry[1, q, 1, iface] = sgeom⁻
        end
    end
    interior_surface_geometry =
        DataLayouts.rebuild(interior_surface_geometry, DA)

    boundary_surface_geometries =
        map(Topologies.boundary_tags(topology)) do boundarytag
            boundary_faces =
                Topologies.boundary_faces(topology, boundarytag)
            boundary_surface_geometry = VIJH{SG, 1, Nq, 1, nothing}(
                Array{FT},
                length(boundary_faces),
            )
            for (iface, (elem, face)) in enumerate(boundary_faces)
                local_geometry_slab = slab(local_geometry, 1, elem)
                for q in 1:Nq
                    boundary_surface_geometry[1, q, 1, iface] =
                        compute_surface_geometry(
                            local_geometry_slab,
                            quad_weights,
                            face,
                            q,
                            false,
                        )
                end
            end
            DataLayouts.rebuild(boundary_surface_geometry, DA)
        end
    return (interior_surface_geometry, boundary_surface_geometries)
end

function _SpectralElementGrid2D(
    topology,
    quadrature_style,
    ::Type{VIJH};
    enable_bubble,
    autodiff_metric,
    enable_mask,
    discretization,
) where {VIJH}
    DA = ClimaComms.array_type(topology)
    domain = Topologies.domain(topology)
    FT = Domains.float_type(domain)
    global_geometry = if domain isa Domains.SphereDomain
        Geometry.SphericalGlobalGeometry(topology.mesh.domain.radius)
    else
        Geometry.CartesianGlobalGeometry()
    end
    CoordType2D = get_CoordType2D(topology)
    AIdx = Geometry.coordinate_axis(CoordType2D)
    Nh = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    LG = Geometry.LocalGeometryType(CoordType2D, FT, AIdx)

    local_geometry = VIJH{LG, 1, Nq, Nq, nothing}(Array{FT}, Nh)
    compute_nodal_local_geometries!(
        local_geometry,
        topology,
        quadrature_style,
        global_geometry,
        autodiff_metric,
        enable_bubble,
    )

    SG = Geometry.SurfaceGeometry{
        FT,
        Geometry.LocalVector{FT, AIdx, SVector{2, FT}},
    }
    _, quad_weights = Quadratures.quadrature_points(FT, quadrature_style)
    if quadrature_style isa Quadratures.GLL
        (interior_surface_geometry, boundary_surface_geometries) =
            compute_surface_geometries(
                VIJH,
                SG,
                FT,
                DA,
                local_geometry,
                topology,
                quad_weights,
                Nq,
            )
    else
        interior_surface_geometry = nothing
        boundary_surface_geometries = nothing
    end

    device_local_geometry = DataLayouts.rebuild(local_geometry, DA)
    # Construct the mask from the device-side geometry, so that its data is
    # stored on the same device as the rest of the grid.
    mask =
        enable_mask ? DataLayouts.IJHMask(device_local_geometry) :
        DataLayouts.NoMask()
    return SpectralElementGrid2D(
        topology,
        quadrature_style,
        global_geometry,
        device_local_geometry,
        compute_dss_weights(device_local_geometry, topology, discretization),
        interior_surface_geometry,
        boundary_surface_geometries,
        mask,
        enable_bubble,
        autodiff_metric,
        discretization,
    )
end

get_mask(grid::SpectralElementGrid2D) = grid.mask

function ξ_at_nodal_point(FT, quadrature_style, i, j)
    quad_points = Quadratures.quadrature_points(FT, quadrature_style)[1]
    return SVector(quad_points[i], quad_points[j])
end

function ∂f∂ξ_at_nodal_point(f, FT, quadrature_style, autodiff_metric, i, j)
    if autodiff_metric
        ξ = ξ_at_nodal_point(FT, quadrature_style, i, j)
        return ForwardDiff.jacobian(f, ξ)
    end
    nodal_indices = SOneTo(Quadratures.degrees_of_freedom(quadrature_style))
    deriv_matrix = Quadratures.differentiation_matrix(FT, quadrature_style)
    ∂f∂ξ¹ = sum(nodal_indices) do i′
        deriv_matrix[i, i′] * f(ξ_at_nodal_point(FT, quadrature_style, i′, j))
    end
    ∂f∂ξ² = sum(nodal_indices) do j′
        deriv_matrix[j, j′] * f(ξ_at_nodal_point(FT, quadrature_style, i, j′))
    end
    return hcat(∂f∂ξ¹, ∂f∂ξ²)
end

function local_geometry_at_nodal_point(
    global_geometry::Geometry.SphericalGlobalGeometry,
    topology,
    quadrature_style,
    autodiff_metric,
    elem,
    i,
    j,
)
    FT = eltype(Topologies.coordinate_type(topology))
    AIdx = Geometry.coordinate_axis(get_CoordType2D(topology))
    ξ = ξ_at_nodal_point(FT, quadrature_style, i, j)
    x = Meshes.coordinates(topology.mesh, elem, ξ)
    ∂x∂ξ = Geometry.Tensor(
        ∂f∂ξ_at_nodal_point(FT, quadrature_style, autodiff_metric, i, j) do ξ
            Geometry.components(Meshes.coordinates(topology.mesh, elem, ξ))
        end,
        (Geometry.UVWAxis(), Geometry.Components{Geometry.Covariant, AIdx}()),
    )
    u = Geometry.LatLongPoint(x, global_geometry)
    G = Geometry.local_to_cartesian(global_geometry, u)
    ∂u∂ξ = Geometry.project(Geometry.Components{Geometry.Orthonormal, AIdx}(), G' * ∂x∂ξ)
    return u, ∂u∂ξ
end
function local_geometry_at_nodal_point(
    ::Geometry.AbstractGlobalGeometry,
    topology,
    quadrature_style,
    autodiff_metric,
    elem,
    i,
    j,
)
    FT = eltype(Topologies.coordinate_type(topology))
    AIdx = Geometry.coordinate_axis(get_CoordType2D(topology))
    ξ = ξ_at_nodal_point(FT, quadrature_style, i, j)
    u = Meshes.coordinates(topology.mesh, elem, ξ)
    ∂u∂ξ = Geometry.Tensor(
        ∂f∂ξ_at_nodal_point(FT, quadrature_style, autodiff_metric, i, j) do ξ
            Geometry.components(Meshes.coordinates(topology.mesh, elem, ξ))
        end,
        (
            Geometry.Components{Geometry.Orthonormal, AIdx}(),
            Geometry.Components{Geometry.Covariant, AIdx}(),
        ),
    )
    return u, ∂u∂ξ
end

function compute_surface_geometry(
    local_geometry_slab,
    quad_weights,
    face,
    q,
    reversed = false,
)
    Nq = length(quad_weights)
    @assert size(local_geometry_slab) == (1, Nq, Nq, 1)
    i, j = Topologies.face_node_index(face, Nq, q, reversed)

    local_geometry = local_geometry_slab[1, i, j, 1]
    (; J, ∂ξ∂x) = local_geometry

    # surface mass matrix
    n = if face == 4
        -J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 2
        J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 1
        -J * ∂ξ∂x[2, :] * quad_weights[i]
    elseif face == 3
        J * ∂ξ∂x[2, :] * quad_weights[i]
    else
        error("Uncaught case")
    end
    sWJ = norm(n)
    n = n / sWJ
    n = Geometry.project(_orth_axis(local_geometry), n)
    return Geometry.SurfaceGeometry(sWJ, n)
end

@inline _orth_axis(::Geometry.LocalGeometry{I}) where {I} =
    Geometry.Components{Geometry.Orthonormal, I}()

compute_dss_weights(local_geometry, topology, ::DG) = nothing
function compute_dss_weights(local_geometry, topology, ::CG)
    # Although the weights are defined as WJ / Σ collocated WJ, we can use J
    # instead of WJ if the weights are symmetric across element boundaries.
    dss_weights = copy(local_geometry.J)
    Topologies.dss!(dss_weights, topology)
    @. dss_weights = local_geometry.J / dss_weights
    return dss_weights
end

# accessors

"""
    Grids.discretization(grid)
    Spaces.discretization(space)

The [`Discretization`](@ref) of `grid` (or of `space`'s grid): [`CG`](@ref)`()`
or [`DG`](@ref)`()`, as given at grid construction. Grids with no horizontal
spectral elements are `CG()`, since every node belongs to one element.

There is no fallback for `AbstractGrid`: a new grid type needs its own method,
so that it cannot silently report a discretization it never chose.
"""
discretization(grid::SpectralElementGrid1D) = grid.discretization
discretization(grid::SpectralElementGrid2D) = grid.discretization

"""
    Grids.is_continuous(grid)
    Spaces.is_continuous(space)

Whether fields on `grid` (or on `space`'s grid) are members of the continuous
(CG) function space: `Grids.discretization(grid) isa CG`. Discontinuous (DG)
grids skip [`Spaces.weighted_dss!`](@ref) and couple elements through
numerical fluxes instead.
"""
is_continuous(grid::AbstractGrid) = discretization(grid) isa CG

topology(grid::AbstractSpectralElementGrid) = grid.topology

local_geometry_data(grid::AbstractSpectralElementGrid, ::Nothing) =
    grid.local_geometry
global_geometry(grid::AbstractSpectralElementGrid) = grid.global_geometry

quadrature_style(grid::AbstractSpectralElementGrid) = grid.quadrature_style
dss_weights(grid::AbstractSpectralElementGrid, ::Nothing) = grid.dss_weights

## GPU compatibility
struct DeviceSpectralElementGrid1D{Q, GG, LG} <: AbstractSpectralElementGrid
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
end
struct DeviceSpectralElementGrid2D{Q, GG, LG, M} <: AbstractSpectralElementGrid
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
    mask::M
end

ClimaComms.context(grid::DeviceSpectralElementGrid1D) = DeviceSideContext()
ClimaComms.device(grid::DeviceSpectralElementGrid1D) = DeviceSideDevice()

ClimaComms.context(grid::DeviceSpectralElementGrid2D) = DeviceSideContext()
ClimaComms.device(grid::DeviceSpectralElementGrid2D) = DeviceSideDevice()

## aliases
const RectilinearSpectralElementGrid2D =
    SpectralElementGrid2D{<:Topologies.RectilinearTopology2D}
const CubedSphereSpectralElementGrid2D =
    SpectralElementGrid2D{<:Topologies.CubedSphereTopology2D}
