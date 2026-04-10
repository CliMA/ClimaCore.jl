

abstract type AbstractSpectralElementGrid <: AbstractGrid end

"""
    SpectralElementGrid1D(mesh::Meshes.IntervalMesh, quadrature_style::Quadratures.QuadratureStyle)

A one-dimensional space: within each element the space is represented as a polynomial.
"""
mutable struct SpectralElementGrid1D{
    T,
    Q,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
    D,
} <: AbstractSpectralElementGrid
    topology::T
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
    dss_weights::D
end

Adapt.@adapt_structure SpectralElementGrid1D

local_geometry_type(
    ::Type{SpectralElementGrid1D{T, Q, GG, LG}},
) where {T, Q, GG, LG} = eltype(LG) # calls eltype from DataLayouts

# non-view grids are cached based on their input arguments
# this means that if data is saved in two different files, reloading will give fields which live on the same grid
function SpectralElementGrid1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle;
    horizontal_layout_type = DataLayouts.IFH,
)
    get!(
        Cache.OBJECT_CACHE,
        (SpectralElementGrid1D, topology, quadrature_style),
    ) do
        _SpectralElementGrid1D(
            topology,
            quadrature_style,
            horizontal_layout_type,
        )
    end
end

_SpectralElementGrid1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle,
    horizontal_layout_type = DataLayouts.IFH,
) = _SpectralElementGrid1D(
    topology,
    quadrature_style,
    Val(Topologies.nlocalelems(topology)),
    horizontal_layout_type,
)

function _SpectralElementGrid1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle,
    ::Val{Nh},
    ::Type{horizontal_layout_type},
) where {Nh, horizontal_layout_type}
    DA = ClimaComms.array_type(topology)
    global_geometry = Geometry.CartesianGlobalGeometry()
    CoordType = Topologies.coordinate_type(topology)
    AIdx = Geometry.coordinate_axis(CoordType)
    FT = eltype(CoordType)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)

    _∂x∂ξ_bases = (
        Geometry.Basis{Geometry.Orthonormal, AIdx}(),
        Geometry.Basis{Geometry.Covariant, AIdx}(),
    )
    LG = Geometry.LocalGeometry{
        AIdx, CoordType, FT,
        Geometry.Metric{Geometry.Tensor{2, FT, typeof(_∂x∂ξ_bases), SMatrix{1, 1, FT, 1}}},
    }
    local_geometry = horizontal_layout_type{LG, Nq}(Array{FT}, Nh)
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

    for elem in 1:Nh
        local_geometry_slab = slab(local_geometry, elem)
        for i in 1:Nq
            ξ = quad_points[i]
            # TODO: we need to massage the coordinate points because the grid is assumed 2D
            vcoords = Topologies.vertex_coordinates(topology, elem)
            x = Geometry.linear_interpolate(vcoords, ξ)
            ∂x∂ξ =
                (
                    Geometry.component(vcoords[2], 1) -
                    Geometry.component(vcoords[1], 1)
                ) / 2
            J = abs(∂x∂ξ)
            WJ = J * quad_weights[i]
            local_geometry_slab[slab_index(i)] = Geometry.LocalGeometry(
                x,
                J,
                WJ,
                Geometry.Tensor(SMatrix{1, 1}(∂x∂ξ), _∂x∂ξ_bases),
            )
        end
    end

    device_local_geometry = DataLayouts.rebuild(local_geometry, DA)
    return SpectralElementGrid1D(
        topology,
        quadrature_style,
        global_geometry,
        device_local_geometry,
        compute_dss_weights(device_local_geometry, topology, quadrature_style),
    )
end



"""
    SpectralElementSpace2D <: AbstractSpace

A two-dimensional space: within each element the space is represented as a polynomial.
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
} <: AbstractSpectralElementGrid
    topology::T
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
    dss_weights::D
    internal_surface_geometry::IS
    boundary_surface_geometries::BS
    mask::M
    enable_bubble::Bool
    autodiff_metric::Bool
end

Adapt.@adapt_structure SpectralElementGrid2D

local_geometry_type(
    ::Type{SpectralElementGrid2D{T, Q, GG, LG, D, IS, BS, M}},
) where {T, Q, GG, LG, D, IS, BS, M} = eltype(LG) # calls eltype from DataLayouts

"""
    SpectralElementSpace2D(
        topology,
        quadrature_style;
        enable_bubble,
        autodiff_metric,
        horizontal_layout_type = DataLayouts.IJFH
        enable_mask::Bool,
    )

Construct a `SpectralElementSpace2D` instance given a `topology` and `quadrature`. The
flag `enable_bubble` enables the `bubble correction` for more accurate element areas.
The flag `autodiff_metric` enables the use of automatic differentiation instead of the
SEM for computing metric terms.

# Input arguments:
- topology: Topology2D
- quadrature_style: QuadratureStyle
- enable_bubble: Bool
- autodiff_metric: Bool
- horizontal_layout_type: Type{<:AbstractData}
- enable_mask: Boolean used to skip operations where the space's mask is 0

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
    horizontal_layout_type = DataLayouts.IJFH,
    enable_bubble::Bool = false,
    autodiff_metric::Bool = true,
    enable_mask::Bool = false,
)
    get!(
        Cache.OBJECT_CACHE,
        (
            SpectralElementGrid2D,
            topology,
            quadrature_style,
            enable_bubble,
            autodiff_metric,
            horizontal_layout_type,
            enable_mask,
        ),
    ) do
        _SpectralElementGrid2D(
            topology,
            quadrature_style,
            horizontal_layout_type;
            enable_bubble,
            autodiff_metric,
            enable_mask,
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

_SpectralElementGrid2D(
    topology::Topologies.Topology2D,
    quadrature_style::Quadratures.QuadratureStyle,
    horizontal_layout_type = DataLayouts.IJFH;
    enable_bubble::Bool,
    autodiff_metric::Bool,
    enable_mask::Bool = false,
) = _SpectralElementGrid2D(
    topology,
    quadrature_style,
    Val(Topologies.nlocalelems(topology)),
    horizontal_layout_type;
    enable_bubble,
    autodiff_metric,
    enable_mask,
)

function _SpectralElementGrid2D(
    topology::Topologies.Topology2D,
    quadrature_style::Quadratures.QuadratureStyle,
    ::Val{Nh},
    ::Type{horizontal_layout_type};
    enable_bubble::Bool,
    autodiff_metric::Bool,
    enable_mask::Bool = false,
) where {Nh, horizontal_layout_type}
    @assert horizontal_layout_type <: Union{DataLayouts.IJHF, DataLayouts.IJFH}
    surface_layout_type = if horizontal_layout_type <: DataLayouts.IJFH
        DataLayouts.IFH
    elseif horizontal_layout_type <: DataLayouts.IJHF
        DataLayouts.IHF
    else
        error("Uncaught case")
    end
    # 1. compute localgeom for local elememts
    # 2. ghost exchange of localgeom
    # 3. do a round of dss on WJs
    # 4. compute dss weights (WJ ./ dss(WJ)) (local and ghost)

    # DSS on a field would consist of
    # 1. copy to send buffers
    # 2. start exchange
    # 3. dss of internal connections
    #  - option for weighting and transformation
    # 4. finish exchange
    # 5. dss of ghost connections

    ### How to DSS multiple fields?
    # 1. allocate buffers externally
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
    ngelems = Topologies.nghostelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    high_order_quadrature_style = Quadratures.GLL{Nq * 2}()
    high_order_Nq = Quadratures.degrees_of_freedom(high_order_quadrature_style)

    _∂x∂ξ_bases2D = (
        Geometry.Basis{Geometry.Orthonormal, AIdx}(),
        Geometry.Basis{Geometry.Covariant, AIdx}(),
    )
    LG = Geometry.LocalGeometry{
        AIdx,
        CoordType2D,
        FT,
        Geometry.Metric{
            Geometry.Tensor{2, FT, typeof(_∂x∂ξ_bases2D), SMatrix{2, 2, FT, 4}},
        },
    }

    local_geometry = horizontal_layout_type{LG, Nq}(Array{FT}, Nh)
    mask = if enable_mask
        DataLayouts.ColumnMask(FT, horizontal_layout_type, DA, Val(Nq), Val(Nh))
    else
        DataLayouts.NoMask()
    end

    _, quad_weights = Quadratures.quadrature_points(FT, quadrature_style)
    _, high_order_quad_weights =
        Quadratures.quadrature_points(FT, high_order_quadrature_style)
    for (lidx, elem) in enumerate(Topologies.localelems(topology))
        elem_area = zero(FT)
        high_order_elem_area = zero(FT)
        Δarea = zero(FT)
        interior_elem_area = zero(FT)
        rel_interior_elem_area_Δ = zero(FT)
        local_geometry_slab = slab(local_geometry, lidx)
        lg_args =
            (global_geometry, topology, quadrature_style, autodiff_metric, elem)
        high_order_lg_args = (
            global_geometry,
            topology,
            high_order_quadrature_style,
            autodiff_metric,
            elem,
        )
        # high-order quadrature loop for computing geometric element face area.
        for i in 1:high_order_Nq, j in 1:high_order_Nq
            u, ∂u∂ξ = local_geometry_at_nodal_point(high_order_lg_args..., i, j)
            J_high_order = det(parent(∂u∂ξ))
            WJ_high_order =
                J_high_order *
                high_order_quad_weights[i] *
                high_order_quad_weights[j]
            high_order_elem_area += WJ_high_order
        end
        # low-order quadrature loop for computing numerical element face area
        for i in 1:Nq, j in 1:Nq
            u, ∂u∂ξ = local_geometry_at_nodal_point(lg_args..., i, j)
            J = det(parent(∂u∂ξ))
            WJ = J * quad_weights[i] * quad_weights[j]
            elem_area += WJ
            if !enable_bubble
                local_geometry_slab[slab_index(i, j)] =
                    Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
            end
        end

        # If enabled, apply bubble correction
        if enable_bubble
            if abs(elem_area - high_order_elem_area) ≤ eps(FT)
                for i in 1:Nq, j in 1:Nq
                    u, ∂u∂ξ = local_geometry_at_nodal_point(lg_args..., i, j)
                    J = det(parent(∂u∂ξ))
                    WJ = J * quad_weights[i] * quad_weights[j]
                    local_geometry_slab[slab_index(i, j)] =
                        Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
                end
            else
                # The idea behind the so-called `bubble_correction` is that
                # the numerical area of the domain (e.g., the sphere) is given by the sum
                # of nodal integration weights times their corresponding Jacobians. However,
                # this discrete sum is not exactly equal to the exact geometric area
                # (4pi*radius^2 for the sphere). It is required that numerical area = geometric area.
                # The "epsilon bubble" approach modifies the inner weights in each
                # element so that geometric and numerical areas of each element match.

                # Compute difference between geometric area of an element and its approximate numerical area
                Δarea = high_order_elem_area - elem_area

                # Linear elements: Nq == 2 (SpectralElementSpace2D cannot have Nq < 2)
                # Use uniform bubble correction
                if Nq == 2
                    for i in 1:Nq, j in 1:Nq
                        u, ∂u∂ξ =
                            local_geometry_at_nodal_point(lg_args..., i, j)
                        J = det(parent(∂u∂ξ))
                        J += Δarea / Nq^2
                        WJ = J * quad_weights[i] * quad_weights[j]
                        local_geometry_slab[slab_index(i, j)] =
                            Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
                    end
                else # Higher-order elements: Use HOMME bubble correction for the interior nodes
                    for i in 2:(Nq - 1), j in 2:(Nq - 1)
                        u, ∂u∂ξ =
                            local_geometry_at_nodal_point(lg_args..., i, j)
                        J = det(parent(∂u∂ξ))
                        WJ = J * quad_weights[i] * quad_weights[j]
                        interior_elem_area += WJ
                    end
                    # Check that interior_elem_area is not too small
                    if abs(interior_elem_area) ≤ sqrt(eps(FT))
                        error(
                            "Bubble correction cannot be performed; sum of inner weights is too small.",
                        )
                    end
                    rel_interior_elem_area_Δ = Δarea / interior_elem_area

                    for i in 1:Nq, j in 1:Nq
                        u, ∂u∂ξ =
                            local_geometry_at_nodal_point(lg_args..., i, j)
                        J = det(parent(∂u∂ξ))
                        # Modify J only for interior nodes
                        if i != 1 && j != 1 && i != Nq && j != Nq
                            J *= (1 + rel_interior_elem_area_Δ)
                        end
                        WJ = J * quad_weights[i] * quad_weights[j]
                        # Finally allocate local geometry
                        local_geometry_slab[slab_index(i, j)] =
                            Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
                    end
                end
            end
        end
    end

    SG = Geometry.SurfaceGeometry{
        FT,
        Geometry.LocalVector{FT, AIdx, SVector{2, FT}},
    }
    interior_faces = Array(Topologies.interior_faces(topology))

    if quadrature_style isa Quadratures.GLL
        internal_surface_geometry =
            surface_layout_type{SG, Nq}(Array{FT}, length(interior_faces))
        for (iface, (lidx⁻, face⁻, lidx⁺, face⁺, reversed)) in
            enumerate(interior_faces)
            internal_surface_geometry_slab =
                slab(internal_surface_geometry, iface)

            local_geometry_slab⁻ = slab(local_geometry, lidx⁻)
            local_geometry_slab⁺ = slab(local_geometry, lidx⁺)

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

                internal_surface_geometry_slab[slab_index(q)] = sgeom⁻
            end
        end
        internal_surface_geometry =
            DataLayouts.rebuild(internal_surface_geometry, DA)

        boundary_surface_geometries =
            map(Topologies.boundary_tags(topology)) do boundarytag
                boundary_faces =
                    Topologies.boundary_faces(topology, boundarytag)
                boundary_surface_geometry = surface_layout_type{SG, Nq}(
                    Array{FT},
                    length(boundary_faces),
                )
                for (iface, (elem, face)) in enumerate(boundary_faces)
                    boundary_surface_geometry_slab =
                        slab(boundary_surface_geometry, iface)
                    local_geometry_slab = slab(local_geometry, elem)
                    for q in 1:Nq
                        boundary_surface_geometry_slab[slab_index(q)] =
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
    else
        internal_surface_geometry = nothing
        boundary_surface_geometries = nothing
    end

    device_local_geometry = DataLayouts.rebuild(local_geometry, DA)
    return SpectralElementGrid2D(
        topology,
        quadrature_style,
        global_geometry,
        device_local_geometry,
        compute_dss_weights(device_local_geometry, topology, quadrature_style),
        internal_surface_geometry,
        boundary_surface_geometries,
        mask,
        enable_bubble,
        autodiff_metric,
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
        (Geometry.UVWAxis(), Geometry.Basis{Geometry.Covariant, AIdx}()),
    )
    u = Geometry.LatLongPoint(x, global_geometry)
    G = Geometry.local_to_cartesian(global_geometry, u)
    ∂u∂ξ = Geometry.project(Geometry.Basis{Geometry.Orthonormal, AIdx}(), G' * ∂x∂ξ)
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
            Geometry.Basis{Geometry.Orthonormal, AIdx}(),
            Geometry.Basis{Geometry.Covariant, AIdx}(),
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
    @assert size(local_geometry_slab) == (Nq, Nq, 1, 1, 1)
    i, j = Topologies.face_node_index(face, Nq, q, reversed)

    local_geometry = local_geometry_slab[slab_index(i, j)]
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
    return Geometry.SurfaceGeometry(sWJ, n)
end

function compute_dss_weights(local_geometry, topology, quadrature_style)
    Quadratures.requires_dss(quadrature_style) || return nothing

    # Although the weights are defined as WJ / Σ collocated WJ, we can use J
    # instead of WJ if the weights are symmetric across element boundaries.
    dss_weights = copy(local_geometry.J)
    Topologies.dss!(dss_weights, topology)
    @. dss_weights = local_geometry.J / dss_weights
    return dss_weights
end

# accessors

topology(grid::AbstractSpectralElementGrid) = grid.topology

local_geometry_data(grid::AbstractSpectralElementGrid, ::Nothing) =
    grid.local_geometry
global_geometry(grid::AbstractSpectralElementGrid) = grid.global_geometry

quadrature_style(grid::AbstractSpectralElementGrid) = grid.quadrature_style
dss_weights(grid::AbstractSpectralElementGrid, ::Nothing) = grid.dss_weights

## GPU compatibility
struct DeviceSpectralElementGrid2D{Q, GG, LG, M} <: AbstractSpectralElementGrid
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
    mask::M
end

ClimaComms.context(grid::DeviceSpectralElementGrid2D) = DeviceSideContext()
ClimaComms.device(grid::DeviceSpectralElementGrid2D) = DeviceSideDevice()

## aliases
const RectilinearSpectralElementGrid2D =
    SpectralElementGrid2D{<:Topologies.RectilinearTopology2D}
const CubedSphereSpectralElementGrid2D =
    SpectralElementGrid2D{<:Topologies.CubedSphereTopology2D}
