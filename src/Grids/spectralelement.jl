

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

local_geometry_type(
    ::Type{SpectralElementGrid1D{T, Q, GG, LG}},
) where {T, Q, GG, LG} = eltype(LG) # calls eltype from DataLayouts

# non-view grids are cached based on their input arguments
# this means that if data is saved in two different files, reloading will give fields which live on the same grid
function SpectralElementGrid1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle,
)
    get!(
        Cache.OBJECT_CACHE,
        (SpectralElementGrid1D, topology, quadrature_style),
    ) do
        _SpectralElementGrid1D(topology, quadrature_style)
    end
end

function _SpectralElementGrid1D(
    topology::Topologies.IntervalTopology,
    quadrature_style::Quadratures.QuadratureStyle,
)
    global_geometry = Geometry.CartesianGlobalGeometry()
    CoordType = Topologies.coordinate_type(topology)
    AIdx = Geometry.coordinate_axis(CoordType)
    FT = eltype(CoordType)
    nelements = Topologies.nlocalelems(topology)
    Nh = nelements
    Nq = Quadratures.degrees_of_freedom(quadrature_style)

    LG = Geometry.LocalGeometry{AIdx, CoordType, FT, SMatrix{1, 1, FT, 1}}
    local_geometry = DataLayouts.IFH{LG, Nq, Nh}(Array{FT})
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

    for elem in 1:nelements
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
                Geometry.AxisTensor(
                    (
                        Geometry.LocalAxis{AIdx}(),
                        Geometry.CovariantAxis{AIdx}(),
                    ),
                    ∂x∂ξ,
                ),
            )
        end
    end
    dss_weights = copy(local_geometry.J)
    dss_weights .= one(FT)
    Topologies.dss_1d!(topology, dss_weights)
    dss_weights = one(FT) ./ dss_weights

    return SpectralElementGrid1D(
        topology,
        quadrature_style,
        global_geometry,
        local_geometry,
        dss_weights,
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
} <: AbstractSpectralElementGrid
    topology::T
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
    local_dss_weights::D
    internal_surface_geometry::IS
    boundary_surface_geometries::BS
end

local_geometry_type(
    ::Type{SpectralElementGrid2D{T, Q, GG, LG, D, IS, BS}},
) where {T, Q, GG, LG, D, IS, BS} = eltype(LG) # calls eltype from DataLayouts

"""
    SpectralElementSpace2D(topology, quadrature_style; enable_bubble)

Construct a `SpectralElementSpace2D` instance given a `topology` and `quadrature`. The
flag `enable_bubble` enables the `bubble correction` for more accurate element areas.

# Input arguments:
- topology: Topology2D
- quadrature_style: QuadratureStyle
- enable_bubble: Bool

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
    enable_bubble::Bool = false,
)
    get!(
        Cache.OBJECT_CACHE,
        (SpectralElementGrid2D, topology, quadrature_style, enable_bubble),
    ) do
        _SpectralElementGrid2D(topology, quadrature_style; enable_bubble)
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

function _SpectralElementGrid2D(
    topology::Topologies.Topology2D,
    quadrature_style::Quadratures.QuadratureStyle;
    enable_bubble::Bool,
)

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
    nlelems = Topologies.nlocalelems(topology)
    Nh = nlelems
    ngelems = Topologies.nghostelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    high_order_quadrature_style = Quadratures.GLL{Nq * 2}()
    high_order_Nq = Quadratures.degrees_of_freedom(high_order_quadrature_style)

    LG = Geometry.LocalGeometry{AIdx, CoordType2D, FT, SMatrix{2, 2, FT, 4}}

    local_geometry = DataLayouts.IJFH{LG, Nq, Nh}(Array{FT})

    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)
    high_order_quad_points, high_order_quad_weights =
        Quadratures.quadrature_points(FT, high_order_quadrature_style)
    for (lidx, elem) in enumerate(Topologies.localelems(topology))
        elem_area = zero(FT)
        high_order_elem_area = zero(FT)
        Δarea = zero(FT)
        interior_elem_area = zero(FT)
        rel_interior_elem_area_Δ = zero(FT)
        local_geometry_slab = slab(local_geometry, lidx)
        # high-order quadrature loop for computing geometric element face area.
        for i in 1:high_order_Nq, j in 1:high_order_Nq
            ξ = SVector(high_order_quad_points[i], high_order_quad_points[j])
            u, ∂u∂ξ = compute_local_geometry(
                global_geometry,
                topology,
                elem,
                ξ,
                Val(AIdx),
            )
            J_high_order = det(Geometry.components(∂u∂ξ))
            WJ_high_order =
                J_high_order *
                high_order_quad_weights[i] *
                high_order_quad_weights[j]
            high_order_elem_area += WJ_high_order
        end
        # low-order quadrature loop for computing numerical element face area
        for i in 1:Nq, j in 1:Nq
            ξ = SVector(quad_points[i], quad_points[j])
            u, ∂u∂ξ = compute_local_geometry(
                global_geometry,
                topology,
                elem,
                ξ,
                Val(AIdx),
            )
            J = det(Geometry.components(∂u∂ξ))
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
                    ξ = SVector(quad_points[i], quad_points[j])
                    u, ∂u∂ξ = compute_local_geometry(
                        global_geometry,
                        topology,
                        elem,
                        ξ,
                        Val(AIdx),
                    )
                    J = det(Geometry.components(∂u∂ξ))
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
                        ξ = SVector(quad_points[i], quad_points[j])
                        u, ∂u∂ξ = compute_local_geometry(
                            global_geometry,
                            topology,
                            elem,
                            ξ,
                            Val(AIdx),
                        )
                        J = det(Geometry.components(∂u∂ξ))
                        J += Δarea / Nq^2
                        WJ = J * quad_weights[i] * quad_weights[j]
                        local_geometry_slab[slab_index(i, j)] =
                            Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
                    end
                else # Higher-order elements: Use HOMME bubble correction for the interior nodes
                    for i in 2:(Nq - 1), j in 2:(Nq - 1)
                        ξ = SVector(quad_points[i], quad_points[j])
                        u, ∂u∂ξ = compute_local_geometry(
                            global_geometry,
                            topology,
                            elem,
                            ξ,
                            Val(AIdx),
                        )
                        J = det(Geometry.components(∂u∂ξ))
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
                        ξ = SVector(quad_points[i], quad_points[j])
                        u, ∂u∂ξ = compute_local_geometry(
                            global_geometry,
                            topology,
                            elem,
                            ξ,
                            Val(AIdx),
                        )
                        J = det(Geometry.components(∂u∂ξ))
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

    # dss_weights = J ./ dss(J)
    J = DataLayouts.rebuild(local_geometry.J, DA)
    dss_local_weights = copy(J)
    if quadrature_style isa Quadratures.GLL
        Topologies.dss!(dss_local_weights, topology)
    end
    dss_local_weights .= J ./ dss_local_weights

    SG = Geometry.SurfaceGeometry{
        FT,
        Geometry.AxisVector{FT, Geometry.LocalAxis{AIdx}, SVector{2, FT}},
    }
    interior_faces = Array(Topologies.interior_faces(topology))

    if quadrature_style isa Quadratures.GLL
        internal_surface_geometry =
            DataLayouts.IFH{SG, Nq, length(interior_faces)}(Array{FT})
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
                boundary_surface_geometry =
                    DataLayouts.IFH{SG, Nq, length(boundary_faces)}(Array{FT})
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
    return SpectralElementGrid2D(
        topology,
        quadrature_style,
        global_geometry,
        DataLayouts.rebuild(local_geometry, DA),
        dss_local_weights,
        internal_surface_geometry,
        boundary_surface_geometries,
    )
end

function compute_local_geometry(
    global_geometry::Geometry.SphericalGlobalGeometry,
    topology,
    elem,
    ξ,
    ::Val{AIdx},
) where {AIdx}
    x = Meshes.coordinates(topology.mesh, elem, ξ)
    u = Geometry.LatLongPoint(x, global_geometry)
    ∂x∂ξ = Geometry.AxisTensor(
        (Geometry.Cartesian123Axis(), Geometry.CovariantAxis{AIdx}()),
        ForwardDiff.jacobian(ξ) do ξ
            Geometry.components(Meshes.coordinates(topology.mesh, elem, ξ))
        end,
    )
    G = Geometry.local_to_cartesian(global_geometry, u)
    ∂u∂ξ = Geometry.project(Geometry.LocalAxis{AIdx}(), G' * ∂x∂ξ)

    return u, ∂u∂ξ
end
function compute_local_geometry(
    global_geometry::Geometry.AbstractGlobalGeometry,
    topology,
    elem,
    ξ,
    ::Val{AIdx},
) where {AIdx}
    u = Meshes.coordinates(topology.mesh, elem, ξ)
    ∂u∂ξ = Geometry.AxisTensor(
        (Geometry.LocalAxis{AIdx}(), Geometry.CovariantAxis{AIdx}()),
        ForwardDiff.jacobian(ξ) do ξ
            Geometry.components(Meshes.coordinates(topology.mesh, elem, ξ))
        end,
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


# accessors

topology(grid::AbstractSpectralElementGrid) = grid.topology

local_geometry_data(grid::AbstractSpectralElementGrid, ::Nothing) =
    grid.local_geometry
global_geometry(grid::AbstractSpectralElementGrid) = grid.global_geometry

quadrature_style(grid::AbstractSpectralElementGrid) = grid.quadrature_style
local_dss_weights(grid::SpectralElementGrid1D) = grid.dss_weights
local_dss_weights(grid::SpectralElementGrid2D) = grid.local_dss_weights

## GPU compatibility
struct DeviceSpectralElementGrid2D{Q, GG, LG} <: AbstractSpectralElementGrid
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
end

ClimaComms.context(grid::DeviceSpectralElementGrid2D) = DeviceSideContext()
ClimaComms.device(grid::DeviceSpectralElementGrid2D) = DeviceSideDevice()

Adapt.adapt_structure(to, grid::SpectralElementGrid2D) =
    DeviceSpectralElementGrid2D(
        Adapt.adapt(to, grid.quadrature_style),
        Adapt.adapt(to, grid.global_geometry),
        Adapt.adapt(to, grid.local_geometry),
    )

## aliases
const RectilinearSpectralElementGrid2D =
    SpectralElementGrid2D{<:Topologies.RectilinearTopology2D}
const CubedSphereSpectralElementGrid2D =
    SpectralElementGrid2D{<:Topologies.CubedSphereTopology2D}
