abstract type AbstractSpectralElementSpace <: AbstractSpace end

Topologies.nlocalelems(space::AbstractSpectralElementSpace) =
    Topologies.nlocalelems(Spaces.topology(space))

local_geometry_data(space::AbstractSpectralElementSpace) = space.local_geometry
ghost_geometry_data(space::AbstractSpectralElementSpace) = space.ghost_geometry

eachslabindex(space::AbstractSpectralElementSpace) =
    1:Topologies.nlocalelems(Spaces.topology(space))

function Base.show(io::IO, space::AbstractSpectralElementSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, nameof(typeof(space)), ":")
    if hasfield(typeof(space), :topology)
        # some reduced spaces (like slab space) do not have topology
        println(iio, " "^(indent + 2), space.topology)
    end
    print(iio, " "^(indent + 2), space.quadrature_style)
end

Device.device(space::AbstractSpectralElementSpace) =
    Device.device(topology(space))
Device.device_array_type(space::AbstractSpectralElementSpace) =
    Device.device_array_type(Device.device(space))
topology(space::AbstractSpectralElementSpace) = space.topology
quadrature_style(space::AbstractSpectralElementSpace) = space.quadrature_style

abstract type AbstractPerimeter end

"""
    Perimeter2D <: AbstractPerimeter

Iterate over the perimeter degrees of freedom of a 2D spectral element.
"""
struct Perimeter2D{Nq} <: AbstractPerimeter end

"""
    Perimeter2D(Nq)

Construct a perimeter iterator for a 2D spectral element of degree `(Nq-1)`.
"""
Perimeter2D(Nq) = Perimeter2D{Nq}()
Adapt.adapt_structure(to, x::Perimeter2D) = x

function Base.iterate(perimeter::Perimeter2D{Nq}, loc = 1) where {Nq}
    if loc < 5
        return (Topologies.vertex_node_index(loc, Nq), loc + 1)
    elseif loc ≤ nperimeter2d(Nq)
        f = cld(loc - 4, Nq - 2)
        n = mod(loc - 4, Nq - 2) == 0 ? (Nq - 2) : mod(loc - 4, Nq - 2)
        return (Topologies.face_node_index(f, Nq, 1 + n), loc + 1)
    else
        return nothing
    end
end

function Base.getindex(perimeter::Perimeter2D{Nq}, loc = 1) where {Nq}
    if loc < 1 || loc > nperimeter2d(Nq)
        return (-1, -1)
    elseif loc < 5
        return Topologies.vertex_node_index(loc, Nq)
    else
        f = cld(loc - 4, Nq - 2)
        n = mod(loc - 4, Nq - 2) == 0 ? (Nq - 2) : mod(loc - 4, Nq - 2)
        return Topologies.face_node_index(f, Nq, 1 + n)
    end
end

nperimeter2d(Nq) = 4 + (Nq - 2) * 4
nperimeter(::Perimeter2D{Nq}) where {Nq} = nperimeter2d(Nq)
Base.length(::Perimeter2D{Nq}) where {Nq} = nperimeter2d(Nq)

"""
    SpectralElementSpace1D <: AbstractSpace

A one-dimensional space: within each element the space is represented as a polynomial.
"""
struct SpectralElementSpace1D{
    T,
    Q,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
    D,
} <: AbstractSpectralElementSpace
    topology::T
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
    dss_weights::D
end

function SpectralElementSpace1D(
    topology::Topologies.IntervalTopology,
    quadrature_style,
)
    global_geometry = Geometry.CartesianGlobalGeometry()
    CoordType = Topologies.coordinate_type(topology)
    AIdx = Geometry.coordinate_axis(CoordType)
    FT = eltype(CoordType)
    nelements = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)

    LG = Geometry.LocalGeometry{AIdx, CoordType, FT, SMatrix{1, 1, FT, 1}}
    local_geometry = DataLayouts.IFH{LG, Nq}(Array{FT}, nelements)
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
            local_geometry_slab[i] = Geometry.LocalGeometry(
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
    dss_1d!(topology, dss_weights)
    dss_weights = one(FT) ./ dss_weights

    return SpectralElementSpace1D(
        topology,
        quadrature_style,
        global_geometry,
        local_geometry,
        dss_weights,
    )
end

nlevels(space::SpectralElementSpace1D) = 1

const IntervalSpectralElementSpace1D =
    SpectralElementSpace1D{<:Topologies.IntervalTopology{<:Meshes.IntervalMesh}}

"""
    SpectralElementSpace2D <: AbstractSpace

A two-dimensional space: within each element the space is represented as a polynomial.
"""
struct SpectralElementSpace2D{
    T,
    Q,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
    D,
    IS,
    BS,
} <: AbstractSpectralElementSpace
    topology::T
    quadrature_style::Q
    global_geometry::GG
    local_geometry::LG
    ghost_geometry::LG
    local_dss_weights::D
    ghost_dss_weights::D
    internal_surface_geometry::IS
    boundary_surface_geometries::BS
end

Adapt.adapt_structure(to, space::SpectralElementSpace2D) =
    SpectralElementSpace2D(
        nothing, # drop topology
        Adapt.adapt(to, space.quadrature_style),
        Adapt.adapt(to, space.global_geometry),
        Adapt.adapt(to, space.local_geometry),
        Adapt.adapt(to, space.ghost_geometry),
        Adapt.adapt(to, space.local_dss_weights),
        Adapt.adapt(to, space.ghost_dss_weights),
        Adapt.adapt(to, space.internal_surface_geometry),
        Adapt.adapt(to, space.boundary_surface_geometries),
    )



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
function SpectralElementSpace2D(
    topology,
    quadrature_style;
    enable_bubble = false,
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
    DA = Device.device_array_type(topology)
    domain = Topologies.domain(topology)
    if domain isa Domains.SphereDomain
        CoordType3D = Topologies.coordinate_type(topology)
        FT = Geometry.float_type(CoordType3D)
        CoordType2D = Geometry.LatLongPoint{FT} # Domains.coordinate_type(topology)
        global_geometry =
            Geometry.SphericalGlobalGeometry(topology.mesh.domain.radius)
    else
        CoordType2D = Topologies.coordinate_type(topology)
        FT = Geometry.float_type(CoordType2D)
        global_geometry = Geometry.CartesianGlobalGeometry()
    end
    AIdx = Geometry.coordinate_axis(CoordType2D)
    nlelems = Topologies.nlocalelems(topology)
    ngelems = Topologies.nghostelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    high_order_quadrature_style = Spaces.Quadratures.GLL{Nq * 2}()
    high_order_Nq = Quadratures.degrees_of_freedom(high_order_quadrature_style)

    LG = Geometry.LocalGeometry{AIdx, CoordType2D, FT, SMatrix{2, 2, FT, 4}}

    local_geometry = DataLayouts.IJFH{LG, Nq}(Array{FT}, nlelems)
    ghost_geometry = DataLayouts.IJFH{LG, Nq}(Array{FT}, ngelems)

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
            u, ∂u∂ξ =
                compute_local_geometry(global_geometry, topology, elem, ξ, AIdx)
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
            u, ∂u∂ξ =
                compute_local_geometry(global_geometry, topology, elem, ξ, AIdx)
            J = det(Geometry.components(∂u∂ξ))
            WJ = J * quad_weights[i] * quad_weights[j]
            elem_area += WJ
            if !enable_bubble
                local_geometry_slab[i, j] =
                    Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
            end
        end

        # If enabled, apply bubble correction
        if enable_bubble
            if abs(elem_area - high_order_elem_area) ≤ eps(FT)
                @warn "The numerical and geometric areas of the element are equal. The bubble correction will not be performed."
                for i in 1:Nq, j in 1:Nq
                    ξ = SVector(quad_points[i], quad_points[j])
                    u, ∂u∂ξ = compute_local_geometry(
                        global_geometry,
                        topology,
                        elem,
                        ξ,
                        AIdx,
                    )
                    J = det(Geometry.components(∂u∂ξ))
                    WJ = J * quad_weights[i] * quad_weights[j]
                    local_geometry_slab[i, j] =
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
                            AIdx,
                        )
                        J = det(Geometry.components(∂u∂ξ))
                        J += Δarea / Nq^2
                        WJ = J * quad_weights[i] * quad_weights[j]
                        local_geometry_slab[i, j] =
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
                            AIdx,
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
                            AIdx,
                        )
                        J = det(Geometry.components(∂u∂ξ))
                        # Modify J only for interior nodes
                        if i != 1 && j != 1 && i != Nq && j != Nq
                            J *= (1 + rel_interior_elem_area_Δ)
                        end
                        WJ = J * quad_weights[i] * quad_weights[j]
                        # Finally allocate local geometry
                        local_geometry_slab[i, j] =
                            Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
                    end
                end
            end
        end
    end

    # alternatively, we could do a ghost exchange here?
    if topology isa Topologies.Topology2D
        for (ridx, elem) in enumerate(Topologies.ghostelems(topology))
            ghost_geometry_slab = slab(ghost_geometry, ridx)
            for i in 1:Nq, j in 1:Nq
                ξ = SVector(quad_points[i], quad_points[j])
                u, ∂u∂ξ = compute_local_geometry(
                    global_geometry,
                    topology,
                    elem,
                    ξ,
                    AIdx,
                )
                J = det(Geometry.components(∂u∂ξ))
                WJ = J * quad_weights[i] * quad_weights[j]

                ghost_geometry_slab[i, j] =
                    Geometry.LocalGeometry(u, J, WJ, ∂u∂ξ)
            end
        end
        if !isnothing(ghost_geometry) && DA ≠ Array
            ghost_geometry = DataLayouts.rebuild(ghost_geometry, DA)
        end
    end
    # dss_weights = J ./ dss(J)
    J = DataLayouts.rebuild(local_geometry.J, DA)
    dss_local_weights = copy(J)
    if quadrature_style isa Quadratures.GLL
        dss!(dss_local_weights, topology, quadrature_style)
    end
    dss_local_weights .= J ./ dss_local_weights
    dss_ghost_weights = copy(J) # not currently used

    SG = Geometry.SurfaceGeometry{
        FT,
        Geometry.AxisVector{FT, Geometry.LocalAxis{AIdx}, SVector{2, FT}},
    }
    interior_faces = Array(Topologies.interior_faces(topology))

    if quadrature_style isa Quadratures.GLL
        internal_surface_geometry =
            DataLayouts.IFH{SG, Nq}(Array{FT}, length(interior_faces))
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

                internal_surface_geometry_slab[q] = sgeom⁻
            end
        end
        internal_surface_geometry =
            DataLayouts.rebuild(internal_surface_geometry, DA)

        boundary_surface_geometries =
            map(Topologies.boundary_tags(topology)) do boundarytag
                boundary_faces =
                    Topologies.boundary_faces(topology, boundarytag)
                boundary_surface_geometry =
                    DataLayouts.IFH{SG, Nq}(Array{FT}, length(boundary_faces))
                for (iface, (elem, face)) in enumerate(boundary_faces)
                    boundary_surface_geometry_slab =
                        slab(boundary_surface_geometry, iface)
                    local_geometry_slab = slab(local_geometry, elem)
                    for q in 1:Nq
                        boundary_surface_geometry_slab[q] =
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
    return SpectralElementSpace2D(
        topology,
        quadrature_style,
        global_geometry,
        DataLayouts.rebuild(local_geometry, DA),
        ghost_geometry,
        dss_local_weights,
        dss_ghost_weights,
        internal_surface_geometry,
        boundary_surface_geometries,
    )
end

nlevels(space::SpectralElementSpace2D) = 1
perimeter(space::SpectralElementSpace2D) =
    Perimeter2D(Quadratures.degrees_of_freedom(space.quadrature_style))

const RectilinearSpectralElementSpace2D = SpectralElementSpace2D{
    <:Topologies.Topology2D{
        <:ClimaComms.AbstractCommsContext,
        <:Meshes.RectilinearMesh,
    },
}

const CubedSphereSpectralElementSpace2D = SpectralElementSpace2D{
    <:Topologies.Topology2D{
        <:ClimaComms.AbstractCommsContext,
        <:Meshes.AbstractCubedSphere,
    },
}

function compute_local_geometry(
    global_geometry::Geometry.SphericalGlobalGeometry,
    topology,
    elem,
    ξ,
    AIdx,
)
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
    AIdx,
)
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

    local_geometry = local_geometry_slab[i, j]
    @unpack J, ∂ξ∂x = local_geometry

    # surface mass matrix
    n = if face == 4
        -J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 2
        J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 1
        -J * ∂ξ∂x[2, :] * quad_weights[i]
    elseif face == 3
        J * ∂ξ∂x[2, :] * quad_weights[i]
    end
    sWJ = norm(n)
    n = n / sWJ
    return Geometry.SurfaceGeometry(sWJ, n)
end

function variational_solve!(data, space::AbstractSpace)
    data .= RecursiveApply.rdiv.(data, space.local_geometry.WJ)
end

"""
    SpectralElementSpaceSlab <: AbstractSpace

A view into a `SpectralElementSpace2D` for a single slab.
"""
struct SpectralElementSpaceSlab{Q, G} <: AbstractSpectralElementSpace
    quadrature_style::Q
    local_geometry::G
end

const SpectralElementSpaceSlab1D =
    SpectralElementSpaceSlab{Q, DL} where {Q, DL <: DataLayouts.DataSlab1D}

const SpectralElementSpaceSlab2D =
    SpectralElementSpaceSlab{Q, DL} where {Q, DL <: DataLayouts.DataSlab2D}

nlevels(space::SpectralElementSpaceSlab1D) = 1
nlevels(space::SpectralElementSpaceSlab2D) = 1

Base.@propagate_inbounds function slab(
    space::AbstractSpectralElementSpace,
    v,
    h,
)
    SpectralElementSpaceSlab(
        space.quadrature_style,
        slab(space.local_geometry, v, h),
    )
end
Base.@propagate_inbounds slab(space::AbstractSpectralElementSpace, h) =
    @inbounds slab(space, 1, h)

Base.@propagate_inbounds function column(space::SpectralElementSpace1D, i, h)
    local_geometry = column(local_geometry_data(space), i, h)
    PointSpace(local_geometry)
end
Base.@propagate_inbounds column(space::SpectralElementSpace1D, i, j, h) =
    column(space, i, h)

Base.@propagate_inbounds function column(space::SpectralElementSpace2D, i, j, h)
    local_geometry = column(local_geometry_data(space), i, j, h)
    PointSpace(local_geometry)
end

# XXX: this cannot take `space` as it must be constructed beforehand so
# that the `space` constructor can do DSS (to compute DSS weights)
function setup_comms(
    Context::Type{<:ClimaComms.AbstractCommsContext},
    topology::Topologies.AbstractDistributedTopology,
    quad_style::Spaces.Quadratures.QuadratureStyle,
    Nv,
    Nf = 2,
)
    Ni = Quadratures.degrees_of_freedom(quad_style)
    Nj = Ni
    AT = Array # XXX: get this from `space`/`topology`?
    FT = Geometry.float_type(Topologies.coordinate_type(topology))

    # Determine send and receive buffer dimensions for each neighbor PID
    # and add the neighbors in the same order as they are stored in
    # `neighbor_pids`!
    nbrs = ClimaComms.Neighbor[]
    for (nidx, npid) in enumerate(Topologies.neighbors(topology))
        nse = Topologies.nsendelems(topology, nidx)
        nge = Topologies.nghostelems(topology, nidx)
        send_dims = (Nv, Ni, Nj, Nf, nse)
        recv_dims = (Nv, Ni, Nj, Nf, nge)
        push!(
            nbrs,
            ClimaComms.Neighbor(Context, npid, AT, FT, send_dims, recv_dims),
        )
    end
    return Context(nbrs)
end

function all_nodes(space::SpectralElementSpace2D)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)
    nelem = Topologies.nlocalelems(space.topology)
    Iterators.product(Iterators.product(1:Nq, 1:Nq), 1:nelem)
end

"""
    unique_nodes(space::SpectralElementField2D)

An iterator over the unique nodes of `space`. Each node is represented by the
first `((i,j), e)` triple.

This function is experimental, and may change in future.
"""
unique_nodes(space::SpectralElementSpace2D) =
    unique_nodes(space, space.quadrature_style)

unique_nodes(space::SpectralElementSpace2D, quad::Quadratures.QuadratureStyle) =
    UniqueNodeIterator(space)
unique_nodes(space::SpectralElementSpace2D, ::Quadratures.GL) = all_nodes(space)

struct UniqueNodeIterator{S}
    space::S
end

Base.eltype(iter::UniqueNodeIterator{<:SpectralElementSpace2D}) =
    Tuple{Tuple{Int, Int}, Int}

function Base.length(iter::UniqueNodeIterator{<:SpectralElementSpace2D})
    space = iter.space
    topology = space.topology
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    nelem = Topologies.nlocalelems(topology)
    nvert = length(Topologies.local_vertices(topology))
    nface_interior = length(Topologies.interior_faces(topology))
    if isempty(Topologies.boundary_tags(topology))
        nface_boundary = 0
    else
        nface_boundary = sum(Topologies.boundary_tags(topology)) do tag
            length(Topologies.boundary_faces(topology, tag))
        end
    end
    return nelem * (Nq - 2)^2 +
           nvert +
           nface_interior * (Nq - 2) +
           nface_boundary * (Nq - 2)
end
Base.iterate(::UniqueNodeIterator{<:SpectralElementSpace2D}) =
    ((1, 1), 1), ((1, 1), 1)
function Base.iterate(
    iter::UniqueNodeIterator{<:SpectralElementSpace2D},
    ((i, j), e),
)
    space = iter.space
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)
    while true
        # find next node
        i += 1
        if i > Nq
            i = 1
            j += 1
        end
        if j > Nq
            j = 1
            e += 1
        end
        if e > Topologies.nlocalelems(space) # we're done
            return nothing
        end
        # check if this node has been seen
        # this assumes we don't have any shared vertices that are connected in a diagonal order,
        # e.g.
        #  1 | 3
        #  --+--
        #  4 | 2
        # we could check this by walking along the vertices as we go
        # this also doesn't deal with the case where eo == e
        if j == 1
            # face 1
            eo, _, _ = Topologies.opposing_face(space.topology, e, 1)
            if 0 < eo < e
                continue
            end
        end
        if i == Nq
            # face 2
            eo, _, _ = Topologies.opposing_face(space.topology, e, 2)
            if 0 < eo < e
                continue
            end
        end
        if j == Nq
            # face 3
            eo, _, _ = Topologies.opposing_face(space.topology, e, 3)
            if 0 < eo < e
                continue
            end
        end
        if i == 1
            # face 4
            eo, _, _ = Topologies.opposing_face(space.topology, e, 4)
            if 0 < eo < e
                continue
            end
        end
        return ((i, j), e), ((i, j), e)
    end
end
