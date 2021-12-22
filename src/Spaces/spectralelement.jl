abstract type AbstractSpectralElementSpace <: AbstractSpace end

Topologies.nlocalelems(space::AbstractSpectralElementSpace) =
    Topologies.nlocalelems(Spaces.topology(space))

local_geometry_data(space::AbstractSpectralElementSpace) = space.local_geometry

eachslabindex(space::AbstractSpectralElementSpace) =
    1:Topologies.nlocalelems(Spaces.topology(space))

function Base.show(io::IO, space::AbstractSpectralElementSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, nameof(typeof(space)), ":")
    println(iio, " "^(indent + 2), space.topology)
    print(iio, " "^(indent + 2), space.quadrature_style)
end

topology(space::AbstractSpectralElementSpace) = space.topology
quadrature_style(space::AbstractSpectralElementSpace) = space.quadrature_style

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
    dss_1d!(dss_weights, dss_weights, local_geometry, topology, Nq)
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
    dss_weights::D
    internal_surface_geometry::IS
    boundary_surface_geometries::BS
end

"""
    SpectralElementSpace2D(topology, quadrature_style)

Construct a `SpectralElementSpace2D` instance given a `topology` and `quadrature`.
"""
function SpectralElementSpace2D(topology, quadrature_style)
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
    nelements = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)

    LG = Geometry.LocalGeometry{AIdx, CoordType2D, FT, SMatrix{2, 2, FT, 4}}

    local_geometry = DataLayouts.IJFH{LG, Nq}(Array{FT}, nelements)
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

    for elem in 1:nelements
        local_geometry_slab = slab(local_geometry, elem)
        for i in 1:Nq, j in 1:Nq
            if domain isa Domains.SphereDomain
                # compute the coordinate and partial derivative matrices for each quadrature point
                # Guba (2014))
                ξ = SVector(quad_points[i], quad_points[j])
                x = Geometry.spherical_bilinear_interpolate(
                    CoordType3D.(
                        Topologies.vertex_coordinates(topology, elem),
                    ),
                    ξ[1],
                    ξ[2],
                    global_geometry.radius,
                )
                u = Geometry.LatLongPoint(x, global_geometry)
                # [∂x1/∂ξ¹ ∂x1/∂ξ²
                #  ∂x2/∂ξ¹ ∂x2/∂ξ²
                #  ∂x3/∂ξ¹ ∂x3/∂ξ²]
                ∂x∂ξ = ForwardDiff.jacobian(ξ) do ξ
                    Geometry.components(
                        Geometry.spherical_bilinear_interpolate(
                            CoordType3D.(
                                Topologies.vertex_coordinates(topology, elem),
                            ),
                            ξ[1],
                            ξ[2],
                            global_geometry.radius,
                        ),
                    )
                end
                ϕ = u.lat
                λ = u.long
                # [∂u/∂x1 ∂u/∂x2 ∂u/∂x3
                #  ∂v/∂x1 ∂v/∂x2 ∂v/∂x3]
                # at the pole we orient u and v by taking the limit approaching
                # from the line λ == 0
                ∂u∂x = if ϕ == 90
                    # north pole => u axis is aligned with x2, v is aligned with -x1
                    @assert λ == 0
                    @SMatrix [
                        0 one(ϕ) 0
                        -one(ϕ) 0 0
                    ]
                elseif ϕ == -90
                    # south pole => u axis is aligned with x2, v is aligned with x1
                    @assert λ == 0
                    @SMatrix [
                        0 one(ϕ) 0
                        one(ϕ) 0 0
                    ]
                else
                    #=
                        # TODO: this might be more stable?
                        [
                            -sind(λ) cosd(λ) 0
                            -sind(ϕ)*cosd(λ) -sind(ϕ)*sind(λ) cosd(ϕ)
                        ]
                    =#
                    @SMatrix [
                        -sind(λ) cosd(λ) 0
                        0 0 1/cosd(ϕ)
                    ]
                end
                ∂u∂ξ = ∂u∂x * ∂x∂ξ
            else

                # this hard-codes a bunch of assumptions, and will unnecesarily duplicate data
                # e.g. where all metric terms are uniform over the space
                # alternatively: move local_geometry to a different object entirely, to support overintegration
                # (where the integration is of different order)
                ξ = SVector(quad_points[i], quad_points[j])
                u = Geometry.bilinear_interpolate(
                    CoordType2D.(
                        Topologies.vertex_coordinates(topology, elem),
                    ),
                    ξ[1],
                    ξ[2],
                )
                ∂u∂ξ = ForwardDiff.jacobian(ξ) do ξ
                    local x
                    x = Geometry.bilinear_interpolate(
                        CoordType2D.(
                            Topologies.vertex_coordinates(topology, elem),
                        ),
                        ξ[1],
                        ξ[2],
                    )
                    SVector(Geometry.component(x, 1), Geometry.component(x, 2))
                end
            end
            J = det(∂u∂ξ)
            WJ = J * quad_weights[i] * quad_weights[j]

            local_geometry_slab[i, j] = Geometry.LocalGeometry(
                u,
                J,
                WJ,
                Geometry.AxisTensor(
                    (
                        Geometry.LocalAxis{AIdx}(),
                        Geometry.CovariantAxis{AIdx}(),
                    ),
                    ∂u∂ξ,
                ),
            )
        end
    end

    # dss_weights = J ./ dss(J)
    dss_weights = copy(local_geometry.J)
    dss_2d!(dss_weights, local_geometry.J, local_geometry, topology, Nq)
    dss_weights .= local_geometry.J ./ dss_weights

    SG = Geometry.SurfaceGeometry{
        FT,
        Geometry.AxisVector{FT, Geometry.LocalAxis{AIdx}, SVector{2, FT}},
    }
    interior_faces = Topologies.interior_faces(topology)

    internal_surface_geometry =
        DataLayouts.IFH{SG, Nq}(Array{FT}, length(interior_faces))
    for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
        enumerate(interior_faces)
        internal_surface_geometry_slab = slab(internal_surface_geometry, iface)

        local_geometry_slab⁻ = slab(local_geometry, elem⁻)
        local_geometry_slab⁺ = slab(local_geometry, elem⁺)

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

    boundary_surface_geometries =
        map(Topologies.boundary_tags(topology)) do boundarytag
            boundary_faces = Topologies.boundary_faces(topology, boundarytag)
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
            boundary_surface_geometry
        end

    return SpectralElementSpace2D(
        topology,
        quadrature_style,
        global_geometry,
        local_geometry,
        dss_weights,
        internal_surface_geometry,
        boundary_surface_geometries,
    )
end

nlevels(space::SpectralElementSpace2D) = 1

const CubedSphereSpectralElementSpace2D = SpectralElementSpace2D{
    <:Topologies.Topology2D{<:Meshes.AbstractCubedSphere},
}

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

function slab(space::AbstractSpectralElementSpace, v, h)
    SpectralElementSpaceSlab(
        space.quadrature_style,
        slab(space.local_geometry, v, h),
    )
end
