abstract type AbstractSpectralElementSpace <: AbstractSpace end

Topologies.nlocalelems(space::AbstractSpectralElementSpace) =
    Topologies.nlocalelems(Spaces.topology(space))

local_geometry_data(space::AbstractSpectralElementSpace) = space.local_geometry

eachslabindex(space::AbstractSpectralElementSpace) =
    1:Topologies.nlocalelems(Spaces.topology(space))

function Base.show(io::IO, space::AbstractSpectralElementSpace)
    typname = Base.typename(typeof(space)).name
    println(io, "$(typname):")
    println(io, "  topology: ", space.topology)
    println(io, "  quadrature: ", space.quadrature_style)
end

topology(space::AbstractSpectralElementSpace) = space.topology
quadrature_style(space::AbstractSpectralElementSpace) = space.quadrature_style

"""
    SpectralElementSpace1D <: AbstractSpace
A one-dimensional space: within each element the space is represented as a polynomial.
"""
struct SpectralElementSpace1D{T, Q, G, D, M} <: AbstractSpectralElementSpace
    topology::T
    quadrature_style::Q
    local_geometry::G
    dss_weights::D
    inverse_mass_matrix::M
end

function SpectralElementSpace1D(topology, quadrature_style)
    # TODO: we need to massage the coordinate points because the grid is assumed 2D
    CoordType = Topologies.coordinate_type(topology)
    CoordType1D = Geometry.coordinate_type(CoordType, 1)
    AIdx = Geometry.coordinate_axis(CoordType1D)
    FT = eltype(CoordType1D)
    nelements = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)

    Mxξ = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.CartesianAxis{AIdx}, Geometry.CovariantAxis{AIdx}},
        SMatrix{1, 1, FT, 1},
    }
    Mξx = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.ContravariantAxis{AIdx}, Geometry.CartesianAxis{AIdx}},
        SMatrix{1, 1, FT, 1},
    }
    LG = Geometry.LocalGeometry{CoordType1D, FT, Mxξ, Mξx}
    local_geometry = DataLayouts.IFH{LG, Nq}(Array{FT}, nelements)
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

    for elem in 1:nelements
        local_geometry_slab = slab(local_geometry, elem)
        for i in 1:Nq
            ξ = quad_points[i]
            # TODO: we need to massage the coordinate points because the grid is assumed 2D
            vcoords = Topologies.vertex_coordinates(topology, elem)
            vcoords1D = (
                Geometry.coordinate(vcoords[1], 1),
                Geometry.coordinate(vcoords[2], 1),
            )
            x = Geometry.linear_interpolate(vcoords1D, ξ)
            ∂x∂ξ =
                (
                    Geometry.component(vcoords1D[2], 1) -
                    Geometry.component(vcoords1D[1], 1)
                ) / 2
            J = abs(∂x∂ξ)
            ∂ξ∂x = inv(∂x∂ξ)
            WJ = J * quad_weights[i]
            local_geometry_slab[i] = Geometry.LocalGeometry(
                x,
                J,
                WJ,
                Geometry.AxisTensor(
                    (
                        Geometry.CartesianAxis{AIdx}(),
                        Geometry.CovariantAxis{AIdx}(),
                    ),
                    ∂x∂ξ,
                ),
                Geometry.AxisTensor(
                    (
                        Geometry.ContravariantAxis{AIdx}(),
                        Geometry.CartesianAxis{AIdx}(),
                    ),
                    ∂ξ∂x,
                ),
            )
        end
    end
    dss_weights = copy(local_geometry.J)
    dss_weights .= one(FT)
    dss_1d!(dss_weights, dss_weights, topology, Nq)
    dss_weights = one(FT) ./ dss_weights

    inverse_mass_matrix = copy(local_geometry.WJ)
    dss_1d!(inverse_mass_matrix, inverse_mass_matrix, topology, Nq)
    inverse_mass_matrix = one(FT) ./ inverse_mass_matrix

    return SpectralElementSpace1D(
        topology,
        quadrature_style,
        local_geometry,
        dss_weights,
        inverse_mass_matrix,
    )
end

nlevels(space::SpectralElementSpace1D) = 1

"""
    SpectralElementSpace2D <: AbstractSpace

A two-dimensional space: within each element the space is represented as a polynomial.
"""
struct SpectralElementSpace2D{T, Q, G, D, IS, BS} <:
       AbstractSpectralElementSpace
    topology::T
    quadrature_style::Q
    local_geometry::G
    dss_weights::D
    internal_surface_geometry::IS
    boundary_surface_geometries::BS
end

"""
    SpectralElementSpace2D(topology, quadrature_style)

Construct a `SpectralElementSpace2D` instance given a `topology` and `quadrature`.
"""
function SpectralElementSpace2D(topology, quadrature_style)
    CoordType2D = Topologies.coordinate_type(topology)
    AIdx = Geometry.coordinate_axis(CoordType2D)
    FT = eltype(CoordType2D)
    nelements = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)

    Mxξ = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.CartesianAxis{AIdx}, Geometry.CovariantAxis{AIdx}},
        SMatrix{2, 2, FT, 4},
    }
    Mξx = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.ContravariantAxis{AIdx}, Geometry.CartesianAxis{AIdx}},
        SMatrix{2, 2, FT, 4},
    }
    LG = Geometry.LocalGeometry{CoordType2D, FT, Mxξ, Mξx}

    local_geometry = DataLayouts.IJFH{LG, Nq}(Array{FT}, nelements)
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

    for elem in 1:nelements
        local_geometry_slab = slab(local_geometry, elem)
        for i in 1:Nq, j in 1:Nq
            # this hard-codes a bunch of assumptions, and will unnecesarily duplicate data
            # e.g. where all metric terms are uniform over the space
            # alternatively: move local_geometry to a different object entirely, to support overintegration
            # (where the integration is of different order)
            ξ = SVector(quad_points[i], quad_points[j])
            x = Geometry.bilinear_interpolate(
                CoordType2D.(Topologies.vertex_coordinates(topology, elem),),
                ξ[1],
                ξ[2],
            )
            ∂x∂ξ = ForwardDiff.jacobian(ξ) do ξ
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
            J = det(∂x∂ξ)
            ∂ξ∂x = inv(∂x∂ξ)
            WJ = J * quad_weights[i] * quad_weights[j]

            local_geometry_slab[i, j] = Geometry.LocalGeometry(
                x,
                J,
                WJ,
                Geometry.AxisTensor(
                    (
                        Geometry.CartesianAxis{AIdx}(),
                        Geometry.CovariantAxis{AIdx}(),
                    ),
                    ∂x∂ξ,
                ),
                Geometry.AxisTensor(
                    (
                        Geometry.ContravariantAxis{AIdx}(),
                        Geometry.CartesianAxis{AIdx}(),
                    ),
                    ∂ξ∂x,
                ),
            )
        end
    end

    # dss_weights = J ./ dss(J)
    dss_weights = copy(local_geometry.J)
    dss_2d!(dss_weights, local_geometry.J, topology, Nq)
    dss_weights .= local_geometry.J ./ dss_weights

    # TODO: this assumes XYDomain, need to dispatch SG Vector type on domain axis type
    SG = Geometry.SurfaceGeometry{FT, Geometry.Cartesian12Vector{FT}}
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
        map(Topologies.boundaries(topology)) do boundarytag
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
        local_geometry,
        dss_weights,
        internal_surface_geometry,
        boundary_surface_geometries,
    )
end

nlevels(space::SpectralElementSpace2D) = 1

"""
    SpectralElementSpace2D_sphere(topology, quadrature_style)

Construct a `SpectralElementSpace2D` instance given a `topology` and `quadrature`.
"""
function SpectralElementSpace2D(
    topology::Topologies.Grid2DTopology{
        <:Meshes.Mesh2D{<:Domains.SphereDomain},
    },
    quadrature_style,
)
    FT = eltype(topology.mesh)
    CT = Geometry.LatLongPoint{FT} # Domains.coordinate_type(topology)
    CoordType3D = Topologies.coordinate_type(topology)
    AIdx = Geometry.coordinate_axis(CoordType3D)
    radius = topology.mesh.domain.radius
    nelements = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    # types of the partial derivative tensors
    # ∂r∂ξ
    Mxξ = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.UVAxis, Geometry.Covariant12Axis},
        SMatrix{2, 2, FT, 4},
    }
    # ∂ξ∂r
    Mξx = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Contravariant12Axis, Geometry.UVAxis},
        SMatrix{2, 2, FT, 4},
    }
    LG = Geometry.LocalGeometry{CT, FT, Mxξ, Mξx}
    local_geometry = DataLayouts.IJFH{LG, Nq}(Array{FT}, nelements)
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)
    for elem in 1:nelements
        local_geometry_slab = slab(local_geometry, elem)
        for i in 1:Nq, j in 1:Nq
            # compute the coordinate and partial derivative matrices for each quadrature point
            # Guba (2014))
            ξ = SVector(quad_points[i], quad_points[j])
            x = Geometry.spherical_bilinear_interpolate(
                CoordType3D.(Topologies.vertex_coordinates(topology, elem),),
                ξ[1],
                ξ[2],
            )
            xl = Geometry.LatLongPoint(x)
            ∂x∂ξ = ForwardDiff.jacobian(ξ) do ξ
                Geometry.components(
                    Geometry.spherical_bilinear_interpolate(
                        CoordType3D.(
                            Topologies.vertex_coordinates(topology, elem),
                        ),
                        ξ[1],
                        ξ[2],
                    ),
                )
            end
            if abs(xl.lat) ≈ oftype(xl.lat, 90.0)
                # at the pole: choose u to line with x1, v to line with x2
                ∂u∂ξ = ∂x∂ξ[SOneTo(2), :]
            else
                ϕ = xl.lat
                λ = xl.long
                F = @SMatrix [
                    -sind(λ) cosd(λ) 0
                    0 0 1/cosd(ϕ)
                ]
                ∂u∂ξ = F * ∂x∂ξ
            end

            J = abs(det(∂u∂ξ))
            @assert J ≈ sqrt(det(∂x∂ξ' * ∂x∂ξ))
            ∂ξ∂u = inv(∂u∂ξ)
            WJ = J * quad_weights[i] * quad_weights[j]

            local_geometry_slab[i, j] = Geometry.LocalGeometry(
                xl,
                J,
                WJ,
                Geometry.AxisTensor(
                    (Geometry.UVAxis(), Geometry.Covariant12Axis()),
                    ∂u∂ξ,
                ),
                Geometry.AxisTensor(
                    (Geometry.Contravariant12Axis(), Geometry.UVAxis()),
                    ∂ξ∂u,
                ),
            )
        end
    end
    # dss_weights = J ./ dss(J)
    dss_weights = copy(local_geometry.J)
    dss_2d!(dss_weights, local_geometry.J, topology, Nq)
    dss_weights .= local_geometry.J ./ dss_weights

    SG = Geometry.SurfaceGeometry{FT, Geometry.UVVector{FT}}
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
        map(Topologies.boundaries(topology)) do boundarytag
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
        local_geometry,
        dss_weights,
        internal_surface_geometry,
        boundary_surface_geometries,
    )
    return nothing
end

const CubedSphereSpectralElementSpace2D = SpectralElementSpace2D{
    T,
} where {
    T <:
    Topologies.Grid2DTopology{
        M,
    },
} where {M <: Meshes.Mesh2D{<:Domains.SphereDomain}}

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
