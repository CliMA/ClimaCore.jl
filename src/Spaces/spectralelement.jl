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
    FT = eltype(topology.mesh)
    CT = Geometry.XPoint{FT} # Domains.coordinate_type(topology)
    nelements = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)

    Mxξ = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Cartesian1Axis, Geometry.Covariant1Axis},
        SMatrix{1, 1, FT, 1},
    }
    Mξx = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Contravariant1Axis, Geometry.Cartesian1Axis},
        SMatrix{1, 1, FT, 1},
    }
    LG = Geometry.LocalGeometry{CT, FT, Mxξ, Mξx}
    local_geometry = DataLayouts.IFH{LG, Nq}(Array{FT}, nelements)
    quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

    for elem in 1:nelements
        local_geometry_slab = slab(local_geometry, elem)
        for i in 1:Nq
            ξ = quad_points[i]
            # TODO: we need to reconsruct the points becuase the grid is assumed 2D
            vcoords = Topologies.vertex_coordinates(topology, elem)
            vcoords1D = (CT(vcoords[1].x1), CT(vcoords[2].x1))
            x = Geometry.interpolate(vcoords1D, ξ)
            ∂x∂ξ = (vcoords1D[2].x - vcoords1D[1].x) / 2
            J = abs(∂x∂ξ)
            ∂ξ∂x = inv(∂x∂ξ)
            WJ = J * quad_weights[i]
            local_geometry_slab[i] = Geometry.LocalGeometry(
                x,
                J,
                WJ,
                Geometry.AxisTensor(
                    (Geometry.Cartesian1Axis(), Geometry.Covariant1Axis()),
                    ∂x∂ξ,
                ),
                Geometry.AxisTensor(
                    (Geometry.Contravariant1Axis(), Geometry.Cartesian1Axis()),
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
    CT = Domains.coordinate_type(topology)
    FT = eltype(CT)
    nelements = Topologies.nlocalelems(topology)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)

    Mxξ = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Cartesian12Axis, Geometry.Covariant12Axis},
        SMatrix{2, 2, FT, 4},
    }
    Mξx = Geometry.Axis2Tensor{
        FT,
        Tuple{Geometry.Contravariant12Axis, Geometry.Cartesian12Axis},
        SMatrix{2, 2, FT, 4},
    }
    LG = Geometry.LocalGeometry{CT, FT, Mxξ, Mξx}

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
            x = Geometry.interpolate(
                Topologies.vertex_coordinates(topology, elem),
                ξ[1],
                ξ[2],
            )
            ∂x∂ξ = ForwardDiff.jacobian(ξ) do ξ
                local x
                x = Geometry.interpolate(
                    Topologies.vertex_coordinates(topology, elem),
                    ξ[1],
                    ξ[2],
                )
                SVector(getfield(x, 1), getfield(x, 2))
            end
            J = det(∂x∂ξ)
            ∂ξ∂x = inv(∂x∂ξ)
            WJ = J * quad_weights[i] * quad_weights[j]

            local_geometry_slab[i, j] = Geometry.LocalGeometry(
                x,
                J,
                WJ,
                Geometry.AxisTensor(
                    (Geometry.Cartesian12Axis(), Geometry.Covariant12Axis()),
                    ∂x∂ξ,
                ),
                Geometry.AxisTensor(
                    (
                        Geometry.Contravariant12Axis(),
                        Geometry.Cartesian12Axis(),
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
                false,
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
    n = if face == 1
        -J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 2
        J * ∂ξ∂x[1, :] * quad_weights[j]
    elseif face == 3
        -J * ∂ξ∂x[2, :] * quad_weights[i]
    elseif face == 4
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
