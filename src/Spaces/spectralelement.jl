"""
    SpectralElementSpace2D <: AbstractSpace

A two-dimensional space: within each element the space is represented as a polynomial.
"""
struct SpectralElementSpace2D{T, Q, G, D, IS, BS} <: AbstractSpace
    topology::T
    quadrature_style::Q
    local_geometry::G
    dss_weights::D
    internal_surface_geometry::IS
    boundary_surface_geometries::BS
end

Topologies.nlocalelems(Space::AbstractSpace) =
    Topologies.nlocalelems(Space.topology)

undertype(space::AbstractSpace) =
    DataLayouts.basetype(eltype(space.local_geometry))

function Base.show(io::IO, Space::SpectralElementSpace2D)
    println(io, "SpectralElementSpace2D:")
    println(io, "  topology: ", Space.topology)
    println(io, "  quadrature: ", Space.quadrature_style)
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
    LG = Geometry.LocalGeometry{CT, FT, SMatrix{2, 2, FT, 4}}

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
                SVector(x.x1, x.x2)
            end
            J = det(∂x∂ξ)
            ∂ξ∂x = inv(∂x∂ξ)
            WJ = J * quad_weights[i] * quad_weights[j]

            local_geometry_slab[i, j] =
                Geometry.LocalGeometry(x, J, WJ, ∂x∂ξ, ∂ξ∂x)
        end
    end

    # dss_weights = J ./ dss(J)
    dss_weights = copy(local_geometry.J)
    horizontal_dss!(dss_weights, local_geometry.J, topology, Nq)
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
                local_geometry_slab⁻,
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

function compute_surface_geometry(
    local_geometry_slab,
    quad_weights,
    face,
    q,
    reversed = false,
)
    Nq = length(quad_weights)
    @assert size(local_geometry_slab) == (Nq, Nq)
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
    return Geometry.SurfaceGeometry(sWJ, Geometry.Cartesian12Vector(n...))
end

coordinates_data(space::SpectralElementSpace2D) =
    space.local_geometry.coordinates
local_geometry_data(space::SpectralElementSpace2D) = space.local_geometry

function variational_solve!(data, space::AbstractSpace)
    data .= RecursiveApply.rdiv.(data, space.local_geometry.WJ)
end

"""
    SpectralElementSpaceSlab <: AbstractSpace

A view into a `SpectralElementSpace2D` for a single slab.
"""
struct SpectralElementSpaceSlab{Q, G} <: AbstractSpace
    quadrature_style::Q
    local_geometry::G
end

function slab(space::SpectralElementSpace2D, h)
    SpectralElementSpaceSlab(
        space.quadrature_style,
        slab(space.local_geometry, h),
    )
end
