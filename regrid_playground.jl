push!(LOAD_PATH, joinpath(@__DIR__, "."))

# using Plots
using ClimaCore: Domains, Meshes, Topologies, Spaces, Fields
using ClimaCore: Geometry, Operators
using ClimaCore.Spaces: Quadratures
using LinearAlgebra, IntervalSets, UnPack, SparseArrays

# Create domain (-2π,2π) x (-2π, 2π)
# domain = Domains.RectangleDomain(
#     Geometry.XPoint(0.)..Geometry.XPoint(1.),
#     Geometry.YPoint(0.)..Geometry.YPoint(1.),
#     x1periodic = true,
#     x2periodic = true,
# )

# # Gauss-Legendre-Lobatto quadrature with Nq quadrature points.
# Nq = 1 # number of quadrature points
# quad = Spaces.Quadratures.GL{Nq}()
# m1, n1 = 3,2 # elements
# # mesh: (:domain, :n1, :n2, :range1, :range2); ranges are the discretized intervals.
# mesh1 = Meshes.EquispacedRectangleMesh(domain, m1, n1)
# # topology: (:mesh, :boundaries); here, boundaries is empty since we have a periodic domain.
# topology1 = Topologies.GridTopology(mesh1)
# # space: (:topology, :quadrature_style, :local_geometry, :dss_weights,
# # :internal_surface_geometry, :boundary_surface_geometries)
# space1 = Spaces.SpectralElementSpace2D(topology1, quad) #***

# m2, n2 = 2,2
# mesh2 = Meshes.EquispacedRectangleMesh(domain, m2, n2)
# topology2 = Topologies.GridTopology(mesh2)
# space2 = Spaces.SpectralElementSpace2D(topology2, quad)

# R = Operators.LinearRemap(space2, space1)

# function init_state(coord)
#     @unpack x, y = coord
#     # set initial state
#     return 13.
# end

# # Fields.coordinate_field(space): returns x,y coords
# y1 = init_state.(Fields.coordinate_field(space1))


# plot(y1.θ)
#=
# **Gauss-Lobatto** quadrature with Nq quadrature points.
quad = Spaces.Quadratures.GL{Nq}()
# space: (:topology, :quadrature_style, :local_geometry, :dss_weights,
# :internal_surface_geometry, :boundary_surface_geometries)
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

quad_points, quad_weights =
        Quadratures.quadrature_points(FT, quadrature_style)

endpt = neither, left, right, or both and specifies which endpoints of the integration
interval should be included in the quadrature points.
function legendre(::Type{T}, n::Integer, 
                  endpt::EndPt=neither) where {T<:AbstractFloat}
    @assert n ≥ 1
    a, b = legendre_coefs(T, n)
    return custom_gauss_rule(-one(T), one(T), a, b, endpt)
end
custom_gauss_rule(-one(T), one(T), a, b, neither)
    ([0.0], [2.0000000000000004])
    neither chooses the the midpoint
custom_gauss_rule(-one(T), one(T), a, b, left)
    ([-1.0], [2.0000000000000004])
julia> custom_gauss_rule(-one(T), one(T), a, b, right)
    ([1.0], [2.0000000000000004])
custom_gauss_rule(-one(T), one(T), a, b, both)
    ERROR: Must have at least two points for both ends.
    breaks since Nq = 1 for GLL (not a thing for GL)
=#

################################################################################
# domain = Domains.IntervalDomain(
#     Geometry.XPoint(0.)..Geometry.XPoint(1.),
#     periodic = true,
# )

# # Gauss-Legendre-Lobatto quadrature with Nq quadrature points.
# Nq = 1 # number of quadrature points
# quad = Spaces.Quadratures.GL{Nq}()
# n1 = 4 # elements
# # mesh: (:domain, :n1, :n2, :range1, :range2); ranges are the discretized intervals.
# mesh1 = Meshes.IntervalMesh(domain; nelems = n1)
# # topology: (:mesh, :boundaries); here, boundaries is empty since we have a periodic domain.
# topology1 = Topologies.IntervalTopology(mesh1)
# # space: (:topology, :quadrature_style, :local_geometry, :dss_weights,
# # :internal_surface_geometry, :boundary_surface_geometries)
# space1 = Spaces.SpectralElementSpace1D(topology1, quad) #***

# n2 = 2
# mesh2 = Meshes.IntervalMesh(domain; nelems = n2)
# topology2 = Topologies.IntervalTopology(mesh2)
# space2 = Spaces.SpectralElementSpace1D(topology2, quad)

# R = Operators.LinearRemap(space2, space1)

################################################################################
domain = Domains.IntervalDomain(
    Geometry.XPoint(0.)..Geometry.XPoint(1.),
    periodic = true,
)

# Gauss-Legendre-Lobatto quadrature with Nq quadrature points.
Nq1 = 1 # number of quadrature points
quad1= Spaces.Quadratures.GL{Nq1}()
n1 = 2 # elements
# mesh: (:domain, :n1, :n2, :range1, :range2); ranges are the discretized intervals.
mesh1 = Meshes.IntervalMesh(domain; nelems = n1)
# topology: (:mesh, :boundaries); here, boundaries is empty since we have a periodic domain.
topology1 = Topologies.IntervalTopology(mesh1)
# space: (:topology, :quadrature_style, :local_geometry, :dss_weights,
# :internal_surface_geometry, :boundary_surface_geometries)
source = Spaces.SpectralElementSpace1D(topology1, quad1)

Nq2 = 3
quad2 = Spaces.Quadratures.GLL{Nq2}()
n2 = 1
mesh2 = Meshes.IntervalMesh(domain; nelems = n2)
topology2 = Topologies.IntervalTopology(mesh2)
target = Spaces.SpectralElementSpace1D(topology2, quad2)

# (FV -> Spectral)
function overlap_weights(target, source)
    FT = Spaces.undertype(source)
    target_topo = Spaces.topology(target)
    source_topo = Spaces.topology(source)
    nelems_t = Topologies.nlocalelems(target)
    nelems_s = Topologies.nlocalelems(source)
    QS_t = Spaces.quadrature_style(target)
    QS_s = Spaces.quadrature_style(source)
    Nq_t = Quadratures.degrees_of_freedom(QS_t)
    Nq_s = Quadratures.degrees_of_freedom(QS_s)
    W_ov = spzeros(nelems_t * Nq_t, nelems_s * Nq_s)
    
    # Calculate element overlap pattern
    # X_ov[i,j] = overlap length between target elem i and source elem j
    for i in 1:nelems_t
        vertices_i = Topologies.vertex_coordinates(target_topo, i)
        min_i, max_i = Geometry.component(vertices_i[1],1), Geometry.component(vertices_i[2],1)
        for j in 1:nelems_s
            vertices_j = Topologies.vertex_coordinates(source_topo, j)
            # get interval for quadrature
            min_j, max_j = Geometry.component(vertices_j[1],1), Geometry.component(vertices_j[2],1)
            min_ov, max_ov = max(min_i, min_j), min(max_i, max_j)
            ξ, w = Quadratures.quadrature_points(FT, QS_t)
            x_ov = FT(0.5) * (min_ov + max_ov) .+ FT(0.5) * (max_ov - min_ov) * ξ
            x_t = FT(0.5) * (min_i + max_i) .+ FT(0.5) * (max_i - min_i) * ξ

            I_mat = Quadratures.interpolation_matrix(x_t, x_ov)

            w' * I_mat * x_ov # I_mat * x_ov = x_t

            W_ov[i, j] = overlap
        end
    end
    return W_ov
end

function f(coord)
    @unpack x = coord
    x > 0.5 ? (return 1.0) : (return 0.0)
end

function overlap_space(target::T, source::S, xmin, xmax) where
    {T <: Spaces.SpectralElementSpace1D, S <: Spaces.SpectralElementSpace1D}
    Nq_t = Quadratures.degrees_of_freedom(Spaces.quadrature_style(target))
    Nq_s = Quadratures.degrees_of_freedom(Spaces.quadrature_style(source))
    Nq = max(Nq_t, Nq_s)
    quad = GLL{Nq}()
    domain = Domains.IntervalDomain(Geometry.XPoint(xmin)..Geometry.XPoint(xmax), boundary_tags = (:left, :right))
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(mesh)
    return Spaces.SpectralElementSpace1D(topology, quad)
end