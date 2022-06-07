using IntervalSets

struct RegularLatLongMesh <: AbstractMesh2D
    domain::Domains.SphereDomain
    nlat::Int
    nlong::Int
end

domain(mesh::RegularLatLongMesh) = mesh.domain

function coordinates_list(
    mesh::RegularLatLongMesh,
)
    r = domain(mesh).radius
    nlat, nlong = mesh.nlat, mesh.nlong
    FT = typeof(r)
    θ = range(Interval{:closed, :open}(FT(0), FT(2π)), nlong) # long (0, 2π]
    ϕ = range(ClosedInterval{FT}(0, π), nlat) # lat [0, π]
    coords = []
    for lat in ϕ
        for long in θ
            push!(coords, Geometry.LatLongPoint(lat, long))
        end
    end
    return coords
end

# space needs a spherical setup
# operator array will need to map from nodes in src space to pts..
using SparseArrays
function generate_midpoint_remap(target::Meshes.RegularLatLongMesh, source)
    target_coords = coordinates_list(target)
    nelems = Topologies.nlocalelems(source)
    QS_s = Spaces.quadrature_style(source)
    Nq_s = Quadratures.degrees_of_freedom(QS_s)
    op = spzeros(length(target_coords), nelems * Nq_s)
    source_mesh = Spaces.topology(space).mesh
    for coord in target_coords
        elem = Meshes.containing_element(source_mesh, coord)
        ξ = Meshes.reference_coordinates(source_mesh, elem, coord)

    end


end

#= playground
FT = Float64
radius = FT(3)
ne = 4
Nq = 4
spheredomain = Domains.SphereDomain(radius)
mesh = Meshes.EquiangularCubedSphere(spheredomain, ne)
topology = Topologies.Topology2D(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(topology, quad)

rllmesh = Meshes.RegularLatLongMesh(spheredomain, 5, 5)


=#