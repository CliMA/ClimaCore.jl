using IntervalSets

struct RegularLatLongMesh{S <: Domains.SphereDomain, R <: StepRangeLen} <: AbstractMesh2D
    domain::S
    lat::R
    long::R
end

function RegularLatLongMesh(domain, nlat::Int, nlong::Int, lat_bounds = (-π/2, π/2), long_bounds = (-π, π))
    @assert 0 < (lat_bounds[2] - lat_bounds[1]) <= π
    @assert 0 < (long_bounds[2] - long_bounds[1]) <= 2π
    
    FT = typeof(domain.radius)
    lat = range(ClosedInterval{FT}(lat_bounds[1], lat_bounds[2]), nlat) # lat [-π/2, π/2]
    if long_bounds[1] + 2π ≈ long_bounds[2] # periodic, long_bounds[2] "=" long_bounds[1]
        long = range(Interval{:closed, :open, FT}(long_bounds[1], long_bounds[2]), nlong) # long (-π, π]
    else
        long = range(ClosedInterval{FT}(long_bounds[1], long_bounds[2]), nlong) # long (-π, π]
    end
    return RegularLatLongMesh{typeof.((domain, lat))...}(domain, lat, long)
end

domain(mesh::RegularLatLongMesh) = mesh.domain

function coordinates_list(
    mesh::RegularLatLongMesh,
)
    latitude, longitude = mesh.lat, mesh.long
    coords = []
    for lat in latitude
        for long in longitude
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