using IntervalSets

struct RegularLatLongMesh{S <: Domains.SphereDomain, R <: StepRangeLen} <:
       AbstractMesh2D
    domain::S
    lat::R
    long::R
end

function RegularLatLongMesh(
    domain,
    nlat::Int,
    nlong::Int,
    lat_bounds = (-π / 2, π / 2),
    long_bounds = (-π, π),
)
    @assert 0 < (lat_bounds[2] - lat_bounds[1]) <= π
    @assert 0 < (long_bounds[2] - long_bounds[1]) <= 2π

    FT = typeof(domain.radius)
    lat = range(ClosedInterval{FT}(lat_bounds[1], lat_bounds[2]), nlat) # lat [-π/2, π/2]
    if long_bounds[1] + 2π ≈ long_bounds[2] # periodic, long_bounds[2] "=" long_bounds[1]
        long = range(
            Interval{:closed, :open, FT}(long_bounds[1], long_bounds[2]),
            nlong,
        ) # long (-π, π]
    else
        long = range(ClosedInterval{FT}(long_bounds[1], long_bounds[2]), nlong) # long (-π, π]
    end
    return RegularLatLongMesh{typeof.((domain, lat))...}(domain, lat, long)
end

domain(mesh::RegularLatLongMesh) = mesh.domain
ncoordinates(mesh::RegularLatLongMesh) = length(mesh.lat) * length(mesh.long)
