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
    lat_bounds = (-90, 90),
    long_bounds = (-180, 180),
)
    @assert -90 <= lat_bounds[1] < lat_bounds[2] <= 90
    @assert -180 <= long_bounds[1] < long_bounds[2] <= 180

    FT = typeof(domain.radius)
    lat = range(lat_bounds[1], lat_bounds[2], length = nlat)
    if long_bounds[1] + 360 ≈ long_bounds[2] # periodic, long_bounds[2] "=" long_bounds[1]
        long = range(
            Interval{:closed, :open, FT}(long_bounds[1], long_bounds[2]),
            nlong,
        )
    else
        long = range(long_bounds[1], long_bounds[2], length = nlong)
    end
    return RegularLatLongMesh{typeof.((domain, lat))...}(domain, lat, long)
end

domain(mesh::RegularLatLongMesh) = mesh.domain
ncoordinates(mesh::RegularLatLongMesh) = length(mesh.lat) * length(mesh.long)
