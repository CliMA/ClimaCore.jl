
import ClimaCore
using Makie, GLMakie, ClimaCoreMakie

using ClimaCore.CommonSpaces
space = CubedSphereSpace(;
    radius = 10,
    n_quad_points = 4,
    h_elem = 10,
)
coords = ClimaCore.Fields.coordinate_field(space)


# a scalar field
x = map(coords) do coord
    ϕ = coord.lat
    λ = coord.long

    cosd(λ) * sind(ϕ)
end


plot(x)

# 1. GeoMakie: make it play nice with basic projections
#  - cuts/seams: we use degrees:  lat in (-90,90), long in (-180, 180)
#  - if you use an even number of elements across each panel, you don't have to do anything
#  - (not now, but eventually)
#     - odd numbers of elements
#     - deal nicely with pole (avoid singularity)
#     - polar or other local projections (which don't show the whole map)
#  - map overlay
#  - mesh overlay (ClimaCore mesh, not Makie mesh), also helpful in 3D sphere
#  - node overlay (less important)
# 2. Performance / interface
#  - if doing dashboards, etc, avoid unnecessary computations (recomputing coordinates, triangulations)
#    - not sure where best to cache this stuff?
#    - interface to update values without updating the mesh
#  - make it work for large numbers of elements (e.g. 100 elemnts across each face = 6*100^2*(4*4*2)) = 1_920_000
# 3. Usability: a simple prototype script that gets an interactive dashboard
#  - ideally support both local and web (initially can just be one)
#  - sliders for time and level (altitude slice)
# 4. Upgrade ClimaCoreMakie to latest Makie release
# 5. Plots along latitudinal slices (long on x axis, altitude / pressure as vertical)
#   - we could remap to LatLong grid
#   - alternativeley, handle the slicing directly on the GPU
# 6. Have support for vector types (velocity, momentum, fluxes): medium priority, tackle if straightforward, but can skip
#  - it would be nice to represent them as arrows (quiver plot)
#  - auto rescaling (or tunable) to make it useful
#  - store vectors (covariant, contravariant, local): can convert to Cartesian123Vector
#  - only needed for 3D generally

# a vector field (in local Cartesian coordinates)
u = map(coords) do coord
    u0 = 20.0
    α0 = 45.0
    ϕ = coord.lat
    λ = coord.long

    uu = u0 * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
    uv = -u0 * sind(α0) * sind(λ)
    ClimaCore.Geometry.UVVector(uu, uv)
end

# a vector field (in Cartesian coordinates)
u_cart = ClimaCore.Geometry.Cartesian123Vector.(u)
