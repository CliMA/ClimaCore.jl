
"""
    PointCloudGrid(
        context :: ClimaComms.AbstractCommsContext,
        points  :: AbstractVector{Geometry.LatLongPoint{FT}},
    )

A horizontal grid consisting of N arbitrary, disconnected (lat, long) locations
on a sphere. There is no connectivity between columns; no spectral element
basis, DSS, or horizontal operators are supported on this grid.

This is the horizontal component used by a "point cloud" extruded space (N
independent columns at user-chosen sphere locations).

The `local_geometry` is stored as a `VIJFH{LG, 1, 1, 1, N}` data layout, with
each of the `N` locations represented by an element with one nodal point. Based
on the [metric
tensor](https://en.wikipedia.org/wiki/Metric_tensor#The_round_metric_on_a_sphere)
of a sphere, the horizontal Jacobian `∂x∂ξ` is given by the diagonal matrix
`diag(R·π/180, R·cosd(lat)·π/180)`, with the determinant `J = R²·cosd(lat)·(π/180)²`.
"""
struct PointCloudGrid{
    C <: ClimaComms.AbstractCommsContext,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
} <: AbstractSpectralElementGrid
    context::C
    global_geometry::GG
    local_geometry::LG  # VIJFH{LocalGeometry{(1,2), LatLongPoint{FT}, ...}, 1, 1, 1, N}
end

Adapt.@adapt_structure PointCloudGrid

local_geometry_type(::Type{PointCloudGrid{C, GG, LG}}) where {C, GG, LG} =
    eltype(LG)

ClimaComms.context(grid::PointCloudGrid) = grid.context
ClimaComms.device(grid::PointCloudGrid) = ClimaComms.device(grid.context)

topology(::PointCloudGrid) = error(
    "PointCloudGrid has no topology",
)

local_geometry_data(grid::PointCloudGrid, ::Nothing) = grid.local_geometry
global_geometry(grid::PointCloudGrid) = grid.global_geometry

quadrature_style(::PointCloudGrid) = nothing

"""
    PointCloudGrid(
        points  :: AbstractVector{Geometry.LatLongPoint{FT}};
        radius  :: Real,
        device  :: ClimaComms.AbstractDevice = ClimaComms.device(),
    )

Convenience constructor: build a `PointCloudGrid` from a vector of
`LatLongPoint`s and a sphere `radius`. The horizontal metric terms in
`local_geometry` are set from the sphere geometry at each point:
`∂x∂ξ = diag(R·π/180, R·cosd(lat)·π/180)`, `J = R²·cosd(lat)·(π/180)²`.
"""
function PointCloudGrid(
    points::AbstractVector{Geometry.LatLongPoint{FT}};
    radius::Real,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
) where {FT}
    get!(Cache.OBJECT_CACHE, (PointCloudGrid, copy(points), radius, device)) do
        _PointCloudGrid(points; radius, device)
    end
end

function _PointCloudGrid(
    points::AbstractVector{Geometry.LatLongPoint{FT}};
    radius::Real,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
) where {FT}
    radius > 0 || error("Radius ($radius) must be positive")

    # PointCloudGrid is single-process only; the context is always a
    # SingletonCommsContext built from the given device.
    context = ClimaComms.SingletonCommsContext(device)

    N = length(points)
    global_geometry = Geometry.SphericalGlobalGeometry(FT(radius))

    AIdx = Geometry.coordinate_axis(Geometry.LatLongPoint{FT})  # (1, 2)
    LG = Geometry.LocalGeometryType(Geometry.LatLongPoint{FT}, FT, AIdx)

    # Nv = Ni = Nj = 1 (one node per "column element"), Nh = N
    local_geometry = DataLayouts.VIJFH{LG, 1, 1, 1, nothing}(Array{FT}, N)

    ∂x∂ξ_axes = (
        Geometry.LocalAxis{AIdx}(),
        Geometry.CovariantAxis{AIdx}(),
    )
    deg2rad = FT(π) / 180

    for (h, pt) in enumerate(points)
        abs(pt.lat) < 90 || throw(ArgumentError(
            "Latitude ($(pt.lat)) must satisfy |lat| < 90",
        ))

        # Sphere metric: arc length per degree in each coordinate direction.
        # ∂x∂ξ is diagonal: (R·π/180) in the lat direction,
        #                    (R·cosd(lat)·π/180) in the lon direction.
        s_lat = FT(radius) * deg2rad
        s_lon = FT(radius) * deg2rad * cosd(pt.lat)
        J = s_lat * s_lon   # det of diagonal Jacobian
        ∂x∂ξ_mat = SMatrix{2, 2, FT, 4}(s_lon, zero(FT), zero(FT), s_lat)
        local_geometry[1, 1, 1, h] = Geometry.LocalGeometry(
            pt,
            J,
            J,   # WJ — unit quadrature weight × J
            Geometry.Tensor(∂x∂ξ_mat, ∂x∂ξ_axes),
        )
    end

    DA = ClimaComms.array_type(device)
    return PointCloudGrid(
        context,
        global_geometry,
        DataLayouts.rebuild(local_geometry, DA),
    )
end

function print_pointcloud_horizontal(io::IO, grid::PointCloudGrid, indent)
    println(io, " "^(indent + 2), "horizontal:")
    print(io, " "^(indent + 4), "context: ")
    Topologies.print_context(io, grid.context)
    println(io)
    println(
        io,
        " "^(indent + 4),
        "points: ",
        DataLayouts.nelems(grid.local_geometry),
    )
    print(io, " "^(indent + 4), "radius: ", grid.global_geometry.radius)
end

function Base.show(io::IO, grid::PointCloudGrid)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, nameof(typeof(grid)), ":")
    print_pointcloud_horizontal(iio, grid, indent)
end
