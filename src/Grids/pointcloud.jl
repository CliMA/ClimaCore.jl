
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

The `local_geometry` is stored as an `IFH{LG, 1, N}` data layout, with each of
the `N` locations represented by an element with one nodal point. Based on the
[metric
tensor](https://en.wikipedia.org/wiki/Metric_tensor#The_round_metric_on_a_sphere)
of a sphere, the horizontal Jacobian `∂x∂ξ` is given by the diagonal matrix
`diag(R·π/180, R·cosd(lat)·π/180)`, with the determinant `J =
R²·cosd(lat)·(π/180)²`.
"""
struct PointCloudGrid{
    C <: ClimaComms.AbstractCommsContext,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
} <: AbstractSpectralElementGrid
    context::C
    global_geometry::GG
    local_geometry::LG  # IFH{LocalGeometry{(1,2), LatLongPoint{FT}, FT, Padded∂x∂ξ{FT}, PaddedContravariantMetric{FT}}, 1, N}
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

quadrature_style(::PointCloudGrid) =
    error("PointCloudGrid has no quadrature style (no spectral element basis)")

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
    # PointCloudGrid is single-process only; the context is always a
    # SingletonCommsContext built from the given device.
    context = ClimaComms.SingletonCommsContext(device)

    N = length(points)
    global_geometry = Geometry.SphericalGlobalGeometry(FT(radius))

    AIdx = Geometry.coordinate_axis(Geometry.LatLongPoint{FT})  # (1, 2)
    LG = Geometry.LocalGeometryType(Geometry.LatLongPoint{FT}, FT, AIdx)

    # Ni = 1 (one node per "column element"), Nh = N
    local_geometry = DataLayouts.IFH{LG, 1}(Array{FT}, N)

    ∂x∂ξ_axes = (
        Geometry.LocalAxis{AIdx}(),
        Geometry.CovariantAxis{AIdx}(),
    )
    deg2rad = FT(π) / 180

    for (h, pt) in enumerate(points)
        # Sphere metric: arc length per degree in each coordinate direction.
        # ∂x∂ξ is diagonal: (R·π/180) in the lat direction,
        #                    (R·cosd(lat)·π/180) in the lon direction.
        s_lat = FT(radius) * deg2rad
        s_lon = FT(radius) * deg2rad * cosd(pt.lat)
        J = s_lat * s_lon   # det of diagonal Jacobian
        ∂x∂ξ_mat = SMatrix{2, 2, FT, 4}(zero(FT), s_lat, s_lon, zero(FT))

        lg_slab = slab(local_geometry, h)
        lg_slab[slab_index(1)] = Geometry.LocalGeometry(
            pt,
            J,
            J,   # WJ — unit quadrature weight × J
            Geometry.AxisTensor(∂x∂ξ_axes, ∂x∂ξ_mat),
        )
    end

    DA = ClimaComms.array_type(device)
    return PointCloudGrid(
        context,
        global_geometry,
        DataLayouts.rebuild(local_geometry, DA),
    )
end

function Base.show(io::IO, grid::PointCloudGrid)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, nameof(typeof(grid)), ":")
    # some reduced spaces (like slab space) do not have topology
    println(iio, " "^(indent + 2), "horizontal:")
    print(iio, " "^(indent + 4), "context: Nothing")
end
