
"""
    PointCloudGrid(
        context :: ClimaComms.AbstractCommsContext,
        points  :: AbstractVector{Geometry.LatLongPoint{FT}},
    )

A horizontal grid consisting of N arbitrary, disconnected (lat, long) locations on a
sphere. There is no connectivity between columns; no spectral element basis, DSS, or
horizontal operators are supported on this grid.

This is the horizontal component used by a "point cloud" extruded space (N independent
columns at user-chosen sphere locations).

The `local_geometry` is stored as an `IFH{LG, 1, N}` data layout — one node per
"element" (`Ni = 1`), N "elements" (`Nh = N`). The horizontal Jacobian `∂x∂ξ` is the
diagonal matrix `diag(R·π/180, R·cosd(lat)·π/180)` reflecting the sphere metric at
each point, and `J = R²·cosd(lat)·(π/180)²`.
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

# No topology — return nothing. Callers that need a topology (e.g. DSS) should
# not be called on a PointCloudGrid.
topology(::PointCloudGrid) = nothing

local_geometry_data(grid::PointCloudGrid, ::Nothing) = grid.local_geometry

global_geometry(grid::PointCloudGrid) = grid.global_geometry

quadrature_style(::PointCloudGrid) =
    error("PointCloudGrid has no quadrature style (no spectral element basis)")

"""
    PointCloudGrid(
        points  :: AbstractVector{Geometry.LatLongPoint{FT}};
        radius  :: Real,
        device  :: ClimaComms.AbstractDevice = ClimaComms.device(),
        context :: ClimaComms.AbstractCommsContext = ClimaComms.context(device),
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
    context::ClimaComms.AbstractCommsContext = ClimaComms.SingletonCommsContext(device),
) where {FT}
    @assert context isa ClimaComms.SingletonCommsContext "PointCloudGrid only supports SingletonCommsContext."

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
        ∂x∂ξ_mat = SMatrix{2, 2, FT, 4}(0, s_lat, s_lon, 0)

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
