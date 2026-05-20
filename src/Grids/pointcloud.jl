
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
"element" (`Ni = 1`), N "elements" (`Nh = N`). The horizontal Jacobian is set to 1 and
`∂x∂ξ` is the 2×2 identity, since horizontal metric terms are not used.
"""
struct PointCloudGrid{
    C <: ClimaComms.AbstractCommsContext,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
} <: AbstractSpectralElementGrid
    context::C
    global_geometry::GG
    local_geometry::LG  # IFH{FullLocalGeometry{(1,2), LatLongPoint{FT}, FT, SMatrix{2,2,FT,4}}, 1, N}
end

Adapt.@adapt_structure PointCloudGrid

local_geometry_type(::Type{PointCloudGrid{C, GG, LG}}) where {C, GG, LG} =
    eltype(LG)

ClimaComms.context(grid::PointCloudGrid) = grid.context
ClimaComms.device(grid::PointCloudGrid) = ClimaComms.device(grid.context)

# No topology — return nothing. Callers that need a topology (e.g. DSS) should
# not be called on a PointCloudGrid.
topology(grid::PointCloudGrid) = nothing

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
`LatLongPoint`s and a sphere `radius`. Coordinates are stored in the
`local_geometry` field; the horizontal Jacobian is set to 1.
"""
function PointCloudGrid(
    points::AbstractVector{Geometry.LatLongPoint{FT}};
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.SingletonCommsContext(device),
) where {FT}
    @assert context isa ClimaComms.SingletonCommsContext "PointCloudGrid only supports SingletonCommsContext."

    N = length(points)
    global_geometry = Geometry.CartesianGlobalGeometry()

    AIdx = Geometry.coordinate_axis(Geometry.LatLongPoint{FT})  # (1, 2)
    LG = Geometry.FullLocalGeometry{
        AIdx,
        Geometry.LatLongPoint{FT},
        FT,
        SMatrix{2, 2, FT, 4},
    }

    # Ni = 1 (one node per "column element"), Nh = N
    # TODO: I part looks weird
    local_geometry = DataLayouts.IFH{LG, 1}(Array{FT}, N)

    I2 = SMatrix{2, 2, FT, 4}(one(FT), zero(FT), zero(FT), one(FT))
    ∂x∂ξ_axes = (
        Geometry.LocalAxis{AIdx}(),
        Geometry.CovariantAxis{AIdx}(),
    )

    for (h, pt) in enumerate(points)
        lg_slab = slab(local_geometry, h)
        lg_slab[slab_index(1)] = Geometry.LocalGeometry(
            pt,
            one(FT),   # J  — no horizontal metric scaling
            one(FT),   # WJ — weight × J
            Geometry.AxisTensor(∂x∂ξ_axes, I2),
        )
    end

    DA = ClimaComms.array_type(device)
    return PointCloudGrid(
        context,
        global_geometry,
        DataLayouts.rebuild(local_geometry, DA),
    )
end
