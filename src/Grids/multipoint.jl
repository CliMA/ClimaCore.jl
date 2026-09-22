
"""
    MultiPointGrid{C, GG, LG} <: AbstractSpectralElementGrid

Horizontal grid of `N` arbitrary, disconnected locations: (lat, long) points on a
sphere, or columns without horizontal coordinates. There is no connectivity between
columns and no spectral element basis, so DSS is not supported. Horizontal
spectral-element derivative operators (`Gradient`, `Divergence`, `Curl`, and their
weak forms) are accepted and evaluate to zero, as on a single-column
`FiniteDifferenceSpace`.

This is the horizontal component of a multi-column extruded space (`N` independent
columns). Construct it with `MultiPointGrid(points; radius, device)` for columns at
(lat, long) points, or with `MultiPointGrid(FT, ncolumns; device)` for columns
without horizontal coordinates, whose extruded columns have `ZPoint` coordinates
and the local geometry of a single column.

# Fields

  - `context`: The `ClimaComms.SingletonCommsContext` of the grid.
  - `global_geometry`: The `Geometry.SphericalGlobalGeometry` of the sphere, or a
    `Geometry.CartesianGlobalGeometry` for columns without horizontal coordinates.
  - `local_geometry`: A `VIJFH{LG, 1, 1, 1, N}` data layout, with each of the `N`
    locations represented by an element with one nodal point.

Every location has unit horizontal metric terms (`∂x∂ξ = I`, `J = WJ = 1`). The
sphere radius is only used for the global geometry.
"""
struct MultiPointGrid{
    C <: ClimaComms.AbstractCommsContext,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
} <: AbstractSpectralElementGrid
    context::C
    global_geometry::GG
    local_geometry::LG  # VIJFH{LocalGeometry, 1, 1, 1, N}
end

Adapt.@adapt_structure MultiPointGrid

local_geometry_type(::Type{MultiPointGrid{C, GG, LG}}) where {C, GG, LG} =
    eltype(LG)

ClimaComms.context(grid::MultiPointGrid) = grid.context
ClimaComms.device(grid::MultiPointGrid) = ClimaComms.device(grid.context)

topology(::MultiPointGrid) = error(
    "MultiPointGrid has no topology",
)

local_geometry_data(grid::MultiPointGrid, ::Nothing) = grid.local_geometry
global_geometry(grid::MultiPointGrid) = grid.global_geometry

quadrature_style(::MultiPointGrid) = nothing

# The horizontal coordinate of a column without horizontal position, internal to
# `MultiPointGrid`. Its product with a vertical coordinate is that coordinate, so the
# extruded columns have `ZPoint` coordinates and the local geometry of a single column.
struct EmptyPoint{FT} <: Geometry.AbstractPoint{FT} end
Base.eltype(::Type{EmptyPoint{FT}}) where {FT} = FT
Geometry.coordinate_axis(::Type{<:EmptyPoint}) = ()
Geometry.product_coordinates(::EmptyPoint, zp::Geometry.ZPoint) = zp
Geometry.product_coordinates(::EmptyPoint, pp::Geometry.PPoint) = pp

"""
    MultiPointGrid(
        points  :: AbstractVector{Geometry.LatLongPoint{FT}};
        radius  :: Real,
        device  :: ClimaComms.AbstractDevice = ClimaComms.device(),
    )

Build a `MultiPointGrid` from a vector of `LatLongPoint`s and a sphere `radius`.
Every location has unit horizontal metric terms.

# Keyword Arguments

  - `radius`: Sphere radius [m]; must be positive.
  - `device`: `ClimaComms.AbstractDevice` on which the geometry is stored; defaults to
    `ClimaComms.device()`.
"""
function MultiPointGrid(
    points::AbstractVector{Geometry.LatLongPoint{FT}};
    radius::Real,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
) where {FT}
    radius > 0 || error("Radius ($radius) must be positive")
    get!(Cache.OBJECT_CACHE, (MultiPointGrid, copy(points), radius, device)) do
        _MultiPointGrid(points, Geometry.SphericalGlobalGeometry(FT(radius)); device)
    end
end

"""
    MultiPointGrid(
        ::Type{FT},
        ncolumns::Integer;
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
    )

Build a `MultiPointGrid` of `ncolumns` columns without horizontal coordinates, on a
`Geometry.CartesianGlobalGeometry`. Extruded along a vertical grid, its columns have
`ZPoint` coordinates and the local geometry of a single-column
`FiniteDifferenceGrid`.
"""
function MultiPointGrid(
    ::Type{FT},
    ncolumns::Integer;
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
) where {FT}
    ncolumns > 0 || error("Number of columns ($ncolumns) must be positive")
    get!(Cache.OBJECT_CACHE, (MultiPointGrid, FT, ncolumns, device)) do
        points = fill(EmptyPoint{FT}(), ncolumns)
        _MultiPointGrid(points, Geometry.CartesianGlobalGeometry(); device)
    end
end

function _MultiPointGrid(
    points::AbstractVector{<:Geometry.AbstractPoint{FT}},
    global_geometry::Geometry.AbstractGlobalGeometry;
    device::ClimaComms.AbstractDevice,
) where {FT}
    # MultiPointGrid is single-process only; the context is always a
    # SingletonCommsContext built from the given device.
    context = ClimaComms.SingletonCommsContext(device)

    N = length(points)

    AIdx = Geometry.coordinate_axis(eltype(points))  # (1, 2) for (lat, long) points
    LG = Geometry.LocalGeometryType(eltype(points), FT, AIdx)

    # Nv = Ni = Nj = 1 (one node per "column element"), Nh = N
    local_geometry = DataLayouts.VIJFH{LG, 1, 1, 1, nothing}(Array{FT}, N)

    ∂x∂ξ = Geometry.Tensor(
        FT(1) * I,
        (
            Geometry.Components{Geometry.Orthonormal, AIdx}(),
            Geometry.Components{Geometry.Covariant, AIdx}(),
        ),
    )

    for (h, pt) in enumerate(points)
        local_geometry[1, 1, 1, h] =
            Geometry.LocalGeometry(pt, one(FT), one(FT), ∂x∂ξ)
    end

    DA = ClimaComms.array_type(device)
    return MultiPointGrid(
        context,
        global_geometry,
        DataLayouts.rebuild(local_geometry, DA),
    )
end

Meshes.domain(grid::MultiPointGrid{<:Any, <:Geometry.AbstractSphericalGlobalGeometry}) =
    Domains.SphereDomain(grid.global_geometry.radius)

function print_multipoint_horizontal(io::IO, grid::MultiPointGrid, indent)
    println(io, " "^(indent + 2), "horizontal:")
    print(io, " "^(indent + 4), "context: ")
    Topologies.print_context(io, grid.context)
    println(io)
    print(io, " "^(indent + 4), "points: ", DataLayouts.nelems(grid.local_geometry))
    if grid.global_geometry isa Geometry.AbstractSphericalGlobalGeometry
        println(io)
        print(io, " "^(indent + 4), "radius: ", grid.global_geometry.radius)
    end
end

function Base.show(io::IO, grid::MultiPointGrid)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, nameof(typeof(grid)), ":")
    print_multipoint_horizontal(iio, grid, indent)
end

# Deprecated alias of `MultiPointGrid`.
Base.@deprecate_binding PointCloudGrid MultiPointGrid false

# Grids with no horizontal spectral elements are continuous: every node
# belongs to exactly one element, so there is nothing for DSS to reconcile.
discretization(grid::AbstractFiniteDifferenceGrid) = CG()
discretization(grid::MultiPointGrid) = CG()
