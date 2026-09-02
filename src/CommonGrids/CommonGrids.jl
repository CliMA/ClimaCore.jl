"""
    CommonGrids

Keyword constructors for the grids used in most ClimaCore configurations:
[`ExtrudedCubedSphereGrid`](@ref), [`CubedSphereGrid`](@ref),
[`ColumnGrid`](@ref), [`Box3DGrid`](@ref), [`SliceXZGrid`](@ref),
[`RectangleXYGrid`](@ref), and [`MultiColumnGrid`](@ref). Each takes the float
type as an optional first argument (default `Float64`) and the configuration as
keyword arguments, and composes the domain, mesh, topology, and grid from the
`Domains`, `Meshes`, `Topologies`, and `Grids` modules. Configurations without
a constructor here are composed from those modules directly.

The constructors build the type parameters of the grid from runtime values, so
their return type is not inferred by the compiler. Code that builds a grid in a
performance-critical section and knows the type parameters at compile time can
call the `Grids` constructors directly.

# Examples

The grid built by hand as

```julia
using ClimaComms
using ClimaCore: DataLayouts, Geometry, Topologies, Quadratures, Domains, Meshes, Grids
FT = Float64
z_elem = 63
z_min = FT(0)
z_max = FT(1)
radius = FT(6.371229e6)
h_elem = 15
n_quad_points = 4
device = ClimaComms.device()
context = ClimaComms.context(device)
hypsography = Grids.Flat()
global_geometry = Geometry.ShallowSphericalGlobalGeometry{FT}(radius)
quad = Quadratures.GLL{n_quad_points}()
h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{FT}(radius), h_elem)
h_topology = Topologies.Topology2D(context, h_mesh)
z_boundary_names = (:bottom, :top)
z_domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(z_min),
    Geometry.ZPoint{FT}(z_max);
    boundary_names = z_boundary_names,
)
z_mesh = Meshes.IntervalMesh(z_domain; nelems = z_elem)
h_grid = Grids.SpectralElementGrid2D(h_topology, quad)
z_topology = Topologies.IntervalTopology(ClimaComms.SingletonCommsContext(device), z_mesh)
z_grid = Grids.FiniteDifferenceGrid(z_topology)
grid = Grids.ExtrudedFiniteDifferenceGrid(
    h_grid,
    z_grid,
    hypsography,
    global_geometry,
)
```

corresponds to

```julia
using ClimaCore.CommonGrids: ExtrudedCubedSphereGrid
grid = ExtrudedCubedSphereGrid(;
    z_elem = 63,
    z_min = 0,
    z_max = 1,
    radius = 6.371229e6,
    h_elem = 15,
    n_quad_points = 4,
)
```
"""
module CommonGrids

export ExtrudedCubedSphereGrid,
    CubedSphereGrid,
    ColumnGrid,
    Box3DGrid,
    SliceXZGrid,
    RectangleXYGrid,
    MultiColumnGrid,
    PointColumnEnsembleGrid

import ClimaComms
import ..DataLayouts,
    ..Meshes, ..Topologies, ..Geometry, ..Domains, ..Quadratures, ..Grids

include("Helpers.jl")
import .Helpers.DefaultSliceXMesh
import .Helpers.DefaultZMesh
import .Helpers.DefaultRectangleXYMesh

#####
##### Grids
#####

"""
    ExtrudedCubedSphereGrid([FT = Float64]; z_elem, z_min, z_max, radius, h_elem,
                            n_quad_points, kwargs...)

Construct a [`Grids.ExtrudedFiniteDifferenceGrid`](@ref) on a cubed sphere: a
[`Grids.SpectralElementGrid2D`](@ref) horizontal grid extruded along a
[`Grids.FiniteDifferenceGrid`](@ref) vertical grid, with a shallow spherical
global geometry.

# Arguments

  - `FT`: The floating-point type, `Float32` or `Float64`.

# Keyword Arguments

  - `z_elem::Integer`: Number of vertical elements.
  - `z_min::Real`, `z_max::Real`: Vertical extent of the domain.
  - `radius::Real`: Radius of the sphere.
  - `h_elem::Integer`: Number of horizontal elements per side of each of the six
    cubed-sphere panels.
  - `n_quad_points::Integer`: Number of quadrature points per element side.
  - `device = ClimaComms.device()`: The `ClimaComms` device.
  - `context = ClimaComms.context(device)`: The `ClimaComms` context; its device
    must equal `device`.
  - `stretch = Meshes.Uniform()`: The `Meshes.StretchingRule` of the default
    vertical mesh; see [`Meshes.Uniform`](@ref).
  - `hypsography_fun = (h_grid, z_grid) -> Grids.Flat()`: A callable that returns
    the `Grids.HypsographyAdaption` for the given horizontal and vertical grids.
  - `global_geometry = Geometry.ShallowSphericalGlobalGeometry{FT}(radius)`: The
    `Geometry.AbstractGlobalGeometry` of the extruded grid.
  - `quad = Quadratures.GLL{n_quad_points}()`: The `Quadratures.QuadratureStyle`.
  - `discretization = nothing`, `VIJH = DataLayouts.VIJFH`,
    `enable_bubble = false`, `enable_mask = false`: Passed to the horizontal
    [`Grids.SpectralElementGrid2D`](@ref), which documents them.
  - `h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{FT}(radius), h_elem)`:
    The horizontal mesh.
  - `h_topology = Topologies.Topology2D(context, h_mesh, Topologies.spacefillingcurve(h_mesh))`:
    The horizontal topology.
  - `z_mesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch)`: The vertical
    `Meshes.IntervalMesh`, with boundaries named `:bottom` and `:top`.

# Examples

```julia
using ClimaCore.CommonGrids
grid = ExtrudedCubedSphereGrid(;
    z_elem = 10,
    z_min = 0,
    z_max = 1,
    radius = 10,
    h_elem = 10,
    n_quad_points = 4,
)
```
"""
ExtrudedCubedSphereGrid(; kwargs...) =
    ExtrudedCubedSphereGrid(Float64; kwargs...)

function ExtrudedCubedSphereGrid(
    ::Type{FT};
    z_elem::Integer,
    z_min::Real,
    z_max::Real,
    radius::Real,
    h_elem::Integer,
    n_quad_points::Integer,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
    hypsography_fun = (h_grid, z_grid) -> Grids.Flat(),
    global_geometry::Geometry.AbstractGlobalGeometry = Geometry.ShallowSphericalGlobalGeometry{
        FT,
    }(
        radius,
    ),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{n_quad_points}(),
    discretization::Union{Nothing, Grids.Discretization} = nothing,
    h_mesh = Meshes.EquiangularCubedSphere(
        Domains.SphereDomain{FT}(radius),
        h_elem,
    ),
    h_topology::Topologies.AbstractDistributedTopology = Topologies.Topology2D(
        context,
        h_mesh,
        Topologies.spacefillingcurve(h_mesh),
    ),
    VIJH::Type{<:DataLayouts.VIJHWithF} = DataLayouts.VIJFH,
    z_mesh::Meshes.IntervalMesh = DefaultZMesh(
        FT;
        z_min,
        z_max,
        z_elem,
        stretch,
    ),
    enable_bubble::Bool = false,
    enable_mask::Bool = false,
) where {FT}
    @assert ClimaComms.device(context) == device "The given device and context device do not match."

    z_boundary_names = (:bottom, :top)
    h_grid = Grids.SpectralElementGrid2D(
        h_topology,
        quad;
        VIJH,
        enable_bubble,
        enable_mask,
        discretization,
    )
    z_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        z_mesh,
    )
    z_grid = Grids.FiniteDifferenceGrid(z_topology)
    return Grids.ExtrudedFiniteDifferenceGrid(
        h_grid,
        z_grid,
        hypsography_fun(h_grid, z_grid),
        global_geometry,
    )
end

"""
    CubedSphereGrid([FT = Float64]; radius, h_elem, n_quad_points, kwargs...)

Construct a [`Grids.SpectralElementGrid2D`](@ref) on the surface of a cubed
sphere.

# Arguments

  - `FT`: The floating-point type, `Float32` or `Float64`.

# Keyword Arguments

  - `radius::Real`: Radius of the sphere.
  - `h_elem::Integer`: Number of elements per side of each of the six
    cubed-sphere panels.
  - `n_quad_points::Integer`: Number of quadrature points per element side.
  - `device = ClimaComms.device()`: The `ClimaComms` device.
  - `context = ClimaComms.context(device)`: The `ClimaComms` context; its device
    must equal `device`.
  - `quad = Quadratures.GLL{n_quad_points}()`: The `Quadratures.QuadratureStyle`.
  - `discretization = nothing`, `VIJH = DataLayouts.VIJFH`,
    `enable_mask = false`: Passed to [`Grids.SpectralElementGrid2D`](@ref),
    which documents them.
  - `h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{FT}(radius), h_elem)`:
    The mesh.
  - `h_topology = Topologies.Topology2D(context, h_mesh, Topologies.spacefillingcurve(h_mesh))`:
    The topology.

# Examples

```julia
using ClimaCore.CommonGrids
grid = CubedSphereGrid(; radius = 10, n_quad_points = 4, h_elem = 10)
```
"""
CubedSphereGrid(; kwargs...) = CubedSphereGrid(Float64; kwargs...)
function CubedSphereGrid(
    ::Type{FT};
    radius::Real,
    h_elem::Integer,
    n_quad_points::Integer,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{n_quad_points}(),
    discretization::Union{Nothing, Grids.Discretization} = nothing,
    h_mesh = Meshes.EquiangularCubedSphere(
        Domains.SphereDomain{FT}(radius),
        h_elem,
    ),
    h_topology::Topologies.AbstractDistributedTopology = Topologies.Topology2D(
        context,
        h_mesh,
        Topologies.spacefillingcurve(h_mesh),
    ),
    VIJH::Type{<:DataLayouts.VIJHWithF} = DataLayouts.VIJFH,
    enable_mask::Bool = false,
) where {FT}
    @assert ClimaComms.device(context) == device "The given device and context device do not match."
    return Grids.SpectralElementGrid2D(
        h_topology,
        quad;
        VIJH,
        enable_mask,
        discretization,
    )
end

"""
    ColumnGrid([FT = Float64]; z_elem, z_min, z_max, kwargs...)

Construct a [`Grids.FiniteDifferenceGrid`](@ref) for a single vertical column.

# Arguments

  - `FT`: The floating-point type, `Float32` or `Float64`.

# Keyword Arguments

  - `z_elem::Integer`: Number of vertical elements.
  - `z_min::Real`, `z_max::Real`: Vertical extent of the domain.
  - `device = ClimaComms.device()`: The `ClimaComms` device.
  - `context = ClimaComms.context(device)`: The `ClimaComms` context; it must be
    a `ClimaComms.SingletonCommsContext` whose device equals `device`.
  - `stretch = Meshes.Uniform()`: The `Meshes.StretchingRule` of the default
    mesh; see [`Meshes.Uniform`](@ref).
  - `z_mesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch)`: The
    `Meshes.IntervalMesh`, with boundaries named `:bottom` and `:top`.

# Examples

```julia
using ClimaCore.CommonGrids
grid = ColumnGrid(; z_elem = 10, z_min = 0, z_max = 10)
```
"""
ColumnGrid(; kwargs...) = ColumnGrid(Float64; kwargs...)
function ColumnGrid(
    ::Type{FT};
    z_elem::Integer,
    z_min::Real,
    z_max::Real,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
    z_mesh::Meshes.IntervalMesh = DefaultZMesh(
        FT;
        z_min,
        z_max,
        z_elem,
        stretch,
    ),
) where {FT}
    @assert ClimaComms.device(context) == device "The given device and context device do not match."
    @assert context isa ClimaComms.SingletonCommsContext "Columns can only be created on Singleton contextes."
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    return Grids.FiniteDifferenceGrid(z_topology)
end

"""
    Box3DGrid([FT = Float64]; z_elem, x_min, x_max, y_min, y_max, z_min, z_max,
              periodic_x, periodic_y, n_quad_points, x_elem, y_elem, kwargs...)

Construct a [`Grids.ExtrudedFiniteDifferenceGrid`](@ref) on a rectangular box: a
[`Grids.SpectralElementGrid2D`](@ref) horizontal grid on the `x`-`y` rectangle
extruded along a [`Grids.FiniteDifferenceGrid`](@ref) vertical grid.

# Arguments

  - `FT`: The floating-point type, `Float32` or `Float64`.

# Keyword Arguments

  - `z_elem::Integer`: Number of vertical elements.
  - `x_min::Real`, `x_max::Real`: Extent of the domain along `x`.
  - `y_min::Real`, `y_max::Real`: Extent of the domain along `y`.
  - `z_min::Real`, `z_max::Real`: Vertical extent of the domain.
  - `periodic_x::Bool`, `periodic_y::Bool`: Whether the domain is periodic along
    `x` and `y`. Non-periodic boundaries are named `:west`/`:east` and
    `:south`/`:north`.
  - `n_quad_points::Integer`: Number of quadrature points per element side.
  - `x_elem::Integer`, `y_elem::Integer`: Number of horizontal elements along `x`
    and `y`.
  - `device = ClimaComms.device()`: The `ClimaComms` device.
  - `context = ClimaComms.context(device)`: The `ClimaComms` context; its device
    must equal `device`.
  - `stretch = Meshes.Uniform()`: The `Meshes.StretchingRule` of the default
    vertical mesh; see [`Meshes.Uniform`](@ref).
  - `hypsography_fun = (h_grid, z_grid) -> Grids.Flat()`: A callable that returns
    the `Grids.HypsographyAdaption` for the given horizontal and vertical grids.
  - `global_geometry = Geometry.CartesianGlobalGeometry()`: The
    `Geometry.AbstractGlobalGeometry` of the extruded grid; see
    [`Geometry.CartesianGlobalGeometry`](@ref).
  - `quad = Quadratures.GLL{n_quad_points}()`: The `Quadratures.QuadratureStyle`.
  - `discretization = nothing`, `VIJH = DataLayouts.VIJFH`,
    `enable_bubble = false`, `enable_mask = false`: Passed to the horizontal
    [`Grids.SpectralElementGrid2D`](@ref), which documents them.
  - `h_topology`: The horizontal `Topologies.Topology2D`. It defaults to the
    topology of the `Meshes.RectilinearMesh` given by the `x` and `y` arguments,
    ordered along a space-filling curve.
  - `z_mesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch)`: The vertical
    `Meshes.IntervalMesh`, with boundaries named `:bottom` and `:top`.

# Examples

```julia
using ClimaCore.CommonGrids
grid = Box3DGrid(;
    z_elem = 10,
    x_min = 0,
    x_max = 1,
    y_min = 0,
    y_max = 1,
    z_min = 0,
    z_max = 10,
    periodic_x = false,
    periodic_y = false,
    n_quad_points = 4,
    x_elem = 3,
    y_elem = 4,
)
```
"""
Box3DGrid(; kwargs...) = Box3DGrid(Float64; kwargs...)
function Box3DGrid(
    ::Type{FT};
    z_elem::Integer,
    x_min::Real,
    x_max::Real,
    y_min::Real,
    y_max::Real,
    z_min::Real,
    z_max::Real,
    periodic_x::Bool,
    periodic_y::Bool,
    n_quad_points::Integer,
    x_elem::Integer,
    y_elem::Integer,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
    hypsography_fun = (h_grid, z_grid) -> Grids.Flat(),
    global_geometry::Geometry.AbstractGlobalGeometry = Geometry.CartesianGlobalGeometry(),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{n_quad_points}(),
    discretization::Union{Nothing, Grids.Discretization} = nothing,
    h_topology::Topologies.AbstractDistributedTopology = Topologies.Topology2D(
        context,
        DefaultRectangleXYMesh(
            FT;
            x_min,
            x_max,
            y_min,
            y_max,
            x_elem,
            y_elem,
            periodic_x,
            periodic_y,
        ),
        Topologies.spacefillingcurve(
            DefaultRectangleXYMesh(
                FT;
                x_min,
                x_max,
                y_min,
                y_max,
                x_elem,
                y_elem,
                periodic_x,
                periodic_y,
            ),
        ),
    ),
    z_mesh::Meshes.IntervalMesh = DefaultZMesh(
        FT;
        z_min,
        z_max,
        z_elem,
        stretch,
    ),
    enable_bubble::Bool = false,
    VIJH::Type{<:DataLayouts.VIJHWithF} = DataLayouts.VIJFH,
    enable_mask::Bool = false,
) where {FT}
    @assert ClimaComms.device(context) == device "The given device and context device do not match."
    h_grid = Grids.SpectralElementGrid2D(
        h_topology,
        quad;
        VIJH,
        enable_bubble,
        enable_mask,
        discretization,
    )
    z_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        z_mesh,
    )
    z_grid = Grids.FiniteDifferenceGrid(z_topology)
    return Grids.ExtrudedFiniteDifferenceGrid(
        h_grid,
        z_grid,
        hypsography_fun(h_grid, z_grid),
        global_geometry,
    )
end

"""
    SliceXZGrid([FT = Float64]; z_elem, x_min, x_max, z_min, z_max, periodic_x,
                n_quad_points, x_elem, kwargs...)

Construct a [`Grids.ExtrudedFiniteDifferenceGrid`](@ref) on an `x`-`z` slice: a
[`Grids.SpectralElementGrid1D`](@ref) horizontal grid along `x` extruded along
a [`Grids.FiniteDifferenceGrid`](@ref) vertical grid. The horizontal topology
is built on a `ClimaComms.SingletonCommsContext`, so the grid is not
distributed.

# Arguments

  - `FT`: The floating-point type, `Float32` or `Float64`.

# Keyword Arguments

  - `z_elem::Integer`: Number of vertical elements.
  - `x_min::Real`, `x_max::Real`: Extent of the domain along `x`.
  - `z_min::Real`, `z_max::Real`: Vertical extent of the domain.
  - `periodic_x::Bool`: Whether the domain is periodic along `x`. Non-periodic
    boundaries are named `:west` and `:east`.
  - `n_quad_points::Integer`: Number of quadrature points per element.
  - `x_elem::Integer`: Number of horizontal elements.
  - `device = ClimaComms.device()`: The `ClimaComms` device.
  - `context = ClimaComms.context(device)`: The `ClimaComms` context; its device
    must equal `device`.
  - `stretch = Meshes.Uniform()`: The `Meshes.StretchingRule` of the default
    vertical mesh; see [`Meshes.Uniform`](@ref).
  - `hypsography_fun = (h_grid, z_grid) -> Grids.Flat()`: A callable that returns
    the `Grids.HypsographyAdaption` for the given horizontal and vertical grids.
  - `global_geometry = Geometry.CartesianGlobalGeometry()`: The
    `Geometry.AbstractGlobalGeometry` of the extruded grid; see
    [`Geometry.CartesianGlobalGeometry`](@ref).
  - `quad = Quadratures.GLL{n_quad_points}()`: The `Quadratures.QuadratureStyle`.
  - `discretization = nothing`, `VIJH = DataLayouts.VIJFH`: Passed to the
    horizontal [`Grids.SpectralElementGrid1D`](@ref), which documents them.
  - `h_mesh = DefaultSliceXMesh(FT; x_min, x_max, periodic_x, x_elem)`: The
    horizontal `Meshes.IntervalMesh`.
  - `z_mesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch)`: The vertical
    `Meshes.IntervalMesh`, with boundaries named `:bottom` and `:top`.

# Examples

```julia
using ClimaCore.CommonGrids
grid = SliceXZGrid(;
    z_elem = 10,
    x_min = 0,
    x_max = 1,
    z_min = 0,
    z_max = 1,
    periodic_x = false,
    n_quad_points = 4,
    x_elem = 4,
)
```
"""
SliceXZGrid(; kwargs...) = SliceXZGrid(Float64; kwargs...)
function SliceXZGrid(
    ::Type{FT};
    z_elem::Integer,
    x_min::Real,
    x_max::Real,
    z_min::Real,
    z_max::Real,
    periodic_x::Bool,
    n_quad_points::Integer,
    x_elem::Integer,
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
    hypsography_fun = (h_grid, z_grid) -> Grids.Flat(),
    global_geometry::Geometry.AbstractGlobalGeometry = Geometry.CartesianGlobalGeometry(),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{n_quad_points}(),
    discretization::Union{Nothing, Grids.Discretization} = nothing,
    VIJH::Type{<:DataLayouts.VIJHWithF} = DataLayouts.VIJFH,
    h_mesh::Meshes.IntervalMesh = DefaultSliceXMesh(
        FT;
        x_min,
        x_max,
        periodic_x,
        x_elem,
    ),
    z_mesh::Meshes.IntervalMesh = DefaultZMesh(
        FT;
        z_min,
        z_max,
        z_elem,
        stretch,
    ),
) where {FT}
    @assert ClimaComms.device(context) == device "The given device and context device do not match."

    h_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        h_mesh,
    )
    h_grid =
        Grids.SpectralElementGrid1D(h_topology, quad; VIJH, discretization)
    z_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        z_mesh,
    )
    z_grid = Grids.FiniteDifferenceGrid(z_topology)
    return Grids.ExtrudedFiniteDifferenceGrid(
        h_grid,
        z_grid,
        hypsography_fun(h_grid, z_grid),
        global_geometry,
    )
end

"""
    RectangleXYGrid([FT = Float64]; x_min, x_max, y_min, y_max, periodic_x,
                    periodic_y, n_quad_points, x_elem, y_elem, kwargs...)

Construct a [`Grids.SpectralElementGrid2D`](@ref) on an `x`-`y` rectangle with
a `Meshes.RectilinearMesh`.

# Arguments

  - `FT`: The floating-point type, `Float32` or `Float64`.

# Keyword Arguments

  - `x_min::Real`, `x_max::Real`: Extent of the domain along `x`.
  - `y_min::Real`, `y_max::Real`: Extent of the domain along `y`.
  - `periodic_x::Bool`, `periodic_y::Bool`: Whether the domain is periodic along
    `x` and `y`. Non-periodic boundaries are named `:west`/`:east` and
    `:south`/`:north`.
  - `n_quad_points::Integer`: Number of quadrature points per element side.
  - `x_elem::Integer`, `y_elem::Integer`: Number of elements along `x` and `y`.
  - `device = ClimaComms.device()`: The `ClimaComms` device.
  - `context = ClimaComms.context(device)`: The `ClimaComms` context; its device
    must equal `device`.
  - `quad = Quadratures.GLL{n_quad_points}()`: The `Quadratures.QuadratureStyle`.
  - `discretization = nothing`, `VIJH = DataLayouts.VIJFH`,
    `enable_bubble = false`, `enable_mask = false`: Passed to
    [`Grids.SpectralElementGrid2D`](@ref), which documents them.
  - `h_topology`: The `Topologies.Topology2D`. It defaults to the topology of the
    `Meshes.RectilinearMesh` given by the `x` and `y` arguments.
  - `hypsography = Grids.Flat()`, `global_geometry = Geometry.CartesianGlobalGeometry()`:
    Accepted for interface compatibility with the extruded grids; a
    `Grids.SpectralElementGrid2D` has neither, so they do not affect the result.

# Examples

```julia
using ClimaCore.CommonGrids
grid = RectangleXYGrid(;
    x_min = 0,
    x_max = 1,
    y_min = 0,
    y_max = 1,
    periodic_x = false,
    periodic_y = false,
    n_quad_points = 4,
    x_elem = 3,
    y_elem = 4,
)
```
"""
RectangleXYGrid(; kwargs...) = RectangleXYGrid(Float64; kwargs...)
function RectangleXYGrid(
    ::Type{FT};
    x_min::Real,
    x_max::Real,
    y_min::Real,
    y_max::Real,
    periodic_x::Bool,
    periodic_y::Bool,
    n_quad_points::Integer,
    x_elem::Integer, # number of horizontal elements
    y_elem::Integer, # number of horizontal elements
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
    hypsography::Grids.HypsographyAdaption = Grids.Flat(),
    global_geometry::Geometry.AbstractGlobalGeometry = Geometry.CartesianGlobalGeometry(),
    quad::Quadratures.QuadratureStyle = Quadratures.GLL{n_quad_points}(),
    discretization::Union{Nothing, Grids.Discretization} = nothing,
    VIJH::Type{<:DataLayouts.VIJHWithF} = DataLayouts.VIJFH,
    h_topology::Topologies.AbstractDistributedTopology = Topologies.Topology2D(
        context,
        DefaultRectangleXYMesh(
            FT;
            x_min,
            x_max,
            y_min,
            y_max,
            x_elem,
            y_elem,
            periodic_x,
            periodic_y,
        ),
    ),
    enable_bubble::Bool = false,
    enable_mask::Bool = false,
) where {FT}
    @assert ClimaComms.device(context) == device "The given device and context device do not match."
    return Grids.SpectralElementGrid2D(
        h_topology,
        quad;
        VIJH,
        enable_bubble,
        enable_mask,
        discretization,
    )
end

"""
    MultiColumnGrid([FT = Float64]; points, z_elem, z_min, z_max, kwargs...)

Construct a [`Grids.ExtrudedFiniteDifferenceGrid`](@ref) of independent
vertical columns at given latitude-longitude locations on a sphere: a
[`Grids.MultiPointGrid`](@ref) horizontal grid extruded along a
[`Grids.FiniteDifferenceGrid`](@ref) vertical grid, with `Grids.Flat()`
hypsography and a shallow spherical global geometry.

The columns have no horizontal connectivity, so horizontal operators are not
defined on the grid; `Fields.bycolumn` iterates over the columns.

# Arguments

  - `FT`: The floating-point type, `Float32` or `Float64`.

# Keyword Arguments

  - `points::AbstractVector{Geometry.LatLongPoint{FT}}`: The location of each
    column.
  - `z_elem::Integer`: Number of vertical elements.
  - `z_min::Real`, `z_max::Real`: Vertical extent of the domain.
  - `radius::Real = FT(6.371229e6)`: Radius of the sphere.
  - `device = ClimaComms.device()`: The `ClimaComms` device.
  - `stretch = Meshes.Uniform()`: The `Meshes.StretchingRule` of the default
    vertical mesh; see [`Meshes.Uniform`](@ref).
  - `z_mesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch)`: The vertical
    `Meshes.IntervalMesh`, with boundaries named `:bottom` and `:top`.

# Examples

```julia
using ClimaCore.CommonGrids, ClimaCore.Geometry
points = [LatLongPoint(0.0, 0.0), LatLongPoint(10.0, 20.0), LatLongPoint(-5.0, 90.0)]
grid = MultiColumnGrid(;
    points = points,
    z_elem = 10,
    z_min = 0,
    z_max = 10_000,
    radius = 6.371229e6,
)
```
"""
MultiColumnGrid(; kwargs...) = MultiColumnGrid(Float64; kwargs...)
function MultiColumnGrid(
    ::Type{FT};
    points::AbstractVector{Geometry.LatLongPoint{FT}},
    z_elem::Integer,
    z_min::Real,
    z_max::Real,
    radius::Real = FT(6.371229e6),
    device::ClimaComms.AbstractDevice = ClimaComms.device(),
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
    z_mesh::Meshes.IntervalMesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch),
) where {FT}
    h_grid = Grids.MultiPointGrid(points; radius, device)
    z_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        z_mesh,
    )
    z_grid = Grids.FiniteDifferenceGrid(z_topology)
    return Grids.ExtrudedFiniteDifferenceGrid(h_grid, z_grid)
end

# Backwards-compatibility alias for the old name.
Base.@deprecate_binding PointColumnEnsembleGrid MultiColumnGrid false

end # module
