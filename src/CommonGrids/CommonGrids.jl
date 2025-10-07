"""
    CommonGrids

CommonGrids contains convenience constructors for common
grids. Constructors in this module are sometimes dynamically
created. You may want to use a different constructor if you're
making the object in a performance-critical section, and if
you know the type parameters at compile time.

If no convenience constructor exists, then you may need to
create a custom grid using our low-level compose-able API.


# Transitioning to using CommonGrids

You may have constructed a grid in the following way:

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
z_topology = Topologies.IntervalTopology(context, z_mesh)
z_grid = Grids.FiniteDifferenceGrid(z_topology)
grid = Grids.ExtrudedFiniteDifferenceGrid(
    h_grid,
    z_grid,
    hypsography,
    global_geometry,
)
```

You may re-write this as:

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
    CubedSphereGrid, ColumnGrid, Box3DGrid, SliceXZGrid, RectangleXYGrid

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
    ExtrudedCubedSphereGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
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
        global_geometry::Geometry.AbstractGlobalGeometry = Geometry.ShallowSphericalGlobalGeometry{FT}(radius),
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{n_quad_points}(),
        h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{FT}(radius), h_elem),
        h_topology::Topologies.AbstractDistributedTopology = Topologies.Topology2D(context, h_mesh),
        horizontal_layout_type = DataLayouts.IJFH,
        z_mesh::Meshes.IntervalMesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch),
        enable_bubble::Bool = false
        enable_mask::Bool = false
    )

A convenience constructor, which builds an
[`Grids.ExtrudedFiniteDifferenceGrid`](@ref), given:

 - `FT` the floating-point type (defaults to `Float64`) [`Float32`, `Float64`]
 - `z_elem` the number of z-points
 - `z_min` the domain minimum along the z-direction.
 - `z_max` the domain maximum along the z-direction.
 - `radius` the radius of the cubed sphere
 - `h_elem` the number of horizontal elements per side of every panel (6 panels in total)
 - `n_quad_points` the number of quadrature points per horizontal element
 - `device` the `ClimaComms.device`
 - `context` the `ClimaComms.context`
 - `stretch` the mesh `Meshes.StretchingRule` (defaults to [`Meshes.Uniform`](@ref))
 - `hypsography_fun` a function or callable object (`hypsography_fun(h_grid, z_grid) -> hypsography`) for constructing the hypsography model.
 - `global_geometry` the global geometry (defaults to [`Geometry.CartesianGlobalGeometry`](@ref))
 - `quad` the quadrature style (defaults to `Quadratures.GLL{n_quad_points}`)
 - `h_mesh` the horizontal mesh (defaults to `Meshes.EquiangularCubedSphere`)
 - `h_topology` the horizontal topology (defaults to `Topologies.Topology2D`)
 - `horizontal_layout_type` the horizontal DataLayout type (defaults to `DataLayouts.IJFH`). This parameter describes how data is arranged in memory. See [`Grids.SpectralElementGrid2D`](@ref) for its use.
 - `z_mesh` the vertical mesh, defaults to an `Meshes.IntervalMesh` along `z` with given `stretch`
 - `enable_bubble` enables the "bubble correction" for more accurate element areas when computing the spectral element space. See [`Grids.SpectralElementGrid2D`](@ref) for more information.
 - `enable_mask` enables a horizontal mask, for skipping operations on specified
                 columns via `set_mask!`.

# Example usage

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
    h_mesh = Meshes.EquiangularCubedSphere(
        Domains.SphereDomain{FT}(radius),
        h_elem,
    ),
    h_topology::Topologies.AbstractDistributedTopology = Topologies.Topology2D(
        context,
        h_mesh,
    ),
    horizontal_layout_type = DataLayouts.IJFH,
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
    @assert horizontal_layout_type <: DataLayouts.AbstractData
    @assert ClimaComms.device(context) == device "The given device and context device do not match."

    z_boundary_names = (:bottom, :top)
    h_grid = Grids.SpectralElementGrid2D(
        h_topology,
        quad;
        horizontal_layout_type,
        enable_bubble,
        enable_mask,
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
    CubedSphereGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        radius::Real,
        h_elem::Integer,
        n_quad_points::Integer,
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
        context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{n_quad_points}(),
        h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{FT}(radius), h_elem),
        h_topology::Topologies.AbstractDistributedTopology = Topologies.Topology2D(context, h_mesh),
        horizontal_layout_type = DataLayouts.IJFH,
        enable_mask = false,
    )

A convenience constructor, which builds a
[`Grids.SpectralElementGrid2D`](@ref) given:

 - `FT` the floating-point type (defaults to `Float64`) [`Float32`, `Float64`]
 - `radius` the radius of the cubed sphere
 - `h_elem` the number of horizontal elements per side of every panel (6 panels in total)
 - `n_quad_points` the number of quadrature points per horizontal element
 - `device` the `ClimaComms.device`
 - `context` the `ClimaComms.context`
 - `quad` the quadrature style (defaults to `Quadratures.GLL{n_quad_points}`)
 - `h_mesh` the horizontal mesh (defaults to `Meshes.EquiangularCubedSphere`)
 - `h_topology` the horizontal topology (defaults to `Topologies.Topology2D`)
 - `horizontal_layout_type` the horizontal DataLayout type (defaults to `DataLayouts.IJFH`). This parameter describes how data is arranged in memory. See [`Grids.SpectralElementGrid2D`](@ref) for its use.
 - `enable_mask` enables a horizontal mask, for skipping operations on specified
                 columns via `set_mask!`.

# Example usage

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
    h_mesh = Meshes.EquiangularCubedSphere(
        Domains.SphereDomain{FT}(radius),
        h_elem,
    ),
    h_topology::Topologies.AbstractDistributedTopology = Topologies.Topology2D(
        context,
        h_mesh,
    ),
    horizontal_layout_type = DataLayouts.IJFH,
    enable_mask::Bool = false,
) where {FT}
    @assert horizontal_layout_type <: DataLayouts.AbstractData
    @assert ClimaComms.device(context) == device "The given device and context device do not match."
    return Grids.SpectralElementGrid2D(
        h_topology,
        quad;
        horizontal_layout_type,
        enable_mask,
    )
end

"""
    ColumnGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_elem::Integer,
        z_min::Real,
        z_max::Real,
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
        context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
        stretch::Meshes.StretchingRule = Meshes.Uniform(),
        z_mesh::Meshes.IntervalMesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch),
    )

A convenience constructor, which builds a
[`Grids.FiniteDifferenceGrid`](@ref) given:

 - `FT` the floating-point type (defaults to `Float64`) [`Float32`, `Float64`]
 - `z_elem` the number of z-points
 - `z_min` the domain minimum along the z-direction.
 - `z_max` the domain maximum along the z-direction.
 - `device` the `ClimaComms.device`
 - `context` the `ClimaComms.context`
 - `stretch` the mesh `Meshes.StretchingRule` (defaults to [`Meshes.Uniform`](@ref))
 - `z_mesh` the vertical mesh, defaults to an `Meshes.IntervalMesh` along `z` with given `stretch`

# Example usage

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
    Box3DGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
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
        horizontal_layout_type = DataLayouts.IJFH,
        [h_topology::Topologies.AbstractDistributedTopology], # optional
        [z_mesh::Meshes.IntervalMesh], # optional
        enable_bubble::Bool = false,
        enable_mask::Bool = false,
    )

A convenience constructor, which builds a
[`Grids.ExtrudedFiniteDifferenceGrid`](@ref) with a
[`Grids.FiniteDifferenceGrid`](@ref) vertical grid and a
[`Grids.SpectralElementGrid2D`](@ref) horizontal grid, given:

 - `z_elem` the number of z-points
 - `x_min` the domain minimum along the x-direction.
 - `x_max` the domain maximum along the x-direction.
 - `y_min` the domain minimum along the y-direction.
 - `y_max` the domain maximum along the y-direction.
 - `z_min` the domain minimum along the z-direction.
 - `z_max` the domain maximum along the z-direction.
 - `periodic_x` Bool indicating to use periodic domain along x-direction
 - `periodic_y` Bool indicating to use periodic domain along y-direction
 - `n_quad_points` the number of quadrature points per horizontal element
 - `x_elem` the number of x-points
 - `y_elem` the number of y-points
 - `device` the `ClimaComms.device`
 - `context` the `ClimaComms.context`
 - `stretch` the mesh `Meshes.StretchingRule` (defaults to [`Meshes.Uniform`](@ref))
 - `hypsography_fun` a function or callable object (`hypsography_fun(h_grid, z_grid) -> hypsography`) for constructing the hypsography model.
 - `global_geometry` the global geometry (defaults to [`Geometry.CartesianGlobalGeometry`](@ref))
 - `quad` the quadrature style (defaults to `Quadratures.GLL{n_quad_points}`)
 - `h_topology` the horizontal topology (defaults to `Topologies.Topology2D`)
 - `z_mesh` the vertical mesh, defaults to an `Meshes.IntervalMesh` along `z` with given `stretch`
 - `enable_bubble` enables the "bubble correction" for more accurate element areas when computing the spectral element space. See [`Grids.SpectralElementGrid2D`](@ref) for more information.
 - `horizontal_layout_type` the horizontal DataLayout type (defaults to `DataLayouts.IJFH`). This parameter describes how data is arranged in memory. See [`Grids.SpectralElementGrid2D`](@ref) for its use.
 - `enable_mask` enables a horizontal mask, for skipping operations on specified
                 columns via `set_mask!`.

# Example usage

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
    z_mesh::Meshes.IntervalMesh = DefaultZMesh(
        FT;
        z_min,
        z_max,
        z_elem,
        stretch,
    ),
    enable_bubble::Bool = false,
    horizontal_layout_type = DataLayouts.IJFH,
    enable_mask::Bool = false,
) where {FT}
    @assert horizontal_layout_type <: DataLayouts.AbstractData
    @assert ClimaComms.device(context) == device "The given device and context device do not match."
    h_grid = Grids.SpectralElementGrid2D(
        h_topology,
        quad;
        horizontal_layout_type,
        enable_bubble,
        enable_mask,
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
    SliceXZGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
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
    )

A convenience constructor, which builds a
[`Grids.ExtrudedFiniteDifferenceGrid`](@ref) with a
[`Grids.FiniteDifferenceGrid`](@ref) vertical grid and a
[`Grids.SpectralElementGrid1D`](@ref) horizontal grid, given:


 - `FT` the floating-point type (defaults to `Float64`) [`Float32`, `Float64`]
 - `z_elem` the number of z-points
 - `x_min` the domain minimum along the x-direction.
 - `x_max` the domain maximum along the x-direction.
 - `z_min` the domain minimum along the z-direction.
 - `z_max` the domain maximum along the z-direction.
 - `periodic_x` Bool indicating to use periodic domain along x-direction
 - `n_quad_points` the number of quadrature points per horizontal element
 - `x_elem` the number of x-points
 - `device` the `ClimaComms.device`
 - `context` the `ClimaComms.context`
 - `stretch` the mesh `Meshes.StretchingRule` (defaults to [`Meshes.Uniform`](@ref))
 - `hypsography_fun` a function or callable object (`hypsography_fun(h_grid, z_grid) -> hypsography`) for constructing the hypsography model.
 - `global_geometry` the global geometry (defaults to [`Geometry.CartesianGlobalGeometry`](@ref))
 - `quad` the quadrature style (defaults to `Quadratures.GLL{n_quad_points}`)

# Example usage

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
    horizontal_layout_type = DataLayouts.IFH,
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
    @assert horizontal_layout_type <: DataLayouts.AbstractData
    @assert ClimaComms.device(context) == device "The given device and context device do not match."

    h_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        h_mesh,
    )
    h_grid =
        Grids.SpectralElementGrid1D(h_topology, quad; horizontal_layout_type)
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
    RectangleXYGrid(
        ::Type{<:AbstractFloat}; # defaults to Float64
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
        enable_mask::Bool = false,
    )

A convenience constructor, which builds a
[`Grids.SpectralElementGrid2D`](@ref) with a horizontal
`RectilinearMesh` mesh, given:

 - `x_min` the domain minimum along the x-direction.
 - `x_max` the domain maximum along the x-direction.
 - `y_min` the domain minimum along the y-direction.
 - `y_max` the domain maximum along the y-direction.
 - `periodic_x` Bool indicating to use periodic domain along x-direction
 - `periodic_y` Bool indicating to use periodic domain along y-direction
 - `n_quad_points` the number of quadrature points per horizontal element
 - `x_elem` the number of x-points
 - `y_elem` the number of y-points
 - `device` the `ClimaComms.device`
 - `context` the `ClimaComms.context`
 - `hypsography_fun` a function or callable object (`hypsography_fun(h_grid, z_grid) -> hypsography`) for constructing the hypsography model.
 - `global_geometry` the global geometry (defaults to [`Geometry.CartesianGlobalGeometry`](@ref))
 - `quad` the quadrature style (defaults to `Quadratures.GLL{n_quad_points}`)
 - `enable_mask` enables a horizontal mask, for skipping operations on specified
                 columns via `set_mask!`.

# Example usage

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
    horizontal_layout_type = DataLayouts.IJFH,
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
    @assert horizontal_layout_type <: DataLayouts.AbstractData
    @assert ClimaComms.device(context) == device "The given device and context device do not match."
    return Grids.SpectralElementGrid2D(
        h_topology,
        quad;
        horizontal_layout_type,
        enable_bubble,
        enable_mask,
    )
end

end # module
