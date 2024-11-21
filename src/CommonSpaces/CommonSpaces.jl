"""
    CommonSpaces

CommonSpaces contains convenience constructors for common spaces, which builds
off of [`CommonGrids`](@ref) and(when appropriate) requires an additional
argument, `staggering::Staggering` to construct the desired space.
"""
module CommonSpaces

export ExtrudedCubedSphereSpace,
    CubedSphereSpace, ColumnSpace, Box3DSpace, SliceXZSpace, RectangleXYSpace

export Grids
import ClimaComms

import ..DataLayouts,
    ..Meshes, ..Topologies, ..Geometry, ..Domains, ..Quadratures, ..Grids

import ..Grids: Staggering
import ..Spaces
import ..CommonGrids
import ..CommonGrids:
    ExtrudedCubedSphereGrid,
    CubedSphereGrid,
    ColumnGrid,
    Box3DGrid,
    SliceXZGrid,
    RectangleXYGrid

"""
	ExtrudedCubedSphereSpace(
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
        global_geometry::Geometry.AbstractGlobalGeometry = Geometry.ShallowSphericalGlobalGeometry(radius),
        quad::Quadratures.QuadratureStyle = Quadratures.GLL{n_quad_points}(),
        h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{FT}(radius), h_elem),
        h_topology::Topologies.AbstractDistributedTopology = Topologies.Topology2D(context, h_mesh),
        horizontal_layout_type = DataLayouts.IJFH,
        z_mesh::Meshes.IntervalMesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch),
        enable_bubble::Bool = false
        staggering::Staggering,
    )

Construct an [`Spaces.ExtrudedFiniteDifferenceSpace`](@ref) for a
cubed sphere configuration, given:

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
 - `staggering` vertical staggering, can be one of [[`Grids.CellFace`](@ref), [`Grids.CellCenter`](@ref)]

Note that these arguments are all the same as
[`CommonGrids.ExtrudedCubedSphereGrid`](@ref), except for `staggering`.

# Example usage

```julia
using ClimaCore.CommonSpaces
space = ExtrudedCubedSphereSpace(;
    z_elem = 10,
    z_min = 0,
    z_max = 1,
    radius = 10,
    h_elem = 10,
    n_quad_points = 4,
    staggering = Grids.CellCenter()
)
```
"""
function ExtrudedCubedSphereSpace end

ExtrudedCubedSphereSpace(; kwargs...) =
    ExtrudedCubedSphereSpace(Float64; kwargs...)
ExtrudedCubedSphereSpace(
    ::Type{FT};
    staggering::Staggering,
    kwargs...,
) where {FT} = Spaces.ExtrudedFiniteDifferenceSpace(
    ExtrudedCubedSphereGrid(FT; kwargs...),
    staggering,
)

"""
	CubedSphereSpace(
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
    )

Construct a [`Spaces.SpectralElementSpace2D`](@ref) for a
cubed sphere configuration, given:

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

Note that these arguments are all the same as [`CommonGrids.CubedSphereGrid`](@ref).

# Example usage

```julia
using ClimaCore.CommonSpaces
space = CubedSphereSpace(;
    radius = 10,
    n_quad_points = 4,
    h_elem = 10,
)
```
"""
function CubedSphereSpace end
CubedSphereSpace(; kwargs...) = CubedSphereSpace(Float64; kwargs...)
CubedSphereSpace(::Type{FT}; kwargs...) where {FT} =
    Spaces.SpectralElementSpace2D(CubedSphereGrid(FT; kwargs...))

"""
	ColumnSpace(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_elem::Integer,
        z_min::Real,
        z_max::Real,
        device::ClimaComms.AbstractDevice = ClimaComms.device(),
        context::ClimaComms.AbstractCommsContext = ClimaComms.context(device),
        stretch::Meshes.StretchingRule = Meshes.Uniform(),
        z_mesh::Meshes.IntervalMesh = DefaultZMesh(FT; z_min, z_max, z_elem, stretch),
    )

Construct a 1D [`Spaces.FiniteDifferenceSpace`](@ref) for a
column configuration, given:

 - `FT` the floating-point type (defaults to `Float64`) [`Float32`, `Float64`]
 - `z_elem` the number of z-points
 - `z_min` the domain minimum along the z-direction.
 - `z_max` the domain maximum along the z-direction.
 - `device` the `ClimaComms.device`
 - `context` the `ClimaComms.context`
 - `stretch` the mesh `Meshes.StretchingRule` (defaults to [`Meshes.Uniform`](@ref))
 - `z_mesh` the vertical mesh, defaults to an `Meshes.IntervalMesh` along `z` with given `stretch`
 - `staggering` vertical staggering, can be one of [[`Grids.CellFace`](@ref), [`Grids.CellCenter`](@ref)]

Note that these arguments are all the  same as [`CommonGrids.ColumnGrid`]
(@ref), except for `staggering`.

# Example usage

```julia
using ClimaCore.CommonSpaces
space = ColumnSpace(;
    z_elem = 10,
    z_min = 0,
    z_max = 10,
    staggering = Grids.CellCenter()
)
```
"""
function ColumnSpace end
ColumnSpace(; kwargs...) = ColumnSpace(Float64; kwargs...)
ColumnSpace(::Type{FT}; staggering::Staggering, kwargs...) where {FT} =
    Spaces.FiniteDifferenceSpace(ColumnGrid(FT; kwargs...), staggering)

"""
	Box3DSpace(
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
        staggering::Staggering
    )

Construct a [`Spaces.ExtrudedFiniteDifferenceSpace`](@ref) for a 3D box
configuration, given:

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
 - `staggering` vertical staggering, can be one of [[`Grids.CellFace`](@ref), [`Grids.CellCenter`](@ref)]

Note that these arguments are all  the same as [`CommonGrids.Box3DGrid`]
(@ref), except for `staggering`.

# Example usage

```julia
using ClimaCore.CommonSpaces
space = Box3DSpace(;
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
    staggering = Grids.CellCenter()
)
```
"""
function Box3DSpace end
Box3DSpace(; kwargs...) = Box3DSpace(Float64; kwargs...)
Box3DSpace(::Type{FT}; staggering::Staggering, kwargs...) where {FT} =
    Spaces.ExtrudedFiniteDifferenceSpace(Box3DGrid(FT; kwargs...), staggering)

"""
	SliceXZSpace(
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
        staggering::Staggering
    )

Construct a [`Spaces.ExtrudedFiniteDifferenceSpace`](@ref) for a 2D slice
configuration, given:

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
 - `staggering` vertical staggering, can be one of [[`Grids.CellFace`](@ref), [`Grids.CellCenter`](@ref)]

Note that these arguments are all the same
as [`CommonGrids.SliceXZGrid`](@ref),
except for `staggering`.

# Example usage

```julia
using ClimaCore.CommonSpaces
space = SliceXZSpace(;
    z_elem = 10,
    x_min = 0,
    x_max = 1,
    z_min = 0,
    z_max = 1,
    periodic_x = false,
    n_quad_points = 4,
    x_elem = 4,
    staggering = Grids.CellCenter()
)
```
"""
function SliceXZSpace end
SliceXZSpace(; kwargs...) = SliceXZSpace(Float64; kwargs...)
SliceXZSpace(::Type{FT}; staggering::Staggering, kwargs...) where {FT} =
    Spaces.ExtrudedFiniteDifferenceSpace(SliceXZGrid(FT; kwargs...), staggering)

"""
	RectangleXYSpace(
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
    )

Construct a [`Spaces.SpectralElementSpace2D`](@ref) space for a 2D rectangular
configuration, given:

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

Note that these arguments are all the same as [`CommonGrids.RectangleXYGrid`]
(@ref), except for `staggering`.

# Example usage

```julia
using ClimaCore.CommonSpaces
space = RectangleXYSpace(;
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
function RectangleXYSpace end
RectangleXYSpace(; kwargs...) = RectangleXYSpace(Float64; kwargs...)
RectangleXYSpace(::Type{FT}; kwargs...) where {FT} =
    Spaces.SpectralElementSpace2D(RectangleXYGrid(FT; kwargs...))

end # module
