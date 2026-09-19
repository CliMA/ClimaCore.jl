"""
    Spaces

Function spaces on which fields are defined. A space combines a grid (domain,
topology, coordinates, metric terms, and quadrature rules and weights) with a
vertical staggering.

# Notes

References: [CEED](https://ceed.exascaleproject.org/ceed-code/) and the
[ClimateMachine sphere helpers](https://github.com/CliMA/ClimateMachine.jl/blob/ans/sphere/test/Numerics/DGMethods/compressible_navier_stokes_equations/sphere/sphere_helper_functions.jl).
"""
module Spaces

using ClimaComms
using Adapt

import ..slab, ..column, ..level
import ..Utilities: PlusHalf, half
import ..DebugOnly: call_post_op_callback, post_op_callback
import ..DataLayouts, ..Geometry, ..Domains, ..Meshes, ..Topologies, ..Grids, ..Quadratures
import ..DataLayouts: PointIndex

import ..Domains: z_max, z_min
import ..Meshes: n_elements_per_panel_direction

import ..Grids:
    Staggering,
    CellFace,
    CellCenter,
    Discretization,
    CG,
    DG,
    topology,
    vertical_topology,
    discretization,
    is_continuous,
    local_geometry_type,
    local_geometry_data,
    global_geometry,
    dss_weights,
    issubgrid,
    set_mask!,
    get_mask,
    quadrature_style,
    hypsography

import ClimaComms
using StaticArrays, ForwardDiff, LinearAlgebra, Adapt

"""
    AbstractSpace

Abstract supertype of spaces. Subtypes define `grid(space)`, `staggering(space)`,
and the constructor `space(grid, staggering)`.
"""
abstract type AbstractSpace end

"""
    Spaces.grid(space::AbstractSpace)

Return the `Grids.AbstractGrid` underlying `space`: the domain, topology,
coordinates, metric terms, and quadrature, without the vertical staggering.
"""
function grid end

"""
    Spaces.staggering(space::AbstractSpace)

Return the vertical staggering of `space`: [`Grids.CellCenter`](@ref) or
[`Grids.CellFace`](@ref) for spaces with a vertical direction, and `nothing`
for purely horizontal spaces.
"""
function staggering end

"""
    Spaces.horizontal_space(space::AbstractSpace)

Return the horizontal space of `space`: `space` itself for a spectral element
space, the wrapped horizontal space of an
[`Spaces.ExtrudedFiniteDifferenceSpace`](@ref) or
[`Spaces.MultiColumnFiniteDifferenceSpace`](@ref), and the first level of a
[`Spaces.FiniteDifferenceSpace`](@ref) (a [`Spaces.PointSpace`](@ref)).
"""
function horizontal_space end

"""
    Spaces.eachslabindex(space::AbstractSpace)

Return an iterator over the indices of the slabs (single elements at a single
level) of `space`: element indices `h` for a spectral element space, and
`(v, h)` tuples of level and element indices for an
[`Spaces.ExtrudedFiniteDifferenceSpace`](@ref). Each index can be passed to
`slab(space, index...)`.
"""
function eachslabindex end

ClimaComms.context(space::AbstractSpace) = ClimaComms.context(grid(space))
ClimaComms.device(space::AbstractSpace) = ClimaComms.device(grid(space))

topology(space::AbstractSpace) = topology(grid(space))
vertical_topology(space::AbstractSpace) = vertical_topology(grid(space))

discretization(space::AbstractSpace) = discretization(grid(space))
is_continuous(space::AbstractSpace) = is_continuous(grid(space))


local_geometry_data(space::AbstractSpace) =
    local_geometry_data(grid(space), staggering(space))
dss_weights(space::AbstractSpace) = dss_weights(grid(space), staggering(space))

function n_elements_per_panel_direction(space::AbstractSpace)
    hspace = horizontal_space(space)
    hmesh = topology(hspace).mesh
    return Meshes.n_elements_per_panel_direction(hmesh)
end

global_geometry(space::AbstractSpace) = global_geometry(grid(space))

"""
    Spaces.radius(space::AbstractSpace)

Return the radius [m] of the sphere on which `space` is defined, read from the
`radius` of its global geometry (`Spaces.global_geometry(space)`). Throws an
`ArgumentError` if the global geometry is not spherical, e.g. for spaces on
planar or interval domains.
"""
function radius(space::AbstractSpace)
    gg = global_geometry(space)
    gg isa Geometry.AbstractSphericalGlobalGeometry || throw(
        ArgumentError(
            "the global geometry of the space is $(nameof(typeof(gg))), which \
             is not spherical and so has no radius",
        ),
    )
    return gg.radius
end

space(refspace::AbstractSpace, staggering::Staggering) =
    space(grid(refspace), staggering)

"""
    Spaces.issubspace(subspace::AbstractSpace, space::AbstractSpace)

Return `true` if fields on `subspace` can be broadcast against fields on
`space`: `subspace` is `space` itself, the horizontal space or a level of an
extruded `space`, or the vertical space or a column of it. Two spaces built on
the same grid with different staggering are not subspaces of each other.
"""
issubspace(subspace::AbstractSpace, space::AbstractSpace) =
    issubgrid(grid(subspace), grid(space)) &&
    (isnothing(staggering(subspace)) || staggering(subspace) == staggering(space))

"""
    Spaces.undertype(space::AbstractSpace)

Return the underlying floating-point type of `space`, i.e. the number type of
the coordinates and metric terms of its local geometry.
"""
undertype(space::AbstractSpace) =
    Geometry.undertype(eltype(local_geometry_data(space)))

"""
    Spaces.coordinates_data(space::AbstractSpace)
    Spaces.coordinates_data(grid::Grids.AbstractGrid)
    Spaces.coordinates_data(staggering, grid::Grids.AbstractGrid)

Return the `DataLayout` of coordinates of `space` (or of `grid` at the given
`staggering`): the `coordinates` of its local geometry.
"""
coordinates_data(space::AbstractSpace) = local_geometry_data(space).coordinates
coordinates_data(grid::Grids.AbstractGrid) =
    local_geometry_data(grid).coordinates
coordinates_data(staggering, grid::Grids.AbstractGrid) =
    local_geometry_data(staggering, grid).coordinates

"""
    Spaces.horizontal_grid(grid::Grids.AbstractGrid)

Return the horizontal grid underlying `grid`: a spectral element grid is its own
horizontal grid, and a `Grids.LevelGrid` returns the horizontal grid of the
extruded grid it is a level of.
"""
horizontal_grid(grid::Grids.AbstractSpectralElementGrid) = grid
horizontal_grid(grid::Grids.LevelGrid) = grid.full_grid.horizontal_grid

"""
    Spaces.vertical_grid(grid::Grids.AbstractGrid)

Return the vertical (finite difference) grid underlying `grid`: a finite
difference grid is its own vertical grid, an extruded grid returns its
`vertical_grid`, and a `Grids.ColumnGrid` returns the vertical grid of the
extruded grid it is a column of.
"""
vertical_grid(grid::Grids.AbstractFiniteDifferenceGrid) = grid
vertical_grid(grid::Grids.ColumnGrid) = vertical_grid(grid.full_grid)
vertical_grid(grid::Grids.AbstractExtrudedFiniteDifferenceGrid) = grid.vertical_grid

half_level_error() = throw(ArgumentError("Cannot use PlusHalf as CellCenter space index"))

staggered_level_index(space, v::Integer) = staggering(space) isa CellFace ? v - half : v
staggered_level_index(space, v::PlusHalf) =
    staggering(space) isa CellFace ? v : half_level_error()

integer_level_index(_, v::Integer) = v
integer_level_index(space, v::PlusHalf) =
    staggering(space) isa CellFace ? v + half : half_level_error()

Base.@propagate_inbounds Base.view(space::AbstractSpace, index::PointIndex) =
    PointSpace(ClimaComms.context(space), view(local_geometry_data(space), index))
Base.@propagate_inbounds Base.view(space::AbstractSpace, indices::PointIndex...) =
    view(space, CartesianIndex(indices...))

include("pointspace.jl")
include("spectralelement.jl")
include("finitedifference.jl")
include("extruded.jl")
include("multicolumn.jl")
include("triangulation.jl")
include("dss.jl")

# Resolve method ambiguities for issubspace.
issubspace(::PointSpace, ::SpectralElementSpaceSlab) = false
issubspace(::SpectralElementSpaceSlab, ::PointSpace) = false

function center_space(space::AbstractSpace)
    error("`center_space` can only be called with vertical/extruded spaces")
end

function face_space(space::AbstractSpace)
    error("`center_space` can only be called with vertical/extruded spaces")
end

weighted_jacobian(space::AbstractSpace) = local_geometry_data(space).WJ

"""
    Spaces.local_area(space::Spaces.AbstractSpace)

Return the length, area, or volume of the part of `space` local to the current
process. See [`Spaces.area`](@ref).
"""
local_area(space::AbstractSpace) = Base.sum(weighted_jacobian(space))

"""
    Spaces.area(space::Spaces.AbstractSpace)

Return the length, area, or volume of `space`, computed as the sum of the quadrature
weights ``W_i`` multiplied by the Jacobian determinants ``J_i``:

```math
\\sum_i W_i J_i \\approx \\int_\\Omega \\, d \\Omega
```

If `space` is distributed, this uses a `ClimaComms.allreduce` operation.
"""
area(space::AbstractSpace) =
    ClimaComms.allreduce(ClimaComms.context(space), local_area(space), +)

ClimaComms.array_type(space::AbstractSpace) =
    ClimaComms.array_type(ClimaComms.device(space))

"""
    z_max(::AbstractSpace)

Return the maximum `z` coordinate of the vertical domain of `space`.
"""
function z_max(space::AbstractSpace)
    mesh = Topologies.mesh(vertical_topology(space))
    domain = Topologies.domain(mesh)
    return Domains.z_max(domain)
end

"""
    z_min(::AbstractSpace)

Return the minimum `z` coordinate of the vertical domain of `space`.
"""
function z_min(space::AbstractSpace)
    mesh = Topologies.mesh(vertical_topology(space))
    domain = Topologies.domain(mesh)
    return Domains.z_min(domain)
end

"""
    nlevels(space::AbstractSpace)

Return the number of vertical levels of `space` at its staggering: the number of
cell centers or cell faces for a staggered space, and 1 for a horizontal space.
"""
function nlevels end

"""
    ncolumns(space::AbstractSpace)

Return the number of columns of `space` on the local process.
"""
ncolumns(space::ExtrudedFiniteDifferenceSpace) =
    ncolumns(horizontal_space(space))

function ncolumns(space::SpectralElementSpace1D)
    Nh = Topologies.nlocalelems(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style(space))
    return Nh * Nq
end
function ncolumns(space::SpectralElementSpace2D)
    Nh = Topologies.nlocalelems(space)
    Nq = Quadratures.degrees_of_freedom(quadrature_style(space))
    return Nh * Nq * Nq
end

get_mask(space::AbstractSpace) = get_mask(grid(space))
get_mask(space::PointSpace) = DataLayouts.NoMask()
get_mask(space::SpectralElementSpaceSlab) = DataLayouts.NoMask()
get_mask(space::ExtrudedFiniteDifferenceSpace) =
    get_mask(horizontal_space(space))

"""
    has_vertical(::AbstractSpace)

Return `true` if the space has a vertical grid.
"""
function has_vertical end
has_vertical(::AbstractSpace) = false
has_vertical(::ExtrudedFiniteDifferenceSpace) = true
has_vertical(::MultiColumnFiniteDifferenceSpace) = true
has_vertical(::FiniteDifferenceSpace) = true

"""
    has_horizontal(::AbstractSpace)

Return `true` if the space has a horizontal grid.
"""
function has_horizontal end
has_horizontal(::AbstractSpace) = false
has_horizontal(::ExtrudedFiniteDifferenceSpace) = true
has_horizontal(::SpectralElementSpace1D) = true
has_horizontal(::SpectralElementSpace2D) = true

set_mask!(fn, space::AbstractSpace) = set_mask!(fn, grid(space))
set_mask!(fn, space::ExtrudedFiniteDifferenceSpace) =
    set_mask!(fn, grid(horizontal_space(space)))
set_mask!(space::AbstractSpace, data::DataLayouts.DataLayout) =
    set_mask!(grid(space), data)

end # module
