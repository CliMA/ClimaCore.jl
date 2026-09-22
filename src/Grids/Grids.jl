module Grids

import ClimaComms, Adapt, ForwardDiff, LinearAlgebra
import LinearAlgebra: det, norm, I
import ..DataLayouts, ..Domains, ..Meshes, ..Topologies, ..Geometry, ..Quadratures
import ..Utilities: PlusHalf, half, Cache
import ..slab, ..column, ..level
import ..DeviceSideDevice, ..DeviceSideContext

using StaticArrays

"""
    Grids.AbstractGrid

Abstract supertype of grids. Subtypes define the following methods:

  - [`topology`](@ref): the topology of the grid.
  - `ClimaComms.context` and `ClimaComms.device` (default to those of the topology).
  - `Meshes.domain` (defaults to that of the topology).
  - [`local_geometry_data`](@ref): the `DataLayout` object containing the local
    geometry of the grid.
"""
abstract type AbstractGrid end

"""
    Grids.topology(grid::AbstractGrid)

Return the topology of `grid`.
"""
function topology end

"""
    Grids.local_geometry_data(
        grid       :: AbstractGrid,
        staggering :: Union{Staggering, Nothing},
    )

Return the `DataLayout` object containing the local geometry of `grid` at the
given `staggering`.

If the grid is not staggered, `staggering` is `nothing`.
"""
function local_geometry_data end

"""
    Grids.local_geometry_type(::Type{<:AbstractGrid})

Return the `LocalGeometry` element type of a grid type. The fallback for
unrecognized types is `Union{}`.
"""
function local_geometry_type end

# Fallback, but this requires user error-handling
local_geometry_type(::Type{T}) where {T} = Union{}

"""
    @host_device_struct struct Name{P...} <: Super
        field::T
        ...
    end

Define a grid as two structs with the same fields and type parameters: a
`mutable struct HostName`, which a space refers to by pointer and
an immutable `struct DeviceName`, which is what kernels receive (see
[`device_twin`](@ref)). `Name` is a `Union` alias of the two, so dispatch,
aliases and constructors written against `Name` accept both, `Name(args...)`
constructs the host grid, and `Name{P...}` in an alias covers both twins.
Adapting either twin with a host adaptor keeps its kind.
"""
macro host_device_struct(ex)
    ex isa Expr && ex.head === :struct ||
        error("@host_device_struct expects a struct definition")
    ex.args[1] === false ||
        error("@host_device_struct expects an immutable struct definition")
    header, body = ex.args[2], ex.args[3]
    if header isa Expr && header.head === :(<:)
        name_params, super = header.args
    else
        name_params, super = header, nothing
    end
    if name_params isa Symbol
        name, params = name_params, Any[]
    else
        name, params = name_params.args[1], name_params.args[2:end]
    end
    param_names = map(p -> p isa Symbol ? p : p.args[1], params)
    fields = filter(a -> !(a isa LineNumberNode), body.args)
    field_names = map(f -> f isa Symbol ? f : f.args[1], fields)
    host, device = Symbol(:Host, name), Symbol(:Device, name)
    function struct_def(struct_name, mutable)
        struct_header = Expr(:curly, struct_name, params...)
        isnothing(super) || (struct_header = Expr(:(<:), struct_header, super))
        return Expr(:struct, mutable, struct_header, Expr(:block, fields...))
    end
    twin_type(struct_name) = Expr(:curly, struct_name, param_names...)
    adapted = [:(Adapt.adapt(to, getfield(grid, $(QuoteNode(f))))) for f in field_names]
    return esc(
        quote
            $(struct_def(host, true))
            $(struct_def(device, false))
            Core.@__doc__ const $(Expr(:curly, name, params...)) =
                Union{$(twin_type(host)), $(twin_type(device))}
            (::Type{$name})(args...) = $host(args...)
            Adapt.adapt_structure(to, grid::$host) = $host($(adapted...))
            Adapt.adapt_structure(to, grid::$device) = $device($(adapted...))
            device_twin(to, grid::$host) = $device($(adapted...))
            public_name(::$name) = $(QuoteNode(name))
        end,
    )
end

"""
    Grids.device_twin(to, grid)

Return the immutable device twin of the host grid `grid` (see
[`@host_device_struct`](@ref)), with every field adapted with `to`. The CUDA
extension calls this from its kernel adaptor, so that kernels receive an isbits
grid while the host keeps the mutable one.
"""
function device_twin end

# The name of a grid without the `Host`/`Device` prefix of its twins.
public_name(grid::AbstractGrid) = nameof(typeof(grid))

"""
    Grids.dss_weights(grid::AbstractGrid, staggering::Union{Staggering, Nothing})

Return the direct stiffness summation (DSS) weights of `grid` at the given
`staggering`: a `DataLayout` of the inverse multiplicity of each node, weighted
by the node's metric Jacobian within each element, which `Spaces.weighted_dss!`
applies to average shared nodes across element boundaries. Return `nothing` for
discontinuous (`DG`) grids. Extruded grids reuse the weights of their horizontal
grid. If the grid is not staggered, `staggering` should be set to `nothing`.
"""
function dss_weights end

"""
    Grids.quadrature_style(grid::AbstractGrid)

Return the `Quadratures.QuadratureStyle` of the horizontal spectral element part
of `grid` (e.g. `Quadratures.GLL{4}()`). Throw a `MethodError` if `grid` has no
such horizontal part.
"""
function quadrature_style end

"""
    Grids.vertical_topology(grid::AbstractGrid)

Return the `Topologies.IntervalTopology` of the vertical part of `grid`: the
topology of a finite difference grid, or that of the vertical grid of an
extruded grid.
"""
function vertical_topology end

"""
    Grids.global_geometry(grid::AbstractGrid)

Return the `Geometry.AbstractGlobalGeometry` of `grid`, which relates its local
coordinates to a global Cartesian frame: `Geometry.CartesianGlobalGeometry` for
planar and interval domains, and a spherical global geometry (carrying the
`radius`) for grids on a sphere.
"""
function global_geometry end

"""
    Grids.hypsography(grid::AbstractGrid)

Return the `HypsographyAdaption` of an extruded grid: [`Flat`](@ref) when the
vertical coordinate is not adapted to surface topography, and a
terrain-following adaption otherwise. Levels of an extruded grid
(`Grids.LevelGrid`) return the hypsography of their full grid.
"""
function hypsography end

# The topology may be `nothing` in a kernel (see `ext/cuda/adapt.jl`), in which
# case the grid is on the device side.
ClimaComms.context(grid::AbstractGrid) =
    isnothing(topology(grid)) ? DeviceSideContext() :
    ClimaComms.context(topology(grid))
ClimaComms.device(grid::AbstractGrid) =
    isnothing(topology(grid)) ? DeviceSideDevice() :
    ClimaComms.device(topology(grid))

Meshes.domain(grid::AbstractGrid) = Meshes.domain(topology(grid))

include("finitedifference.jl")
include("spectralelement.jl")
include("multipoint.jl")
include("extruded.jl")
include("column.jl")
include("level.jl")

function Base.show(io::IO, grid::AbstractGrid)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, public_name(grid), ":")
    if has_horizontal(grid)
        # some reduced spaces (like slab space) do not have topology
        println(iio, " "^(indent + 2), "horizontal:")
        print(iio, " "^(indent + 4), "context: ")
        Topologies.print_context(iio, topology(grid).context)
        println(iio)
        println(iio, " "^(indent + 4), "mesh: ", topology(grid).mesh)
        print(iio, " "^(indent + 4), "quadrature: ", quadrature_style(grid))
    end
    if has_vertical(grid)
        has_horizontal(grid) && println(iio, "")
        println(iio, " "^(indent + 2), "vertical:")
        print(iio, " "^(indent + 4), "mesh: ", vertical_topology(grid).mesh)
    end
end

"""
    has_horizontal(::AbstractGrid)

Return `true` if the grid has a horizontal part.
"""
function has_horizontal end
has_horizontal(::AbstractGrid) = false
has_horizontal(::ExtrudedFiniteDifferenceGrid) = true
has_horizontal(::SpectralElementGrid2D) = true
has_horizontal(::SpectralElementGrid1D) = true
has_horizontal(::MultiPointGrid) = true

"""
    has_vertical(::AbstractGrid)

Return `true` if the grid has a vertical part.
"""
function has_vertical end
has_vertical(::AbstractGrid) = false
has_vertical(::FiniteDifferenceGrid) = true
has_vertical(::ExtrudedFiniteDifferenceGrid) = true

"""
    get_mask(grid::AbstractGrid)

Return the mask of `grid`; `DataLayouts.NoMask()` for grids without a mask.
"""
get_mask(::AbstractGrid) = DataLayouts.NoMask()
get_mask(grid::ExtrudedFiniteDifferenceGrid) = grid.horizontal_grid.mask
get_mask(::ExtrudedFiniteDifferenceGrid{<:MultiPointGrid}) = DataLayouts.NoMask()

"""
    set_mask!(fn, grid)
    set_mask!(grid, data::DataLayouts.DataLayout)

Set the active-node mask of `grid`. With `fn`, the mask is `fn(coord)` evaluated at
every coordinate of the horizontal grid; with `data`, the mask is copied from
`data`. The mask maps are then rebuilt with `DataLayouts.set_mask_maps!`. Does
nothing if the grid mask is a `DataLayouts.NoMask`.
"""
function set_mask! end

set_mask!(fn, grid::ExtrudedFiniteDifferenceGrid) =
    set_mask!(fn, grid.horizontal_grid)
function set_mask!(fn, grid::SpectralElementGrid2D)
    if !(grid.mask isa DataLayouts.NoMask)
        @. grid.mask.is_active = fn(grid.local_geometry.coordinates)
        DataLayouts.set_mask_maps!(grid.mask)
    end
    return nothing
end

set_mask!(grid::ExtrudedFiniteDifferenceGrid, data::DataLayouts.DataLayout) =
    set_mask!(grid.horizontal_grid, data)
function set_mask!(grid::SpectralElementGrid2D, data::DataLayouts.DataLayout)
    if !(grid.mask isa DataLayouts.NoMask)
        @. grid.mask.is_active = data
        DataLayouts.set_mask_maps!(grid.mask)
    end
    return nothing
end

end # module
