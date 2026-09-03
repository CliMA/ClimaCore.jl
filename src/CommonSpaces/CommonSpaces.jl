"""
    CommonSpaces

Keyword constructors for the spaces on the grids of [`CommonGrids`](@ref):
[`ExtrudedCubedSphereSpace`](@ref), [`CubedSphereSpace`](@ref),
[`ColumnSpace`](@ref), [`Box3DSpace`](@ref), [`SliceXZSpace`](@ref),
[`RectangleXYSpace`](@ref), and [`MultiColumnSpace`](@ref). Each forwards its
keyword arguments to the `CommonGrids` constructor of the same name and wraps
the grid in a space. Constructors of spaces with a vertical direction take the
additional keyword argument `staggering`, either [`CellCenter`](@ref)`()` or
[`CellFace`](@ref)`()`; `face_space` and `center_space` convert between the
two.
"""
module CommonSpaces

export ExtrudedCubedSphereSpace,
    CubedSphereSpace,
    ColumnSpace,
    Box3DSpace,
    SliceXZSpace,
    RectangleXYSpace,
    MultiColumnSpace,
    PointColumnEnsembleSpace,
    CellCenter,
    CellFace,
    face_space,
    center_space

import ClimaComms

import ..Grids: Staggering, CellCenter, CellFace
import ..Spaces
import ..CommonGrids
import ..CommonGrids:
    ExtrudedCubedSphereGrid,
    CubedSphereGrid,
    ColumnGrid,
    Box3DGrid,
    SliceXZGrid,
    RectangleXYGrid,
    MultiColumnGrid
import ..Spaces: face_space, center_space


"""
    ExtrudedCubedSphereSpace([FT = Float64]; staggering, kwargs...)

Construct a [`Spaces.ExtrudedFiniteDifferenceSpace`](@ref) with the vertical
`staggering`, either [`CellCenter`](@ref)`()` or [`CellFace`](@ref)`()`, on the
grid `ExtrudedCubedSphereGrid(FT; kwargs...)`. See
[`CommonGrids.ExtrudedCubedSphereGrid`](@ref) for `FT` and the keyword
arguments.

# Examples

```julia
using ClimaCore.CommonSpaces
space = ExtrudedCubedSphereSpace(;
    z_elem = 10,
    z_min = 0,
    z_max = 1,
    radius = 10,
    h_elem = 10,
    n_quad_points = 4,
    staggering = CellCenter(),
)
```

The corresponding face-centered space is `face_space(space)`, or the same call
with `staggering = CellFace()`.
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
    CubedSphereSpace([FT = Float64]; kwargs...)

Construct a [`Spaces.SpectralElementSpace2D`](@ref) on the grid
`CubedSphereGrid(FT; kwargs...)`. See [`CommonGrids.CubedSphereGrid`](@ref) for
`FT` and the keyword arguments.

# Examples

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
    ColumnSpace([FT = Float64]; staggering, kwargs...)

Construct a [`Spaces.FiniteDifferenceSpace`](@ref) with the vertical
`staggering`, either [`CellCenter`](@ref)`()` or [`CellFace`](@ref)`()`, on the
grid `ColumnGrid(FT; kwargs...)`. See [`CommonGrids.ColumnGrid`](@ref) for `FT`
and the keyword arguments.

# Examples

```julia
using ClimaCore.CommonSpaces
space = ColumnSpace(;
    z_elem = 10,
    z_min = 0,
    z_max = 10,
    staggering = CellCenter(),
)
```
"""
function ColumnSpace end
ColumnSpace(; kwargs...) = ColumnSpace(Float64; kwargs...)
ColumnSpace(::Type{FT}; staggering::Staggering, kwargs...) where {FT} =
    Spaces.FiniteDifferenceSpace(ColumnGrid(FT; kwargs...), staggering)

"""
    Box3DSpace([FT = Float64]; staggering, kwargs...)

Construct a [`Spaces.ExtrudedFiniteDifferenceSpace`](@ref) with the vertical
`staggering`, either [`CellCenter`](@ref)`()` or [`CellFace`](@ref)`()`, on the
grid `Box3DGrid(FT; kwargs...)`. See [`CommonGrids.Box3DGrid`](@ref) for `FT`
and the keyword arguments.

# Examples

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
    staggering = CellCenter(),
)
```
"""
function Box3DSpace end
Box3DSpace(; kwargs...) = Box3DSpace(Float64; kwargs...)
Box3DSpace(::Type{FT}; staggering::Staggering, kwargs...) where {FT} =
    Spaces.ExtrudedFiniteDifferenceSpace(Box3DGrid(FT; kwargs...), staggering)

"""
    SliceXZSpace([FT = Float64]; staggering, kwargs...)

Construct a [`Spaces.ExtrudedFiniteDifferenceSpace`](@ref) with the vertical
`staggering`, either [`CellCenter`](@ref)`()` or [`CellFace`](@ref)`()`, on the
grid `SliceXZGrid(FT; kwargs...)`. See [`CommonGrids.SliceXZGrid`](@ref) for
`FT` and the keyword arguments.

# Examples

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
    staggering = CellCenter(),
)
```
"""
function SliceXZSpace end
SliceXZSpace(; kwargs...) = SliceXZSpace(Float64; kwargs...)
SliceXZSpace(::Type{FT}; staggering::Staggering, kwargs...) where {FT} =
    Spaces.ExtrudedFiniteDifferenceSpace(SliceXZGrid(FT; kwargs...), staggering)

"""
    RectangleXYSpace([FT = Float64]; kwargs...)

Construct a [`Spaces.SpectralElementSpace2D`](@ref) on the grid
`RectangleXYGrid(FT; kwargs...)`. See [`CommonGrids.RectangleXYGrid`](@ref) for
`FT` and the keyword arguments.

# Examples

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

"""
    MultiColumnSpace([FT = Float64]; staggering, kwargs...)

Construct a [`Spaces.MultiColumnFiniteDifferenceSpace`](@ref) of independent
vertical columns, with the vertical `staggering` either [`CellCenter`](@ref)`()`
or [`CellFace`](@ref)`()`, on the grid `MultiColumnGrid(FT; kwargs...)`. See
[`CommonGrids.MultiColumnGrid`](@ref) for `FT` and the keyword arguments.

# Examples

```julia
using ClimaCore.CommonSpaces, ClimaCore.Geometry
points = [LatLongPoint(0.0, 0.0), LatLongPoint(10.0, 20.0), LatLongPoint(-5.0, 90.0)]
space = MultiColumnSpace(;
    points = points,
    z_elem = 10,
    z_min = 0,
    z_max = 10_000,
    staggering = CellCenter(),
)
```
"""
function MultiColumnSpace end
MultiColumnSpace(; kwargs...) = MultiColumnSpace(Float64; kwargs...)
MultiColumnSpace(::Type{FT}; staggering::Staggering, kwargs...) where {FT} =
    Spaces.MultiColumnFiniteDifferenceSpace(
        MultiColumnGrid(FT; kwargs...),
        staggering,
    )

# Backwards-compatibility alias for the old name.
Base.@deprecate_binding PointColumnEnsembleSpace MultiColumnSpace false

end # module
