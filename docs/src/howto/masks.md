# Mask horizontal points

A horizontal mask marks nodal columns of a `SpectralElementSpace2D`, or of a
space extruded from one, where operations are skipped. The land model uses it
for degrees of freedom over the ocean, where there is no data to evaluate
expressions on; skipping them keeps the code free of per-point conditionals
and saves the work.

## Prerequisites

A space constructed with `enable_mask = true`, which every constructor that
builds a `SpectralElementGrid2D` accepts.

## Steps

 1. Construct the space with the mask enabled.
 2. Set the mask with `Spaces.set_mask!`, either from a function of the
    coordinates that returns `true` where computation should occur, or from a
    field on the horizontal space that is `1` there and `0` elsewhere.
 3. Operate on fields as usual: mask-aware operations write only the active
    columns.

The example below does all three on a coarse cubed sphere (6 × 10 × 10
elements of 4 × 4 nodes, 10 levels) and checks the results; the counts are
verified when the documentation is built.

```@example masks
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Spaces, Fields, Geometry, Operators
using ClimaCore.CommonSpaces
using Test

FT = Float64
kwargs = (; z_elem = 10, z_min = 0, z_max = 1, radius = 10, h_elem = 10,
    n_quad_points = 4, staggering = CellCenter())
ᶜspace = ExtrudedCubedSphereSpace(FT; enable_mask = true, kwargs...)
ᶠspace = Spaces.face_space(ᶜspace)

# Set the mask from a predicate on the coordinates ...
Spaces.set_mask!(ᶜspace) do coords
    coords.lat > 0.5
end

# ... or from a 0/1 field on the horizontal space. `zeros` is mask-unaware and
# initializes every column; the broadcast that follows writes the active ones.
hspace = Spaces.horizontal_space(ᶜspace)
lat = Fields.coordinate_field(hspace).lat
mask = zeros(hspace)
@. mask = lat > 0.5
Spaces.set_mask!(ᶜspace, mask)

# The mask itself is an internal object; it is inspected here for the check.
is_active = Spaces.get_mask(ᶜspace).is_active
@test count(parent(is_active)) == 4640
@test length(parent(is_active)) == 9600

# Pointwise broadcasts honor the mask: `fill!` and `copyto!` write 4640 of
# the 9600 columns.
ᶜf = zeros(ᶜspace)
@. ᶜf = 1
@test count(==(1), parent(ᶜf)) == 4640 * Spaces.nlevels(ᶜspace)
ᶜz = Fields.coordinate_field(ᶜspace).z
@. ᶜf = 1 + 0 * ᶜz
@test count(==(1), parent(ᶜf)) == 4640 * Spaces.nlevels(ᶜspace)

# Vertical stencils honor it too: a NaN in a masked-out column of the argument
# does not reach the result.
ᶠcoords = Fields.coordinate_field(ᶠspace)
nan_outside(f, coord) = coord.lat > 0.5 ? zero(f) : oftype(f, NaN)
div = Operators.DivergenceF2C()
ᶜdiv = zeros(ᶜspace)
@. ᶜdiv = div(Geometry.WVector(nan_outside(0.0, ᶠcoords)))
@test count(isnan, parent(ᶜdiv)) == 0

# Without a mask, the same expression fills the 4960 columns outside the
# region with NaN on every level.
ᶜspace_no_mask = ExtrudedCubedSphereSpace(FT; kwargs...)
ᶠcoords_no_mask = Fields.coordinate_field(Spaces.face_space(ᶜspace_no_mask))
ᶜdiv_no_mask = zeros(ᶜspace_no_mask)
@. ᶜdiv_no_mask = div(Geometry.WVector(nan_outside(0.0, ᶠcoords_no_mask)))
@test count(isnan, parent(ᶜdiv_no_mask)) == 49600
nothing # hide
```

## Which operations honor the mask

Masks are a property of spaces, so they act on `Fields` operations, not on
`DataLayouts`, which carry no space. Masks exist for `SpectralElementSpace2D`
and the spaces extruded from it; `SpectralElementSpace1D` has none yet.

| Operation                                                        | Mask-aware                                      |
|:---------------------------------------------------------------- |:----------------------------------------------- |
| `fill!` through a broadcast (`@. f = 1`)                         | Yes                                             |
| Pointwise `copyto!` (`@. f = 1 + z`)                             | Yes                                             |
| Finite-difference stencils (`@. ᶜf = div(Geometry.WVector(ᶠf))`) | Yes                                             |
| Spectral-element operators (`@. f = Operators.Divergence()(u)`)  | No; horizontal operators run over every element |
| `FieldVector` broadcasts (`@. Y += 1`)                           | No                                              |
| Reductions (`sum`, `extrema`, `minimum`, `maximum`)              | No; a warning is issued                         |
| Field constructors (`zeros`, `ones`, `copy`, `Fields.Field`)     | No; every column is initialized                 |
| Operations on `parent(field)` or on data layouts                 | No                                              |

Do not rely on the values a mask-unaware operation leaves in masked-out
columns; they are an implementation detail.

## Implementation

The mask types live in `DataLayouts`, because the kernels there consume them.
`DataLayouts.NoMask` is the default and compiles to a no-op. `DataLayouts.IJHMask`
holds the Boolean field `is_active`, the number of active columns, and index
maps from a linear active-column index to the `(i, j, h)` location of each
active column; `Spaces.set_mask!` fills them. On a GPU, the maps let the
kernels launch one thread per active column instead of one per column, so no
threads are spent on masked-out points. The order in which active columns
appear in the maps does not affect results; its effect on GPU performance has
not been measured.
