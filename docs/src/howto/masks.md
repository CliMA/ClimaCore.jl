# Mask horizontal points

A horizontal mask marks nodal columns of a `SpectralElementSpace2D` (or of a
space extruded from one) where operations are skipped. The land model uses it
for degrees of freedom over the ocean, where there is no data to evaluate
expressions on; skipping them keeps the code free of per-point conditionals
and saves the work.

## Prerequisites

A space constructed with `enable_mask = true`, which every constructor that
builds a `SpectralElementSpace2D` accepts.

## Steps

1. Construct the space with the mask enabled.
2. Set the mask with `Spaces.set_mask!`, either from a function of the
   coordinates that returns `true` where computation should occur, or from a
   field that is `1` there and `0` elsewhere.

```@example masks
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Spaces, Fields
using ClimaCore.CommonSpaces
using Test

FT = Float64
ᶜspace = ExtrudedCubedSphereSpace(FT;
    z_elem = 10,
    z_min = 0,
    z_max = 1,
    radius = 10,
    h_elem = 10,
    n_quad_points = 4,
    staggering = CellCenter(),
    enable_mask = true,
)

# How to set the mask
Spaces.set_mask!(ᶜspace) do coords
    coords.lat > 0.5
end
# Or, from a field on the horizontal space that is 1 where computation should
# occur and 0 elsewhere. `zeros` initializes every column (constructors are
# mask-unaware); the broadcast that follows writes the active columns.
hspace = Spaces.horizontal_space(ᶜspace)
lat = Fields.coordinate_field(hspace).lat
mask = zeros(hspace)
@. mask = lat > 0.5
Spaces.set_mask!(ᶜspace, mask)
```

3. Operate on fields as usual. Mask-aware operations are skipped where the
   mask is `0` and applied where it is `1`:

```julia
@. f = 1 # only applied where the mask is equal to 1
```

## A worked example

Vertical operators respect the mask, so a `NaN` in a masked column does not
propagate into the result. The counts below are for this grid (6 × 10 × 10
elements of 4 × 4 nodes, 10 levels) and are checked when the docs are built.

```@example masks
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Spaces, Fields, DataLayouts, Geometry, Operators
using ClimaCore.CommonSpaces
using Test

FT = Float64
ᶜspace = ExtrudedCubedSphereSpace(FT;
    z_elem = 10,
    z_min = 0,
    z_max = 1,
    radius = 10,
    h_elem = 10,
    n_quad_points = 4,
    staggering = CellCenter(),
    enable_mask = true,
)
ᶠspace = Spaces.face_space(ᶜspace)
ᶠcoords = Fields.coordinate_field(ᶠspace)

# How to set the mask
Spaces.set_mask!(ᶜspace) do coords
    coords.lat > 0.5
end

# We also support the syntax `Spaces.set_mask!(::AbstractSpace, ::Field)`

# We can check the mask directly: (internals, only for demonstrative purposes)
mask = Spaces.get_mask(ᶜspace)
@test count(parent(mask.is_active)) == 4640
@test length(parent(mask.is_active)) == 9600

# Let's skip operations that use fill!
ᶜf = zeros(ᶜspace) # ignores mask
@. ᶜf = 1 # tests fill! # abides by mask

# Let's show that 4640 columns were impacted:
@test count(x->x==1, parent(ᶜf)) == 4640 * Spaces.nlevels(axes(ᶜf))
@test length(parent(ᶜf)) == 9600 * Spaces.nlevels(axes(ᶜf))

# Let's skip operations that use copyto!
ᶜz = Fields.coordinate_field(ᶜspace).z
ᶜf = zeros(ᶜspace)
@. ᶜf = 1 + 0 * ᶜz # tests copyto!

# Let's again show that 4640 columns were impacted:
@test count(x->x==1, parent(ᶜf)) == 4640 * Spaces.nlevels(axes(ᶜf))
@test length(parent(ᶜf)) == 9600 * Spaces.nlevels(axes(ᶜf))

# Let's skip operations in FiniteDifference operators
ᶠf = zeros(ᶠspace)
c = Fields.Field(FT, ᶜspace)
div = Operators.DivergenceF2C()
foo(f, cf) = cf.lat > 0.5 ? zero(f) : oftype(f, NaN) # NaN in the masked-out region
@. c = div(Geometry.WVector(foo(ᶠf, ᶠcoords)))

# Check that this field should never yield NaNs
@test count(isnan, parent(c)) == 0

# Doing the same thing with a space without a mask will yield NaNs:
ᶜspace_no_mask = ExtrudedCubedSphereSpace(FT;
    z_elem = 10,
    z_min = 0,
    z_max = 1,
    radius = 10,
    h_elem = 10,
    n_quad_points = 4,
    staggering = CellCenter(),
)
ᶠspace_no_mask = Spaces.face_space(ᶜspace_no_mask)
ᶠcoords_no_mask = Fields.coordinate_field(ᶠspace_no_mask)
c_no_mask = Fields.Field(FT, ᶜspace_no_mask)
ᶠf_no_mask = Fields.Field(FT, ᶠspace_no_mask)
@. c_no_mask = div(Geometry.WVector(foo(ᶠf_no_mask, ᶠcoords_no_mask)))
@test count(isnan, parent(c_no_mask)) == 49600
```

## Supported operations and caveats

Masked _operations_ are supported only for `Fields` (and not
`DataLayouts`) with `SpectralElementSpace2D`s. Masks on `SpectralElementSpace1D`s are future work; `DataLayouts` stay
mask-unaware by design, since a data layout carries no space and hence no mask.

In addition, some operations with masked fields skip masked regions
(i.e., mask-aware), and other operations execute everywhere
(i.e., mask-unaware), effectively ignoring the mask. Here is a list of
operations of mask-aware and mask-unaware:

  - `DataLayout` operations (`Fields.field_values(f) = 1`) mask-unaware (will likely never be mask-aware).
  - `fill!` (`@. f = 1`) mask-aware
  - point-wise `copyto!` (`@. f = 1 + z`) mask-aware
  - stencil `copyto!` (`@. ᶜf = 1 + DivergenceF2C()(Geometry.WVector(ᶠf))`) mask-aware (vertical derivatives and interpolations interpolations)
  - spectral element operations `copyto!` (`@. f = 1 + Operators.Divergence()(f)`), where `Operators.Divergence` carries out a divergence operation in horizontal directions. mask-unaware
  - fieldvector operations `copyto!` (`@. Y += 1`) mask-unaware
  - reductions:
      + `sum` (mask-unaware, warning is thrown)
      + `extrema` (mask-unaware, warning is thrown)
      + `min` (mask-unaware, warning is thrown)
      + `max` (mask-unaware, warning is thrown)
  - field constructors (`copy`, `Fields.Field`, `ones`, `zeros`) are mask-unaware.
    This was a design implementation detail, users should not generally depend on the results where `mask == 0`, in case this is changed in the future.
  - internal array operations (`fill!(parent(field), 0)`) mask-unaware.

## Implementation notes

Mask types live in `DataLayouts`, because the kernels there need them; the
masks are themselves data layouts. The types:

  - abstract `AbstractMask` for subtyping masks and use for generic interface
    methods
  - `NoMask` (the default), which is a lazy object that should effectively result
    in a no-op, without any loss of runtime performance
  - `IJHMask` currently the only supported horizontal mask, which contains
    `is_active` (defined in `set_mask!`), `N` (the number of active columns),
    and maps containing indices to the `i, j, h` locations where `is_active` is
    true. The maps are defined in `set_mask_maps!`, allows us to launch cuda
    kernels to only target the active columns, and threads are not wasted on
    non-existent columns. The logic to handle this is relatively thin, and
    extends our current `ext/cuda/datalayouts_threadblock.jl` api
    (via `masked_partition` and `masked_universal_index`).

The order in which active columns are assigned in the mask maps does not
affect correctness; its effect on GPU performance has not been measured.
