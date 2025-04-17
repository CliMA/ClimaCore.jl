# Masks

## Motivation

ClimaCore spaces, `SpectralElementSpace2D`s in particular, support masks, where
users can set horizontal nodal locations where operations are skipped.

This is especially helpful for the land model, where they may have degrees of
freedom over the ocean, but do not want to evaluate expressions in regions where
data is missing.

Masks in ClimaCore offer a solution to this by, ahead of time prescribing
regions to skip. This helps both with the ergonomics, as well as performance.

## User interface

There are two user-facing parts for ClimaCore masks:

 - set the `enable_mask = true` keyword in the space constructor (when available),
   which is currently any constructor that returns/contains a `SpectralElementSpace2D`.
 - use `set_mask!` to set where the mask is `true` (where compute should occur)
   and `false` (where compute should be skipped)

Here is an example

```julia
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
# Or
mask = Fields.Field(FT, ᶜspace)
mask .= map(cf -> cf.lat > 0.5 ? 0.0 : 1.0, Fields.coordinate_field(mask))
Spaces.set_mask!(ᶜspace, mask)
```

Finally, operations over fields will be skipped where `mask == 0`, and applied
where `mask == 1`:

```
@. f = 1 # only applied where the mask is equal to 1
```

## Example script

Here is a more complex script where the mask is used:

```julia
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
foo(f, cf) = cf.lat > 0.5 ? zero(f) : sqrt(-1) # results in NaN in masked out regions
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

Currently, masked _operations_ are only supported for `Fields` (and not
`DataLayouts`) with `SpectralElementSpace2D`s. We do not yet have support for
masked `SpectralElement1DSpace`s, and we will likely never offer masked
operation support for `DataLayouts`, as they do not have the space, and can
therefore not use the mask.

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
   - `sum` (mask-unaware, warning is thrown)
   - `extrema` (mask-unaware, warning is thrown)
   - `min` (mask-unaware, warning is thrown)
   - `max` (mask-unaware, warning is thrown)
 - field constructors (`copy`, `Fields.Field`, `ones`, `zeros`) are mask-unaware.
   This was a design implementation detail, users should not generally depend on the results where `mask == 0`, in case this is changed in the future. 
 - internal array operations (`fill!(parent(field), 0)`) mask-unaware.

## Temporary work-arounds

We can perform mask-aware reductions with the following work-around

```julia
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

# Set the mask
Spaces.set_mask!(ᶜspace) do coords
    coords.lat > 0.5
end

# get the mask
mask = Spaces.get_mask(ᶜspace)

# make a field of ones
ᶜf = ones(ᶜspace) # ignores mask

# bitmask spanning datalayout
bm = DataLayouts.full_bitmask(mask, Fields.field_values(ᶜf));

# mask-unaware integral (includes jacobian weighting)
@show sum(ᶜf)

# mask-unaware sum (excludes jacobian weighting)
@show sum(Fields.field_values(ᶜf))

# mask-aware sum (excludes jacobian)
@show sum(parent(ᶜf)[bm])

# level mask
ᶜf_lev = Fields.level(ᶜf, 1);
bm_lev = DataLayouts.full_bitmask(mask, Fields.field_values(ᶜf_lev));
@show sum(parent(ᶜf_lev)[bm_lev])
```

## Developer docs

In order to support masks, we define their types in `DataLayouts`, since 
we need access to them from within kernels in `DataLayouts`. We could have made
an API and kept them completely orthogonal, but that would have been a bit more
complicated, also, it was convenient to make the masks themselves data layouts,
so it seemed most natural for them to live there.

We have a couple types:

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

An important note is that when we set the mask maps for active columns, the
order that they are assigned can be permuted without impacting correctness, but
this could have a big impact on performance on the gpu. We should investigate
this.

