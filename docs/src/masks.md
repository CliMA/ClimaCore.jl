# Masks

ClimaCore spaces, `SpectralElement2DSpace`s in particular, support masks, where
users can set horizontal nodal locations where operations are skipped.

Users can enable the mask by using the `enable_mask = true` keyword argument in
any space constructor that results in a `SpectralElement2DSpace`.

Next, users can set the mask with one of two ways-- a function, or field
assignment:

```julia
Spaces.set_mask!(space) do coords
    coords.lat > 0.5
end
```
Or
```julia
mask # a ClimaCore Field
Spaces.set_mask!(space, mask)
```

Finally, operations over fields will be skipped where `mask == 0`, and applied
where `mask == 1`:

```
@. f = 1 # only applied where the mask is equal to 1
```

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

Currently, masks are only supported for `Fields` (and not `DataLayouts`) with
`SpectralElement2DSpace`s. We do not yet have support for
`SpectralElement1DSpace`s, and we will likely never offer support for
`DataLayouts`, as they do not have the space, and can therefore not use the
mask.

In addition, not all operations support masks. Here is a list of operations of
supported and unsupported mask operations:

 - `DataLayout` operations (`Fields.field_values(f) = 1`) unsupported (these )
 - `fill!` (`@. f = 1`) supported
 - point-wise `copyto!` (`@. f = 1 + z`) supported
 - stencil `copyto!` (`@. ᶜf = 1 + DivergenceF2C()(Geometry.WVector(ᶠf))`) supported (vertical derivatives and interpolations interpolations)
 - spectral element operations `copyto!` (`@. ᶜf = 1 + Operators.Divergence()(ᶠf)`) (unsupported)
 - reductions:
   - `sum` (unsupported)
   - `extrema` (unsupported)
   - `min` (unsupported)
   - `max` (unsupported)

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

