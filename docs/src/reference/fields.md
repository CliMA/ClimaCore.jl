# Fields

```@meta
CurrentModule = ClimaCore
```

```@docs
Fields.Field
Fields.FieldVector
Fields.coordinate_field
Fields.local_geometry_field
Fields.local_sum
Fields.Statistics.mean(::Fields.Field)
Fields.LinearAlgebra.norm(::Fields.Field)
Fields.set!
Fields.bycolumn
Fields.Δz_field
```

```@docs
Base.zeros(::Spaces.AbstractSpace)
Base.ones(::Spaces.AbstractSpace)
Base.sum(::Fields.Field)
```

## Accessing the data

```@docs
Fields.field_values
Fields.field_vector_values
Fields.backing_array
Fields.component
```

```@docs
Base.parent(::Fields.Field)
```

## Slicing

`level`, `slab`, and `column` are defined at the top level of `ClimaCore` and
apply to fields, spaces, and data layouts alike. Columns may also be addressed
by a [`Grids.ColumnIndex`](@ref) instead of the standard `(i, j, h)` tuple.

```@docs
ClimaCore.level
ClimaCore.slab
ClimaCore.column
```

## Iterating over a FieldVector

```@docs
Fields.field_iterator
Fields.property_chains
Fields.single_field
Fields.rcompare
Fields.@rprint_diff
```

## Field types

Type aliases naming the field that corresponds to each space, for dispatch.

```@docs
Fields.PointField
Fields.SpectralElementField
Fields.SpectralElementField1D
Fields.SpectralElementField2D
Fields.RectilinearSpectralElementField2D
Fields.CubedSphereSpectralElementField2D
Fields.FiniteDifferenceField
Fields.CenterFiniteDifferenceField
Fields.FaceFiniteDifferenceField
Fields.ExtrudedFiniteDifferenceField
Fields.ExtrudedFiniteDifferenceField2D
Fields.ExtrudedFiniteDifferenceField3D
Fields.CenterExtrudedFiniteDifferenceField
Fields.FaceExtrudedFiniteDifferenceField
Fields.ExtrudedSpectralElementField2D
Fields.ExtrudedRectilinearSpectralElementField3D
Fields.ExtrudedCubedSphereSpectralElementField3D
Fields.MultiColumnFiniteDifferenceField
Fields.CenterMultiColumnFiniteDifferenceField
Fields.FaceMultiColumnFiniteDifferenceField
```

## Conversion to arrays

```@docs
Fields.field2array
Fields.array2field
Fields.fieldvector2array
Fields.fieldvector2array!
Fields.array2fieldvector
Fields.array2fieldvector!
```
