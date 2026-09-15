# Fields

```@meta
CurrentModule = ClimaCore
```

```@docs
Fields.Field
Fields.FieldVector
Fields.coordinate_field
Fields.local_geometry_field
Base.zeros(::Spaces.AbstractSpace)
Base.ones(::Spaces.AbstractSpace)
Base.sum(::Fields.Field)
Fields.local_sum
Fields.Statistics.mean(::Fields.Field)
Fields.LinearAlgebra.norm(::Fields.Field)
Fields.set!
Fields.bycolumn
Fields.Δz_field
```

## Accessing the data

```@docs
Fields.field_values
Base.parent(::Fields.Field)
Fields.field_vector_values
Fields.backing_array
Fields.component
```

## Slicing

`level`, `column` and `slab` are defined at the top level of `ClimaCore` and
apply to fields, spaces and data layouts alike. A column is addressed by a
`Grids.ColumnIndex`.

```@docs
ClimaCore.level
ClimaCore.column
ClimaCore.slab
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

Aliases naming the field of a given space type, for dispatch.

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
