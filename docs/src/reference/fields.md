# Fields

```@meta
CurrentModule = ClimaCore
```

```@docs
Fields.Field
Fields.coordinate_field
Fields.local_geometry_field
Base.zeros(::Spaces.AbstractSpace)
Base.ones(::Spaces.AbstractSpace)
Base.sum(::Fields.Field)
Fields.local_sum
Fields.Statistics.mean(::Fields.Field)
Fields.LinearAlgebra.norm(::Fields.Field)
Fields.set!
Fields.ColumnIndex
Fields.bycolumn
Fields.Δz_field
```

## Slicing

```@docs
ClimaCore.level
ClimaCore.column
ClimaCore.slab
```

## Conversion to arrays

```@docs
Fields.fieldvector2array
Fields.fieldvector2array!
Fields.array2fieldvector
Fields.array2fieldvector!
```
