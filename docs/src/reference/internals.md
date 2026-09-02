# Internal APIs

Symbols on this page are implementation details: they are referenced from
docstrings of the public API and documented here so that those references
resolve, but they are not part of the interface a model relies on and may
change without notice.

## Operators

```@meta
CurrentModule = ClimaCore.Operators
```

```@docs
return_eltype
return_space
stencil_interior_width
stencil_interior
boundary_width
stencil_left_boundary
stencil_right_boundary
left_interior_idx
right_interior_idx
fd_shmem_is_supported
```

```@docs
AbstractOperator
SpectralElementOperator
apply_operator
register_similar
buffer_similar
materialize_buffer
DGConnectivity
dg_connectivity
dg_ghost_connectivity
```

## DataLayouts

```@meta
CurrentModule = ClimaCore.DataLayouts
```

```@docs
RegisterArray
DataLayouts.register_similar
DataLayouts.buffer_similar
static_num_threads
```

## Grids

```@meta
CurrentModule = ClimaCore.Grids
```

```@docs
Grids.topology
Grids.local_geometry_data
```

## Geometry

```@meta
CurrentModule = ClimaCore.Geometry
```

```@docs
Components
bilinear_interpolate
```

## Topologies

```@meta
CurrentModule = ClimaCore.Topologies
```

```@docs
GhostFaceExchange
```

## Utilities

```@meta
CurrentModule = ClimaCore.Utilities
```

```@docs
Utilities.@drop_recursion_limits
```

## MatrixFields

```@meta
CurrentModule = ClimaCore.MatrixFields
```

```@docs
outer_diagonals
band_matrix_row_type
matrix_shape
column_axes
AbstractLazyOperator
replace_lazy_operator
FieldName
@name
FieldNameTree
FieldNameSet
is_lazy
lazy_main_diagonal
lazy_mul
LazySchurComplement
field_matrix_solver_cache
check_field_matrix_solver
run_field_matrix_solver!
solver_algorithm
lazy_preconditioner
preconditioner_cache
check_preconditioner
lazy_or_concrete_preconditioner
apply_preconditioner
get_scalar_keys
field_offset_and_type
```
