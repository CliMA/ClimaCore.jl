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

## Geometry

```@meta
CurrentModule = ClimaCore.Geometry
```

```@docs
bilinear_interpolate
mul_with_projection
mul_return_type
```

## Meshes

```@meta
CurrentModule = ClimaCore.Meshes
```

```@docs
SharedVertices
face_connectivity_matrix
vertex_connectivity_matrix
opposing_face
is_boundary_face
boundary_face_name
```

## Topologies

```@meta
CurrentModule = ClimaCore.Topologies
```

```@docs
GhostFaceExchange
nsendelems
nghostelems
localelemindex
face_node_index
ghost_faces
vertex_node_index
ghost_vertices
ghost_neighboring_elements
dss_transform
dss_transform!
dss_untransform
dss_untransform!
```

## Limiters

```@meta
CurrentModule = ClimaCore.Limiters
```

```@docs
compute_element_bounds!
compute_neighbor_bounds_local!
compute_neighbor_bounds_ghost!
apply_limit_slab!
column_massborrow!
```

## Remapping

```@meta
CurrentModule = ClimaCore.Remapping
```

```@docs
default_target_hcoords
default_target_zcoords
```

## Utilities

```@meta
CurrentModule = ClimaCore.Utilities
```

```@docs
@drop_recursion_limits
stable_view
unionall_type
replace_type_parameter
fieldtype_vals
Utilities.new
is_inferred_type
return_type
unsafe_eltype
safe_eltype
safe_mapreduce
ConvertTo
AutoBroadcaster
is_auto_broadcastable
add_auto_broadcasters
drop_auto_broadcasters
auto_broadcasted
nested_broadcast
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
