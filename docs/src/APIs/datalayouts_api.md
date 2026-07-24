# DataLayouts

```@meta
CurrentModule = ClimaCore
```

## Data layouts

```@docs
DataLayouts.DataLayout
DataLayouts.DataF
DataLayouts.VIJHWithF
DataLayouts.VIJFH
DataLayouts.VIJHF
DataLayouts.VIH1
DataLayouts.IH1JH2
```

## Layout properties

```@docs
DataLayouts.layout_type
DataLayouts.parent_type
DataLayouts.f_dim
DataLayouts.shape_params
DataLayouts.inferred_size
DataLayouts.has_inferred_size
DataLayouts.vijh_params
DataLayouts.nlevels
DataLayouts.nquadpoints
DataLayouts.nelems
DataLayouts.ncomponents
DataLayouts.layout_constructor
DataLayouts.rebuild
DataLayouts.reassign
```

## Data scopes

```@docs
DataLayouts.DataScope
DataLayouts.ThisThread
DataLayouts.ThisThreadPool
DataLayouts.partition
DataLayouts.is_subscope
DataLayouts.num_threads
DataLayouts.num_partitions
DataLayouts.thread_rank
DataLayouts.partition_rank
DataLayouts.parallelize_over
DataLayouts.synchronize
DataLayouts.scoped_array
DataLayouts.scoped_static_array
DataLayouts.strided_access
DataLayouts.subscope_indices
```

## Loops and reductions

```@docs
DataLayouts.each_slice_index
DataLayouts.slice_subscope
DataLayouts.foreach_slice
DataLayouts.foreach_point
DataLayouts.foreach_level
DataLayouts.foreach_slab
DataLayouts.foreach_column
DataLayouts.reduce_points
DataLayouts.column_reduce!
```

## Masks

```@docs
DataLayouts.DataMask
DataLayouts.NoMask
DataLayouts.IJHMask
DataLayouts.set_mask_maps!
DataLayouts.should_compute
```

## Struct storage

```@docs
DataLayouts.bitcast_struct
DataLayouts.default_basetype
DataLayouts.check_basetype
DataLayouts.checked_valid_basetype
DataLayouts.num_basetypes
DataLayouts.struct_field_view
DataLayouts.set_struct!
DataLayouts.get_struct
DataLayouts.view_struct
```

## Broadcasting

```@docs
DataLayouts.DataStyle
DataLayouts.LazyDataLayout
DataLayouts.layout_args
DataLayouts.modify_args
```
