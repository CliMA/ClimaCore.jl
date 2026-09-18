# Grids

```@meta
CurrentModule = ClimaCore
```

```@docs
Grids.AbstractGrid
Grids.CellFace
Grids.CellCenter
Grids.ColumnGrid
Grids.FiniteDifferenceGrid
Grids.ExtrudedFiniteDifferenceGrid
Grids.SpectralElementGrid1D
Grids.SpectralElementGrid2D
Grids.MultiPointGrid
Grids.LevelGrid
Grids.ColumnIndex
```

## Accessors

These accept a grid or a space built on it.

```@docs
Grids.topology
Grids.vertical_topology
Grids.quadrature_style
Grids.global_geometry
Grids.local_geometry_data
Grids.dss_weights
Grids.issubgrid
Grids.get_mask
Grids.set_mask!
```

## Discretization

The Galerkin discretization of a spectral-element grid is a type parameter of
the grid, set with the `discretization` keyword of the grid and space
constructors ([Choose CG or DG](../howto/choose_cg_dg.md)).

```@docs
Grids.Discretization
Grids.CG
Grids.DG
Grids.discretization
Grids.is_continuous
```

## Hypsography

```@docs
Grids.Flat
Grids.hypsography
```
