# Grids

```@meta
CurrentModule = ClimaCore
```

```@docs
Grids.CellFace
Grids.CellCenter
Grids.ColumnGrid
Grids.FiniteDifferenceGrid
Grids.ExtrudedFiniteDifferenceGrid
Grids.SpectralElementGrid1D
Grids.SpectralElementGrid2D
Grids.MultiPointGrid
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
```
