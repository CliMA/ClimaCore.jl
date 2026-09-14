# Spaces

```@meta
CurrentModule = ClimaCore
```

A space is a grid together with the information a field needs to live on it:
for a vertical grid, the staggering (cell centers or cell faces). Two
discretizations are provided, spectral elements (continuous or discontinuous
Galerkin) in the horizontal and staggered finite differences in the vertical,
and their product is an *extruded* (hybrid) space.

![3D hybrid discretization in a Cartesian domain](../assets/DiscretizationSketch.png)

*An extruded space on a box: spectral elements with their quadrature nodes in
the horizontal, stacked over the cells of a staggered vertical grid.*

```@docs
Spaces
Spaces.AbstractSpace
Spaces.Δz_data
```

## Finite Difference Spaces

A finite-difference space holds one value per cell of an interval mesh,
either at the cell centers (`CenterFiniteDifferenceSpace`) or at the faces
between cells (`FaceFiniteDifferenceSpace`). Construct one of the two from the
mesh and derive the other from it with `Spaces.face_space` or
`Spaces.center_space`; the two share one grid and no geometry is allocated
twice.

```@docs
Spaces.AbstractFiniteDifferenceSpace
Spaces.FiniteDifferenceSpace
Spaces.CenterFiniteDifferenceSpace
Spaces.FaceFiniteDifferenceSpace
```

## Spectral Element Spaces

```@docs
Spaces.AbstractSpectralElementSpace
Spaces.SpectralElementSpace1D
Spaces.SpectralElementSpace2D
Spaces.RectilinearSpectralElementSpace2D
Spaces.CubedSphereSpectralElementSpace2D
Spaces.SpectralElementSpaceSlab
```

### Discretization: CG or DG

The Galerkin discretization of a spectral-element grid is a type parameter of
the grid, set with the `discretization` keyword of the grid and space
constructors and read back from the space
([Choose CG or DG](../howto/choose_cg_dg.md)).

The types and accessors are documented on the [Grids](grids.md) page:
`Grids.Discretization`, `Grids.CG`, `Grids.DG`, `Grids.discretization`,
`Grids.is_continuous`. `Spaces.discretization` and `Spaces.is_continuous` are
the same functions applied to a space.

```@docs
Spaces.node_horizontal_length_scale
```

## Extruded Finite Difference Spaces

```@docs
Spaces.ExtrudedFiniteDifferenceSpace
Spaces.ExtrudedFiniteDifferenceSpace2D
Spaces.ExtrudedFiniteDifferenceSpace3D
Spaces.CenterExtrudedFiniteDifferenceSpace
Spaces.FaceExtrudedFiniteDifferenceSpace
Spaces.CenterExtrudedFiniteDifferenceSpace2D
Spaces.FaceExtrudedFiniteDifferenceSpace2D
Spaces.CenterExtrudedFiniteDifferenceSpace3D
Spaces.FaceExtrudedFiniteDifferenceSpace3D
Spaces.ExtrudedSpectralElementSpace2D
Spaces.ExtrudedSpectralElementSpace3D
Spaces.ExtrudedRectilinearSpectralElementSpace3D
Spaces.ExtrudedCubedSphereSpectralElementSpace3D
```

## Point Spaces

```@docs
Spaces.AbstractPointSpace
Spaces.PointSpace
```

## Multi-column Spaces

```@docs
Spaces.MultiPointSpace
Spaces.MultiColumnFiniteDifferenceSpace
Spaces.CenterMultiColumnFiniteDifferenceSpace
Spaces.FaceMultiColumnFiniteDifferenceSpace
```

## Accessors

The grid behind a space and its parts. Accessors that a space forwards to its
grid are documented on the [Grids](grids.md) page and are called with the
`Grids` qualifier even when given a space: `Grids.topology(space)`,
`Grids.quadrature_style(space)`, `Grids.global_geometry(space)`,
`Grids.vertical_topology(space)`, `Grids.dss_weights(space)`,
`Grids.set_mask!(space, …)`, `Grids.get_mask(space)`, `Grids.hypsography(space)`. Likewise
`Domains.z_min(space)` and `Domains.z_max(space)`,
`Meshes.n_elements_per_panel_direction(space)`,
`Topologies.create_dss_buffer(field)` and `ClimaCore.level(space, i)`,
`ClimaCore.column(space, i, j, h)`.

```@docs
Spaces.grid
Spaces.staggering
Spaces.horizontal_space
Spaces.horizontal_grid
Spaces.vertical_grid
Spaces.center_space
Spaces.face_space
Spaces.has_horizontal
Spaces.has_vertical
Spaces.nlevels
Spaces.ncolumns
Spaces.undertype
Spaces.coordinates_data
Spaces.radius
Spaces.issubspace
Spaces.eachslabindex
```

## Utilities

```@docs
Spaces.area
Spaces.local_area
```
