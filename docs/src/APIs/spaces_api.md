# Spaces

```@meta
CurrentModule = ClimaCore
```

A `Space` represents a discretized function space over some domain.
Currently two main discretizations are supported: Spectral Element Discretization
(both Continuous Galerkin and Discontinuous Galerkin types) and a staggered
Finite Difference Discretization. Combination of these two in the horizontal/vertical
directions, respectively, is what we call a _hybrid_ space.

Sketch of a 2DX hybrid discretization:

![3D hybrid discretization in a Cartesian domain](../DiscretizationSketch.png)

```@docs
Spaces
Spaces.Δz_data
```

## Finite Difference Spaces

ClimaCore.jl supports staggered Finite Difference discretizations. Finite Differences
discretize an interval domain by approximating the function by a value at either
the center of each element (also referred to as _cell_) (`CenterFiniteDifferenceSpace`),
or the interfaces (faces in 3D, edges in 2D or points in 1D) between elements
(`FaceFiniteDifferenceSpace`).

```@docs
Spaces.FiniteDifferenceSpace
```

Users should construct either the center or face space from the mesh, then construct
the other space from the original one: this internally reuses the same data structures, and avoids allocating additional memory.

### Internals

```@docs
Spaces.Δz_metric_component
```

## Spectral Element Spaces

```@docs
Spaces.SpectralElementSpace1D
Spaces.SpectralElementSpace2D
Spaces.SpectralElementSpaceSlab
```

```@docs
Spaces.node_horizontal_length_scale
```

## Extruded Finite Difference Spaces

```@docs
Spaces.ExtrudedFiniteDifferenceSpace
```

## Utilities

```@docs
Spaces.area
Spaces.local_area
```
