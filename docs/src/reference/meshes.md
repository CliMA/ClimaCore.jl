# Meshes

```@meta
CurrentModule = ClimaCore
```

A mesh is a division of a domain into elements.

## Mesh types

```@docs
Meshes.AbstractMesh
Meshes.AbstractMesh1D
Meshes.AbstractMesh2D
Meshes.IntervalMesh
Meshes.RectilinearMesh
Meshes.AbstractCubedSphere
Meshes.EquiangularCubedSphere
Meshes.EquidistantCubedSphere
Meshes.ConformalCubedSphere
```

## Local element map

```@docs
Meshes.LocalElementMap
Meshes.IntrinsicMap
Meshes.NormalizedBilinearMap
```

## Mesh stretching

```@docs
Meshes.Uniform
Meshes.ExponentialStretching
Meshes.GeneralizedExponentialStretching
Meshes.HyperbolicTangentStretching
```

## Mesh utilities

```@docs
Meshes.truncate_mesh
```

## Interfaces

```@docs
Meshes.domain
Meshes.elements
Meshes.nelements
Meshes.n_elements_per_panel_direction
Meshes.coordinates
Meshes.containing_element
Meshes.reference_coordinates
Meshes.linearindices
Meshes.element_horizontal_length_scale
```
