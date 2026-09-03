# Meshes

```@meta
CurrentModule = ClimaCore
```

A `Mesh` is a division of a domain into elements.

## Mesh types

```@docs
Meshes.AbstractMesh
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
Meshes.is_boundary_face
Meshes.boundary_face_name
Meshes.opposing_face
Meshes.coordinates
Meshes.containing_element
Meshes.reference_coordinates
Meshes.SharedVertices
Meshes.face_connectivity_matrix
Meshes.vertex_connectivity_matrix
Meshes.linearindices
Meshes.element_horizontal_length_scale
```
