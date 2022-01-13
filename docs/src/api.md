# API

```@meta
CurrentModule = ClimaCore
```

## DataLayouts

```@docs
DataLayouts
DataLayouts.IF
DataLayouts.IJF
DataLayouts.VF
DataLayouts.IFH
DataLayouts.IJFH
DataLayouts.VIFH
DataLayouts.VIJFH
```

## Geometry

### Coordinates
```@docs
Geometry.AbstractPoint
Geometry.float_type
```

## Domains

### Types
```@docs
Domains.AbstractDomain
Domains.IntervalDomain
Domains.RectangleDomain
Domains.SphereDomain
```

### Interfaces
```@docs
Domains.boundary_names
```

## Meshes

### Mesh types
```@docs
Meshes.AbstractMesh
Meshes.IntervalMesh
Meshes.RectilinearMesh
Meshes.AbstractCubedSphere
Meshes.EquiangularCubedSphere
Meshes.EquidistantCubedSphere
Meshes.ConformalCubedSphere
```

### Local element map

```@docs
Meshes.LocalElementMap
Meshes.LocalElementMap
Meshes.IntrinsicMap
Meshes.NormalizedBilinearMap
```

### Mesh stretching
```@docs
Meshes.Uniform
Meshes.ExponentialStretching
```

### Interfaces
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
```

## Topologies

### Types
```@docs
Topologies.AbstractTopology
Topologies.IntervalTopology
Topologies.Topology2D
```

### Interfaces
```@docs
Topologies.nlocalelems
Topologies.vertex_coordinates
Topologies.opposing_face
Topologies.interior_faces
Topologies.boundary_tags
Topologies.boundary_tag
Topologies.boundary_faces
Topologies.vertices
```

## Spaces

```@docs
Spaces
```

### Quadratures


```@docs
Spaces.Quadratures.QuadratureStyle
Spaces.Quadratures.GLL
Spaces.Quadratures.GL
Spaces.Quadratures.Uniform
Spaces.Quadratures.degrees_of_freedom
Spaces.Quadratures.polynomial_degree
Spaces.Quadratures.quadrature_points
Spaces.Quadratures.barycentric_weights
Spaces.Quadratures.interpolation_matrix
Spaces.Quadratures.differentiation_matrix
Spaces.Quadratures.orthonormal_poly
```

## RecursiveApply

```@docs
RecursiveApply
RecursiveApply.tuplemap
```

