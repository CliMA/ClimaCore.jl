# Topologies

```@meta
CurrentModule = ClimaCore
```

A `Topology` determines the ordering and connections between elements of a mesh.
![Space-filling curve element ordering for a cubed sphere mesh](../cubedsphere_spacefillingcurve.png)

## Types

```@docs
Topologies.AbstractTopology
Topologies.IntervalTopology
Topologies.Topology2D
Topologies.spacefillingcurve
Topologies.nelems
Topologies.nneighbors
Topologies.nsendelems
Topologies.nghostelems
Topologies.localelemindex
Topologies.face_node_index
Topologies.ghost_faces
Topologies.vertex_node_index
Topologies.local_vertices
Topologies.ghost_vertices
Topologies.neighbors
```

## Interfaces

```@docs
Topologies.mesh
Topologies.nlocalelems
Topologies.vertex_coordinates
Topologies.opposing_face
Topologies.interior_faces
Topologies.boundary_tags
Topologies.boundary_tag
Topologies.boundary_faces
Topologies.local_neighboring_elements
Topologies.ghost_neighboring_elements
```
