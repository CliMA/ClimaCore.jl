"""
    AbstractCubedSphereMesh <: AbstractMesh2D

This is an abstract type of cubed-sphere meshes on `SphereDomain`s. Each mesh has
6 panels, laid out as follows:
```
                                          :    Face 1   :
                            +-------------+-------------+
                            |     +x1     |     +x1     |
                            |             |             |
                            |     Face    |     Face    |
                            |+x3   5   -x3|-x2   6   +x2|
                            |     -x2     |     -x3     |
                            |             |             |
                            |     -x1     |     -x1     |
              +-------------+-------------+-------------+
              |     -x2     |     -x2     |
              |             |             |
              |     Face    |     Face    |
              |+x1   3   -x1|+x3   4   -x3|
              |     +x3     |     -x1     |
              |             |             |
              |     +x2     |     +x2     |
+-------------+-------------+-------------+
|     +x3     |     +x3     |
|             |             |
|     Face    |     Face    |
|-x2   1   +x2|+x1   2   -x1|
|     +x1     |     +x2     |
|             |             |
|     -x3     |     -x3     |
+-------------+-------------+
:    Face 6   :
```

This is the same panel ordering used by the [S2Geometry library](https://s2geometry.io/devguide/s2cell_hierarchy) (though we use 1-based instead of 0-based numering).

Elements are indexed by a `CartesianIndex{3}` object, where the components are:
- horizontal element index (left to right) within each panel.
- vertical element index (bottom to top) within each panel.
- panel number

Subtypes should have the following fields:
- `domain`: a `SphereDomain`
- `ne`: number of elements across each panel
"""
abstract type AbstractCubedSphereMesh <: AbstractMesh2D end


domain(mesh::AbstractCubedSphereMesh) = mesh.domain
nelements(mesh::AbstractCubedSphereMesh) = mesh.ne^2 * 6

elements(mesh::AbstractCubedSphereMesh) =
    CartesianIndices((mesh.ne, mesh.ne, 6))

is_boundary_face(mesh::AbstractCubedSphereMesh, elem, face) = false
boundary_face_name(mesh::AbstractCubedSphereMesh, elem, face) = nothing

function opposing_face(
    mesh::AbstractCubedSphereMesh,
    element::CartesianIndex{3},
    face::Int,
)
    ne = mesh.ne
    x, y, panel = element.I
    if face == 1
        if y > 1
            (CartesianIndex(x, y - 1, panel), 3, true)
        elseif isodd(panel)
            (CartesianIndex(x, ne, mod1(panel - 1, 6)), 3, true)
        else
            (CartesianIndex(ne, ne - x + 1, mod1(panel - 2, 6)), 2, true)
        end
    elseif face == 2
        if x < ne
            (CartesianIndex(x + 1, y, panel), 4, true)
        elseif isodd(panel)
            (CartesianIndex(1, y, mod1(panel + 1, 6)), 4, true)
        else
            (CartesianIndex(ne - y + 1, 1, mod1(panel + 2, 6)), 1, true)
        end
    elseif face == 3
        if y < ne
            (CartesianIndex(x, y + 1, panel), 1, true)
        elseif isodd(panel)
            (CartesianIndex(1, ne - x + 1, mod1(panel + 2, 6)), 4, true)
        else
            (CartesianIndex(x, 1, mod1(panel + 1, 6)), 1, true)
        end
    elseif face == 4
        if x > 1
            (CartesianIndex(x - 1, y, panel), 2, true)
        elseif isodd(panel)
            (CartesianIndex(ne - y + 1, ne, mod1(panel - 2, 6)), 3, true)
        else
            (CartesianIndex(ne, y, mod1(panel - 1, 6)), 2, true)
        end
    end
end

"""
    panel_coordinates(panel, ζ0, ζx, ζy)

Given a point on a sphere (`ζ0, ζx, ζy`) (with reference to panel 1), transform
it to panel `panel`.
"""
function panel_coordinates(panel::Integer, ζ0, ζx, ζy)
    if panel == 1
        return Geometry.Cartesian123Point(ζ0, ζx, ζy)
    elseif panel == 2
        return Geometry.Cartesian123Point(-ζx, ζ0, ζy)
    elseif panel == 3
        return Geometry.Cartesian123Point(-ζx, -ζy, ζ0)
    elseif panel == 4
        return Geometry.Cartesian123Point(-ζ0, -ζy, -ζx)
    elseif panel == 5
        return Geometry.Cartesian123Point(ζy, -ζ0, -ζx)
    elseif panel == 6
        return Geometry.Cartesian123Point(ζy, ζx, -ζ0)
    end
    error("invalid panel")
end

"""
    EquiangularCubedSphereMesh <: AbstractCubedSphereMesh

An equiangular gnomonic mesh proposed by [Ronchi1996]
Uses the element indexing convention of [`AbstractCubedSphereMesh`](@ref).

# Constructors

    EquiangularCubedSphereMesh(domain::Domains.SphereDomain, ne::Integer)

Constuct an `EquiangularCubedSphereMesh` on `domain` with `ne` elements across
each panel.

# References
- The "Cubed Sphere": A New Method for the Solution of Partial Differential Equations in Spherical Geometry
  C. RONCHI, R. IACONO, AND P. S. PAOLUCCI JOURNAL OF COMPUTATIONAL PHYSICS 124, 93–114 (1996)
"""
struct EquiangularCubedSphereMesh{S <: SphereDomain} <: AbstractCubedSphereMesh
    domain::S
    ne::Int
end
function coordinates(
    mesh::EquiangularCubedSphereMesh,
    elem,
    (ξ1, ξ2)::NTuple{2},
)
    radius = mesh.domain.radius
    ne = mesh.ne
    x, y, panel = elem.I
    ξx = (2 * x - ne - 1 + ξ1) / ne
    ξy = (2 * y - ne - 1 + ξ2) / ne
    ux = tan(pi * ξx / 4)
    uy = tan(pi * ξy / 4)
    ζ0 = radius / hypot(ux, uy, 1)
    ζx = ux * ζ0
    ζy = uy * ζ0
    panel_coordinates(panel, ζ0, ζx, ζy)
end



"""
    EquidistantCubedSphereMesh <: AbstractCubedSphereMesh

An equidistant gnomonic mesh outlined in [Rancic1996] and [Nair2005].
Uses the element indexing convention of [`AbstractCubedSphereMesh`](@ref).

# Constructors

    EquidistantCubedSphereMesh(domain::Domains.SphereDomain, ne::Integer)

Constuct an `EquidistantCubedSphereMesh` on `domain` with `ne` elements across
each panel.

# References

- A global shallow-water model using an expanded spherical cube: Gnomonic versus
  conformal coordinates Rančić M., Purser R. J., Mesinger F.
  https://doi.org/10.1002/qj.49712253209

- A Discontinuous Galerkin Transport Scheme on the Cubed Sphere Ramachandran D.
  Nair, Stephen J. Thomas, and Richard D. Loft https://doi.org/10.1175/MWR2890.1

"""
struct EquidistantCubedSphereMesh{S <: SphereDomain} <: AbstractCubedSphereMesh
    domain::S
    ne::Int
end
function coordinates(
    mesh::EquidistantCubedSphereMesh,
    elem,
    (ξ1, ξ2)::NTuple{2},
)
    radius = mesh.domain.radius
    ne = mesh.ne
    x, y, panel = elem.I
    ξx = (2 * x - ne - 1 + ξ1) / ne
    ξy = (2 * y - ne - 1 + ξ2) / ne
    ζ0 = radius / hypot(ξx, ξy, 1)
    ζx = ξx * ζ0
    ζy = ξy * ζ0
    panel_coordinates(panel, ζ0, ζx, ζy)
end

"""
    ConformalCubedSphereMesh <: AbstractCubedSphereMesh

A conformal mesh outlined in [Rancic1996] and [Nair2005]
Uses the element indexing convention of [`AbstractCubedSphereMesh`](@ref).

# Constructors

    ConformalCubedSphereMesh(domain::Domains.SphereDomain, ne::Integer)

Constuct an `ConformalCubedSphereMesh` on `domain` with `ne` elements across
each panel.

# References
- A global shallow-water model using an expanded spherical cube: Gnomonic
  versus conformal coordinates
  Rančić M., Purser R. J., Mesinger F.
  https://doi.org/10.1002/qj.49712253209

"""
struct ConformalCubedSphereMesh{S <: SphereDomain} <: AbstractCubedSphereMesh
    domain::S
    ne::Int
end
function coordinates(
    mesh::ConformalCubedSphereMesh,
    elem,
    (ξ1, ξ2)::NTuple{2},
)
    radius = mesh.domain.radius
    ne = mesh.ne
    x, y, panel = elem.I
    ξx = (2 * x - ne - 1 + ξ1) / ne
    ξy = (2 * y - ne - 1 + ξ2) / ne
    ζx, ζy, ζ0 = CubedSphere.conformal_cubed_sphere_mapping(ξx, ξy)
    panel_coordinates(panel, radius*ζ0, radius*ζx, radius*ζy)
end
