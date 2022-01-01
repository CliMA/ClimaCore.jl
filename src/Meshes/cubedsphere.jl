"""
    AbstractCubedSphere <: AbstractMesh2D

This is an abstract type of cubed-sphere meshes on `SphereDomain`s. A
cubed-sphere mesh has 6 panels, laid out as follows:
```
                                          :   Panel 1   :
                            +-------------+-------------+
                            |     +x1     |     +x1     |
                            |             |             |
                            |    Panel    |    Panel    |
                            |+x3   5   -x3|-x2   6   +x2|
                            |     -x2     |     -x3     |
                            |             |             |
                            |     -x1     |     -x1     |
              +-------------+-------------+-------------+
              |     -x2     |     -x2     |
              |             |             |
              |    Panel    |    Panel    |
              |+x1   3   -x1|+x3   4   -x3|
              |     +x3     |     -x1     |
              |             |             |
              |     +x2     |     +x2     |
+-------------+-------------+-------------+
|     +x3     |     +x3     |
|             |             |
|    Panel    |    Panel    |
|-x2   1   +x2|+x1   2   -x1|
|     +x1     |     +x2     |
|             |             |
|     -x3     |     -x3     |
+-------------+-------------+
:   Panel 6   :
```

This is the same panel ordering used by the S2 Geometry library (though we use 1-based
instead of 0-based numering).

Elements are indexed by a `CartesianIndex{3}` object, where the components are:
- horizontal element index (left to right) within each panel.
- vertical element index (bottom to top) within each panel.
- panel number

Subtypes should have the following fields:
- `domain`: a `SphereDomain`
- `ne`: number of elements across each panel

# External links
- [S2Geometry library](https://s2geometry.io/devguide/s2cell_hierarchy)
- [MIT GCM exch2](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/exch2.html?highlight=cube%20sphere#fig-48tile)
"""
abstract type AbstractCubedSphere <: AbstractMesh2D end

Base.show(io::IO, mesh::AbstractCubedSphere) = print(
    io,
    mesh.ne,
    "×",
    mesh.ne,
    "×",
    6,
    "-element ",
    nameof(typeof(mesh)),
    " of ",
    mesh.domain,
)

domain(mesh::AbstractCubedSphere) = mesh.domain
elements(mesh::AbstractCubedSphere) = CartesianIndices((mesh.ne, mesh.ne, 6))

is_boundary_face(mesh::AbstractCubedSphere, elem, face) = false
boundary_face_name(mesh::AbstractCubedSphere, elem, face) = nothing

function opposing_face(
    mesh::AbstractCubedSphere,
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
    panel_to_coordinates(panel, coord1::Geometry.Cartesian123Point)

Given a point at `coord1` on panel 1 of a sphere, transform
it to panel `panel`.
"""
function panel_to_coordinates(panel::Integer, coord::Geometry.Cartesian123Point)
    ζ0, ζx, ζy = Geometry.components(coord)
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
    error("invalid panel $panel")
end

"""
    panel, coord1 = coordinates_to_panel(coord::Geometry.Cartesian123Point)

Given a point `coord`, return its panel number (`panel`), and its coordinates in panel 1
(`coord1`).
"""
function coordinates_to_panel(coord::Geometry.Cartesian123Point)
    maxdim = argmax(abs.(Geometry.components(coord)))
    if maxdim == 1 && coord.x1 > 0
        return 1, Geometry.Cartesian123Point(coord.x1, coord.x2, coord.x3)
    elseif maxdim == 2 && coord.x2 > 0
        return 2, Geometry.Cartesian123Point(coord.x2, -coord.x1, coord.x3)
    elseif maxdim == 3 && coord.x3 > 0
        return 3, Geometry.Cartesian123Point(coord.x3, -coord.x1, -coord.x2)
    elseif maxdim == 1 && coord.x1 < 0
        return 4, Geometry.Cartesian123Point(-coord.x1, -coord.x3, -coord.x2)
    elseif maxdim == 2 && coord.x2 < 0
        return 5, Geometry.Cartesian123Point(-coord.x2, -coord.x3, coord.x1)
    elseif maxdim == 3 && coord.x3 < 0
        return 6, Geometry.Cartesian123Point(-coord.x3, coord.x2, coord.x1)
    end
    error("invalid coordinates $x")
end




"""
    EquiangularCubedSphere <: AbstractCubedSphere

An equiangular gnomonic mesh proposed by [Ronchi1996](@cite).
Uses the element indexing convention of [`AbstractCubedSphere`](@ref).

# Constructors

    EquiangularCubedSphere(domain::Domains.SphereDomain, ne::Integer)

Constuct an `EquiangularCubedSphere` on `domain` with `ne` elements across
each panel.
"""
struct EquiangularCubedSphere{S <: SphereDomain} <: AbstractCubedSphere
    domain::S
    ne::Int
end
function coordinates(mesh::EquiangularCubedSphere, elem, (ξ1, ξ2)::NTuple{2})
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
    panel_to_coordinates(panel, Geometry.Cartesian123Point(ζ0, ζx, ζy))
end

function containing_element(
    mesh::EquiangularCubedSphere,
    coord::Geometry.Cartesian123Point,
)
    ne = mesh.ne
    panel, coord1 = coordinates_to_panel(coord)
    ζ0, ζx, ζy = Geometry.components(coord1)
    ux = ζx / ζ0
    uy = ζy / ζ0
    ξx = 4 * atan(ux) / pi
    ξy = 4 * atan(uy) / pi

    x, ξ1 = split_refcoord(ξx, ne)
    y, ξ2 = split_refcoord(ξy, ne)

    return CartesianIndex(x, y, panel), (ξ1, ξ2)
end

"""
    EquidistantCubedSphere <: AbstractCubedSphere

An equidistant gnomonic mesh outlined in [Rancic1996](@cite) and [Nair2005](@cite).
Uses the element indexing convention of [`AbstractCubedSphere`](@ref).

# Constructors

    EquidistantCubedSphere(domain::Domains.SphereDomain, ne::Integer)

Constuct an `EquidistantCubedSphere` on `domain` with `ne` elements across
each panel.
"""
struct EquidistantCubedSphere{S <: SphereDomain} <: AbstractCubedSphere
    domain::S
    ne::Int
end
function coordinates(mesh::EquidistantCubedSphere, elem, (ξ1, ξ2)::NTuple{2})
    radius = mesh.domain.radius
    ne = mesh.ne
    x, y, panel = elem.I
    ξx = (2 * x - ne - 1 + ξ1) / ne
    ξy = (2 * y - ne - 1 + ξ2) / ne
    ζ0 = radius / hypot(ξx, ξy, 1)
    ζx = ξx * ζ0
    ζy = ξy * ζ0
    panel_to_coordinates(panel, Geometry.Cartesian123Point(ζ0, ζx, ζy))
end
function containing_element(
    mesh::EquidistantCubedSphere,
    coord::Geometry.Cartesian123Point,
)
    ne = mesh.ne
    panel, coord1 = coordinates_to_panel(coord)
    ζ0, ζx, ζy = Geometry.components(coord1)
    ξx = ζx / ζ0
    ξy = ζy / ζ0

    x, ξ1 = split_refcoord(ξx, ne)
    y, ξ2 = split_refcoord(ξy, ne)

    return CartesianIndex(x, y, panel), (ξ1, ξ2)
end


"""
    ConformalCubedSphere <: AbstractCubedSphere

A conformal mesh outlined in [Rancic1996](@cite).
Uses the element indexing convention of [`AbstractCubedSphere`](@ref).

# Constructors

    ConformalCubedSphere(domain::Domains.SphereDomain, ne::Integer)

Constuct an `ConformalCubedSphere` on `domain` with `ne` elements across
each panel.
"""
struct ConformalCubedSphere{S <: SphereDomain} <: AbstractCubedSphere
    domain::S
    ne::Int
end
function coordinates(mesh::ConformalCubedSphere, elem, (ξ1, ξ2)::NTuple{2})
    radius = mesh.domain.radius
    ne = mesh.ne
    x, y, panel = elem.I
    ξx = (2 * x - ne - 1 + ξ1) / ne
    ξy = (2 * y - ne - 1 + ξ2) / ne
    ζx, ζy, ζ0 = CubedSphere.conformal_cubed_sphere_mapping(ξx, ξy)
    panel_to_coordinates(
        panel,
        Geometry.Cartesian123Point(radius * ζ0, radius * ζx, radius * ζy),
    )
end
function containing_element(
    mesh::ConformalCubedSphere,
    coord::Geometry.Cartesian123Point,
)
    ne = mesh.ne
    panel, coord1 = coordinates_to_panel(coord)
    ζ0, ζx, ζy = Geometry.components(coord1)
    R = hypot(ζx, ζy, ζ0)
    ξx, ξy = CubedSphere.conformal_cubed_sphere_inverse_mapping(
        abs(ζx) / R,
        abs(ζy) / R,
        ζ0 / R,
    )

    x, ξ1 = split_refcoord(copysign(ξx, ζx), ne)
    y, ξ2 = split_refcoord(copysign(ξy, ζy), ne)

    return CartesianIndex(x, y, panel), (ξ1, ξ2)
end
