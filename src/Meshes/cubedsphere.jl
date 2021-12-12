
"""



              v5 (xs,xe,xe)          v6 (xe,xe,xe)
                 o--------e9----------o
                /|                   /|
               / |                  / |
              /  |                 /  |
            e8   e12              e7  e10
            /    |               /    |
           /     |            v3/     |
       v4 o--------e3----------o      |
          |   v8 o------e11----|------o v7 (xe,xe,xs)
          |     /              |     /
          |    /               |    /
          e4  e5               e2   e6
          |  /                 |  /
          | /                  | /
          |/                   |/
          o--------e1----------o
         v1                    v2
       (xs,xs,xs)               (xe,xs,xs)




            4--4--1--1--2
            ^     ^     ^
            8  5  5  6  6
            |     |     |
      4--8->5--12-8--11-7
      ^     ^     ^
      3  3  9  4  11
      |     |     |
4--3->3--7--6--10-7
^     ^     ^
4  1  2  2  10
|     |     |
1--1->2--6->7

"""
abstract type AbstractCubedSphereMesh <: AbstractMesh{2} end

struct EquiangularCubedSphereMesh{S <: SphereDomain} <: AbstractCubedSphereMesh
    domain::S
    ne::Int
end
#=
struct EquidistantCubedSphereMesh{FT} <: AbstractCubedSphereMesh{FT}
    domain::SphereDomain{FT}
    ne::Int
end

struct ConformalCubedSphereMesh{FT} <: AbstractCubedSphereMesh{FT}
    domain::SphereDomain{FT}
    ne::Int
end
=#
nelements(mesh::AbstractCubedSphereMesh) = mesh.ne^2 * 6

elements(mesh::AbstractCubedSphereMesh) =
    CartesianIndices((mesh.ne, mesh.ne, 6))

is_boundary_face(mesh::AbstractCubedSphereMesh, elem, face) = false

function opposing_face(
    mesh::AbstractCubedSphereMesh,
    element::CartesianIndex{3},
    face::Int,
)
    ne = mesh.ne
    x, y, panel = element.I
    if face == 1
        if y > 1
            (CartesianIndex(x, y - 1, panel), 3, false)
        elseif isodd(panel)
            (CartesianIndex(x, ne, mod1(panel - 1, 6)), 3, false)
        else
            (CartesianIndex(ne, ne - x + 1, mod1(panel - 2, 6)), 2, true)
        end
    elseif face == 2
        if x < ne
            (CartesianIndex(x + 1, y, panel), 4, false)
        elseif isodd(panel)
            (CartesianIndex(1, y, mod1(panel + 1, 6)), 4, false)
        else
            (CartesianIndex(ne - y + 1, 1, mod1(panel + 2, 6)), 1, true)
        end
    elseif face == 3
        if y < ne
            (CartesianIndex(x, y + 1, panel), 1, false)
        elseif isodd(panel)
            (CartesianIndex(1, ne - x + 1, mod1(panel + 2, 6)), 4, true)
        else
            (CartesianIndex(x, 1, mod1(panel + 1, 6)), 1, false)
        end
    elseif face == 4
        if x > 1
            (CartesianIndex(x - 1, y, panel), 2, false)
        elseif isodd(panel)
            (CartesianIndex(ne - y + 1, ne, mod1(panel - 2, 6)), 3, true)
        else
            (CartesianIndex(ne, y, mod1(panel - 1, 6)), 2, false)
        end
    end
end

function coordinates(mesh::EquiangularCubedSphereMesh, elem, vert::Integer)
    FT = typeof(mesh.domain.radius)
    ξ1 = (vert == 1 || vert == 4) ? FT(-1) : FT(1)
    ξ2 = (vert == 1 || vert == 2) ? FT(-1) : FT(1)
    coordinates(mesh, elem, (ξ1, ξ2))
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
    ζ0 = radius / sqrt(1 + ux^2 + uy^2)
    ζx = ux * ζ0
    ζy = uy * ζ0
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
    error("invalid index $elem")
end
