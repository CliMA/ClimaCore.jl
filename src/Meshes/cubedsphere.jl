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

"""
    LocalElementMap

An abstract type of mappings from the reference element to a physical domain.
"""
abstract type LocalElementMap end

"""
    IntrinsicMap()

This [`LocalElementMap`](@ref) uses the intrinsic mapping of the cubed sphere to
map the reference element to the physical domain.
"""
struct IntrinsicMap <: LocalElementMap end

"""
    NormalizedBilinearMap()

The [`LocalElementMap`](@ref) for meshes on spherical domains of
[Guba2014](@cite). It uses bilinear interpolation between the Cartesian
coordinates of the element vertices, then normalizes the result to lie on the
sphere.
"""
struct NormalizedBilinearMap <: LocalElementMap end


Base.show(io::IO, mesh::AbstractCubedSphere) = print(
    io,
    mesh.ne,
    "??",
    mesh.ne,
    "??",
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
    to_panel(panel, coord1::Geometry.Cartesian123Point)

Given a point at `coord1` on panel 1 of a sphere, transform
it to panel `panel`.
"""
function to_panel(panel::Integer, coord::Geometry.Cartesian123Point)
    ??0, ??x, ??y = Geometry.components(coord)
    if panel == 1
        return Geometry.Cartesian123Point(??0, ??x, ??y)
    elseif panel == 2
        return Geometry.Cartesian123Point(-??x, ??0, ??y)
    elseif panel == 3
        return Geometry.Cartesian123Point(-??x, -??y, ??0)
    elseif panel == 4
        return Geometry.Cartesian123Point(-??0, -??y, -??x)
    elseif panel == 5
        return Geometry.Cartesian123Point(??y, -??0, -??x)
    elseif panel == 6
        return Geometry.Cartesian123Point(??y, ??x, -??0)
    end
    error("invalid panel $panel")
end

"""
    panel = cubedspherepanel(coord::Geometry.Cartesian123Point)

Given a point `coord`, return its panel number (an integer between 1 and 6).
"""
function containing_panel(coord::Geometry.Cartesian123Point)
    maxdim = argmax(abs.(Geometry.components(coord)))
    if maxdim == 1
        return coord.x1 > 0 ? 1 : 4
    elseif maxdim == 2
        return coord.x2 > 0 ? 2 : 5
    elseif maxdim == 3
        return coord.x3 > 0 ? 3 : 6
    end
    error("invalid coordinates")
end

"""
    coord1 = from_panel(panel::Int, coord::Geometry.Cartesian123Point)

Given a point `coord` and its panel number `panel`, return its coordinates in panel 1
(`coord1`).
"""
function from_panel(panel::Int, coord::Geometry.Cartesian123Point)
    if panel == 1
        return Geometry.Cartesian123Point(coord.x1, coord.x2, coord.x3)
    elseif panel == 2
        return Geometry.Cartesian123Point(coord.x2, -coord.x1, coord.x3)
    elseif panel == 3
        return Geometry.Cartesian123Point(coord.x3, -coord.x1, -coord.x2)
    elseif panel == 4
        return Geometry.Cartesian123Point(-coord.x1, -coord.x3, -coord.x2)
    elseif panel == 5
        return Geometry.Cartesian123Point(-coord.x2, -coord.x3, coord.x1)
    elseif panel == 6
        return Geometry.Cartesian123Point(-coord.x3, coord.x2, coord.x1)
    end
    error("invalid panel")
end

function coordinates(
    mesh::AbstractCubedSphere,
    elem::CartesianIndex{3},
    vert::Integer,
)
    FT = typeof(mesh.domain.radius)
    ne = mesh.ne
    x, y, panel = elem.I
    # ??x, ??y ??? [-1,1] are the "panel coordinates" of the vertex
    # we arrange this calculation carefully so that if zero, the sign is
    # consistent with other vertices of the element. this makes it easier to
    # deal with branch cuts (e.g. when plotting on lat/long axes)

    if (vert == 1 || vert == 4)
        ??x = FT(2 * (x - 1) - ne) / ne
    else
        ??x = -FT(ne - 2 * x) / ne
    end
    if (vert == 1 || vert == 2)
        ??y = FT(2 * (y - 1) - ne) / ne
    else
        ??y = -FT(ne - 2 * y) / ne
    end
    return _coordinates(mesh, ??x, ??y, panel)
end
coordinates(
    mesh::AbstractCubedSphere,
    elem::CartesianIndex{3},
    ??::StaticArrays.SVector{2},
) = coordinates(mesh, elem, ??, mesh.localelementmap)

function coordinates(
    mesh::AbstractCubedSphere,
    elem::CartesianIndex{3},
    (??1, ??2)::StaticArrays.SVector{2},
    ::IntrinsicMap,
)
    FT = typeof(mesh.domain.radius)
    ne = mesh.ne
    x, y, panel = elem.I
    # again, we want to arrange the calculation carefully so that signed zeros are correct
    # - if isodd(ne) and ??x == 0, then ??1 == 0, and should have the same sign
    ??x = (??1 - FT(ne + 1 - 2 * x)) / ne
    ??y = (??2 - FT(ne + 1 - 2 * y)) / ne
    # - if iseven(ne) and ??x == 0, then ??1 == +/-1, and should have the _opposite_ sign
    if iseven(ne)
        if ??x == 0
            ??x = copysign(??x, -??1)
        end
        if ??y == 0
            ??y = copysign(??y, -??2)
        end
    end
    return _coordinates(mesh, ??x, ??y, panel)
end
function coordinates(
    mesh::AbstractCubedSphere,
    elem::CartesianIndex{3},
    (??1, ??2)::StaticArrays.SVector{2},
    ::NormalizedBilinearMap,
)
    radius = mesh.domain.radius
    c1, c2, c3, c4 = ntuple(v -> coordinates(mesh, elem, v), 4)
    c = Geometry.bilinear_interpolate((c1, c2, c3, c4), ??1, ??2)
    return c * (radius / LinearAlgebra.norm(Geometry.components(c)))
end

function containing_element(
    mesh::AbstractCubedSphere,
    coord::Geometry.Cartesian123Point,
)
    ne = mesh.ne
    panel = containing_panel(coord)
    ??x, ??y = _inv_coordinates(mesh, coord, panel)

    x = refindex(??x, ne)
    y = refindex(??y, ne)
    return CartesianIndex(x, y, panel)
end

reference_coordinates(
    mesh::AbstractCubedSphere,
    elem::CartesianIndex{3},
    coord::Geometry.Cartesian123Point,
) = reference_coordinates(mesh, elem, coord, mesh.localelementmap)

function reference_coordinates(
    mesh::AbstractCubedSphere,
    elem::CartesianIndex{3},
    coord::Geometry.Cartesian123Point,
    ::IntrinsicMap,
)
    x, y, panel = elem.I
    ne = mesh.ne
    ??x, ??y = _inv_coordinates(mesh, coord, panel)

    ??1 = refcoord(??x, ne, x)
    ??2 = refcoord(??y, ne, y)
    return StaticArrays.SVector(??1, ??2)
end
function reference_coordinates(
    mesh::AbstractCubedSphere,
    elem::CartesianIndex{3},
    coord::Geometry.Cartesian123Point,
    ::NormalizedBilinearMap,
)
    panel = elem[3]
    # convert everything to panel 1: this ensures x1 is non-zero
    cc = ntuple(v -> from_panel(panel, coordinates(mesh, elem, v)), 4)
    ??0, ??x, ??y = Geometry.components(from_panel(panel, coord))
    # by taking the ratio of the components, we remove the normalization
    # constant.
    ux = ??x / ??0
    uy = ??y / ??0
    return Geometry.bilinear_invert(
        map(c -> StaticArrays.SVector(c.x2 - ux * c.x1, c.x3 - uy * c.x1), cc),
    )
end




"""
    EquiangularCubedSphere <: AbstractCubedSphere

An equiangular gnomonic mesh proposed by [Ronchi1996](@cite).
Uses the element indexing convention of [`AbstractCubedSphere`](@ref).

# Constructors

    EquiangularCubedSphere(
        domain::Domains.SphereDomain,
        ne::Integer,
        localelementmap=NormalizedBilinearMap()
        )

Constuct an `EquiangularCubedSphere` on `domain` with `ne` elements across
each panel.
"""
struct EquiangularCubedSphere{S <: SphereDomain, E <: LocalElementMap} <:
       AbstractCubedSphere
    domain::S
    ne::Int
    localelementmap::E
end
EquiangularCubedSphere(domain::SphereDomain, ne::Int) =
    EquiangularCubedSphere(domain, ne, NormalizedBilinearMap())

# not yet provided by Base Julia
# https://github.com/JuliaLang/julia/issues/28943
# this appears to be more accurate than tan(pi * x)
tanpi(x) = sinpi(x) / cospi(x)
atanpi(x) = atan(x) / pi

function _coordinates(mesh::EquiangularCubedSphere, ??x, ??y, panel)
    radius = mesh.domain.radius
    ux = tanpi(??x / 4)
    uy = tanpi(??y / 4)
    ??0 = radius / hypot(ux, uy, 1)
    ??x = ux * ??0
    ??y = uy * ??0
    to_panel(panel, Geometry.Cartesian123Point(??0, ??x, ??y))
end
function _inv_coordinates(mesh::EquiangularCubedSphere, coord, panel)
    coord1 = from_panel(panel, coord)
    ??0, ??x, ??y = Geometry.components(coord1)
    ux = ??x / ??0
    uy = ??y / ??0
    ??x = 4 * atanpi(ux)
    ??y = 4 * atanpi(uy)
    return ??x, ??y
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
struct EquidistantCubedSphere{S <: SphereDomain, E <: LocalElementMap} <:
       AbstractCubedSphere
    domain::S
    ne::Int
    localelementmap::E
end
EquidistantCubedSphere(domain::SphereDomain, ne::Int) =
    EquidistantCubedSphere(domain, ne, NormalizedBilinearMap())

function _coordinates(mesh::EquidistantCubedSphere, ??x, ??y, panel)
    radius = mesh.domain.radius
    ??0 = radius / hypot(??x, ??y, 1)
    ??x = ??x * ??0
    ??y = ??y * ??0
    to_panel(panel, Geometry.Cartesian123Point(??0, ??x, ??y))
end
function _inv_coordinates(mesh::EquidistantCubedSphere, coord, panel)
    coord1 = from_panel(panel, coord)
    ??0, ??x, ??y = Geometry.components(coord1)
    ??x = ??x / ??0
    ??y = ??y / ??0
    return ??x, ??y
end


"""
    ConformalCubedSphere <: AbstractCubedSphere

A conformal mesh outlined in [Rancic1996](@cite).
Uses the element indexing convention of [`AbstractCubedSphere`](@ref).

# Constructors

    ConformalCubedSphere(domain::Domains.SphereDomain, ne::Integer)

Constuct a `ConformalCubedSphere` on `domain` with `ne` elements across
each panel.
"""
struct ConformalCubedSphere{S <: SphereDomain, E <: LocalElementMap} <:
       AbstractCubedSphere
    domain::S
    ne::Int
    localelementmap::E
end
ConformalCubedSphere(domain::SphereDomain, ne::Int) =
    ConformalCubedSphere(domain, ne, NormalizedBilinearMap())

function _coordinates(mesh::ConformalCubedSphere, ??x, ??y, panel)
    radius = mesh.domain.radius
    n??x, n??y, n??0 = CubedSphere.conformal_cubed_sphere_mapping(??x, ??y)
    ??0 = radius * n??0
    ??x = radius * n??x
    ??y = radius * n??y
    to_panel(panel, Geometry.Cartesian123Point(??0, ??x, ??y))
end
function _inv_coordinates(mesh::ConformalCubedSphere, coord, panel)
    coord1 = from_panel(panel, coord)
    ??0, ??x, ??y = Geometry.components(coord1)
    R = hypot(??x, ??y, ??0)
    abs??x, abs??y = CubedSphere.conformal_cubed_sphere_inverse_mapping(
        abs(??x) / R,
        abs(??y) / R,
        ??0 / R,
    )
    ??x = copysign(abs??x, ??x)
    ??y = copysign(abs??y, ??y)
    return ??x, ??y
end
