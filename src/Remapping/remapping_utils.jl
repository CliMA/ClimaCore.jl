"""
    AbstractRemappingMethod

Abstract type for horizontal remapping methods. Dispatch on concrete subtypes to avoid
branching in interpolation and Remapper logic.
"""
abstract type AbstractRemappingMethod end

"""
    SpectralElementRemapping <: AbstractRemappingMethod

Use spectral element quadrature weights (e.g. Lagrange at GLL points) for horizontal
interpolation.
"""
struct SpectralElementRemapping <: AbstractRemappingMethod end

"""
    BilinearRemapping{T12, T13, T14, T15} <: AbstractRemappingMethod

Use bilinear interpolation on the 2-point cell containing the target point (1D: linear on
2-point cell; 2D: bilinear on 2×2 cell). Holds precomputed local coordinates (s, t) and
node indices (i, j). For 1D horizontal, `local_bilinear_t` and `local_bilinear_j` are
`nothing`. Call `BilinearRemapping()` with no arguments to use as a method tag; the
Remapper constructor fills in the arrays.
"""
struct BilinearRemapping{T12, T13, T14, T15} <: AbstractRemappingMethod
    local_bilinear_s::T12
    local_bilinear_t::T13
    local_bilinear_i::T14
    local_bilinear_j::T15
end

"""`BilinearRemapping()` with no arguments: method tag; Remapper constructor fills in the arrays."""
BilinearRemapping() = BilinearRemapping(nothing, nothing, nothing, nothing)

"""
    vertical_indices(space, zcoords)

Return the vertical index of the element that contains `zcoords`.

`zcoords` is interpreted as "reference z coordinates".
"""
function vertical_indices(space, zcoords)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh
    return Meshes.containing_element.(Ref(vert_mesh), zcoords)
end

"""
    vertical_reference_coordinates(space, zcoords)

Return the reference coordinates of the element that contains `zcoords`.

Reference coordinates (ξ) typically go from -1 to 1, but for center spaces this function
remaps in such a way that they can directly be used for linear interpolation (so, if ξ is
negative, it is remapped to (0,1), if positive to (-1, 0)). This is best used alongside with
`vertical_bounding_indices`.

`zcoords` is interpreted as "reference z coordinates".
"""
function vertical_reference_coordinates(space, zcoords)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh

    is_cell_center =
        space isa Spaces.CenterExtrudedFiniteDifferenceSpace ||
        space isa Spaces.CenterFiniteDifferenceSpace

    ξ3s = map(zcoords) do zcoord
        velem = Meshes.containing_element(vert_mesh, zcoord)
        ξ3, = Meshes.reference_coordinates(vert_mesh, velem, zcoord)
        # For cell centered spaces, shift ξ3 so that we can use it for linear interpolation
        is_cell_center && (ξ3 = ξ3 < 0 ? ξ3 + 1 : ξ3 - 1)
        return ξ3
    end

    return ξ3s
end


"""
    vertical_bounding_indices(space, zcoords)

Return the vertical element indices needed to perform linear interpolation of `zcoords`.

For centered-valued fields, if `zcoord` is in the top (bottom) half of a top (bottom)
element in a column, no interpolation is performed and the value at the cell center is
returned. Effectively, this means that the interpolation is first-order accurate across the
column, but zeroth-order accurate close to the boundaries.
"""
function vertical_bounding_indices end

function vertical_bounding_indices(
    space::Union{
        Spaces.FaceExtrudedFiniteDifferenceSpace,
        Spaces.FaceFiniteDifferenceSpace,
    },
    zcoords,
)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh
    velems = Meshes.containing_element.(Ref(vert_mesh), zcoords)
    return map(v -> (v, v + 1), velems)
end

function vertical_bounding_indices(
    space::Union{
        Spaces.CenterExtrudedFiniteDifferenceSpace,
        Spaces.CenterFiniteDifferenceSpace,
    },
    zcoords,
)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh
    Nz = Spaces.nlevels(space)

    is_periodic = Topologies.isperiodic(vert_topology)

    vert_indices = map(zcoords) do zcoord
        velem = Meshes.containing_element(vert_mesh, zcoord)
        ξ3, = Meshes.reference_coordinates(vert_mesh, velem, zcoord)
        if ξ3 < 0
            v_lo = is_periodic ? mod1(velem - 1, Nz) : max(velem - 1, 1)
            v_hi = velem
        else
            v_lo = velem
            v_hi = is_periodic ? mod1(velem + 1, Nz) : min(velem + 1, Nz)
        end
        return v_lo, v_hi
    end

    return vert_indices
end


"""
    vertical_interpolation_weights(space, zcoords)

Compute the interpolation weights to vertically interpolate the `zcoords` in the given `space`.

This assumes a linear interpolation, where the first weight is to be multiplied with the "lower"
element, and the second weight with the "higher" element in a stack.

That is, this function returns `A`, `B` such that `f(zcoord) = A f_lo + B f_hi`, where `f_lo` and
`f_hi` are the values on the neighboring elements of `zcoord`.
"""
function vertical_interpolation_weights(space, zcoords)
    ξs = vertical_reference_coordinates(space, zcoords)
    return map(ξ -> ((1 - ξ) / 2, (1 + ξ) / 2), ξs)
end

function default_target_hcoords(
    space::Spaces.FiniteDifferenceSpace;
    hresolution,
)
    return nothing
end

function default_target_zcoords(
    space::Spaces.AbstractSpectralElementSpace;
    zresolution,
)
    return nothing
end

"""
    default_target_hcoords(space::Spaces.AbstractSpace; hresolution)

Return an Array with the Geometry.Points to interpolate uniformly the horizontal
component of the given `space`.
"""
function default_target_hcoords(space::Spaces.AbstractSpace; hresolution = 180)
    return default_target_hcoords(Spaces.horizontal_space(space); hresolution)
end

"""
    default_target_hcoords_as_vectors(space::Spaces.AbstractSpace; hresolution)

Return an Vectors with the coordinate to interpolate uniformly the horizontal
component of the given `space`.
"""
function default_target_hcoords_as_vectors(
    space::Spaces.AbstractSpace;
    hresolution = 180,
)
    return default_target_hcoords_as_vectors(
        Spaces.horizontal_space(space);
        hresolution,
    )
end

function default_target_hcoords(
    space::Spaces.SpectralElementSpace2D;
    hresolution = 180,
)
    topology = Spaces.topology(space)
    domain = Meshes.domain(topology.mesh)
    xrange, yrange = default_target_hcoords_as_vectors(space; hresolution)
    PointType =
        domain isa Domains.SphereDomain ? Geometry.LatLongPoint :
        Topologies.coordinate_type(topology)
    return [PointType(x, y) for x in xrange, y in yrange]
end

function default_target_hcoords_as_vectors(
    space::Spaces.SpectralElementSpace2D;
    hresolution = 180,
)
    FT = Spaces.undertype(space)
    topology = Spaces.topology(space)
    domain = Meshes.domain(topology.mesh)
    if domain isa Domains.SphereDomain
        return FT.(range(-90.0, 90.0, hresolution)),
        FT.(range(-180.0, 180.0, hresolution))
    else
        x1min = Geometry.component(domain.interval1.coord_min, 1)
        x2min = Geometry.component(domain.interval2.coord_min, 1)
        x1max = Geometry.component(domain.interval1.coord_max, 1)
        x2max = Geometry.component(domain.interval2.coord_max, 1)
        return FT.(range(x1min, x1max, hresolution)),
        FT.(range(x2min, x2max, hresolution))
    end
end

function default_target_hcoords(
    space::Spaces.SpectralElementSpace1D;
    hresolution = 180,
)
    topology = Spaces.topology(space)
    PointType = Topologies.coordinate_type(topology)
    return PointType.(default_target_hcoords_as_vectors(space; hresolution))
end

function default_target_hcoords_as_vectors(
    space::Spaces.SpectralElementSpace1D;
    hresolution = 180,
)
    FT = Spaces.undertype(space)
    topology = Spaces.topology(space)
    domain = Meshes.domain(topology.mesh)
    xmin = Geometry.component(domain.coord_min, 1)
    xmax = Geometry.component(domain.coord_max, 1)
    return FT.(range(xmin, xmax, hresolution))
end


"""
    default_target_zcoords(space::Spaces.AbstractSpace; zresolution)

Return an `Array` with the `Geometry.Points` to interpolate the vertical component of the
given `space`.

When `zresolution` is `nothing`, return the levels (which essentially disables vertical
interpolation), otherwise return linearly spaced values.
"""
function default_target_zcoords(space; zresolution = nothing)
    return Geometry.ZPoint.(
        default_target_zcoords_as_vectors(space; zresolution)
    )
end

function default_target_zcoords_as_vectors(space; zresolution = nothing)
    if isnothing(zresolution)
        # If has to be center space for the interpolation to be correct
        cspace = Spaces.space(space, Grids.CellCenter())
        return Array(Fields.field2array(Fields.coordinate_field(cspace).z))[
            :,
            1,
        ]
    else
        return collect(
            range(Domains.z_min(space), Domains.z_max(space), zresolution),
        )
    end
end

"""
    bilinear(c11, c21, c22, c12, s, t)

Bilinear interpolation in (s,t) ∈ [0,1]² (local to 2-point cell; reference element is [-1,1]²):
(1-s)(1-t)*c11 + s*(1-t)*c21 + (1-s)*t*c12 + s*t*c22
"""
@inline bilinear(c11, c21, c22, c12, s, t) =
    (1 - s) * (1 - t) * c11 + s * (1 - t) * c21 + (1 - s) * t * c12 + s * t * c22

"""
    linear(c1, c2, s)

Linear interpolation in s ∈ [0,1] on 2-point cell (1D analogue of bilinear).
"""
@inline linear(c1, c2, s) = (1 - s) * c1 + s * c2
