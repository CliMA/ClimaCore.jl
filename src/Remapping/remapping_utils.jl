"""
    AbstractRemappingMethod

Supertype for horizontal remapping methods.

Subtypes:

  - [`SpectralElementRemapping`](@ref): Lagrange interpolation on all quadrature nodes of
    the containing element.
  - [`BilinearRemapping`](@ref): linear (1D) or bilinear (2D) interpolation between the two
    quadrature nodes that bracket the target point in each direction.

[`Remapper`](@ref), [`interpolate`](@ref), and [`interpolate_array`](@ref) dispatch on the concrete
subtype passed as `horizontal_method`.
"""
abstract type AbstractRemappingMethod end

"""
    SpectralElementRemapping <: AbstractRemappingMethod

Interpolate horizontally with the Lagrange polynomial through all quadrature nodes of the
containing element (the barycentric formula of [Berrut2004](@cite)).
"""
struct SpectralElementRemapping <: AbstractRemappingMethod end

"""
    BilinearRemapping{T12, T13, T14, T15} <: AbstractRemappingMethod
    BilinearRemapping()

Interpolate horizontally between the two quadrature nodes that bracket the target point in
each direction: linear in 1D, bilinear on a 2×2 node cell in 2D.

The no-argument constructor returns a method tag with all fields `nothing`; pass it as
`horizontal_method` to [`Remapper`](@ref), [`interpolate`](@ref), or [`interpolate_array`](@ref).
The `Remapper` constructor fills in the fields for its process-local target points.

# Fields

  - `local_bilinear_s`: local coordinate in the first direction, in `[0, 1]`, per target
    point.
  - `local_bilinear_t`: local coordinate in the second direction, in `[0, 1]`, per target
    point; `nothing` in 1D.
  - `local_bilinear_i`: index of the lower bracketing node in the first direction, per
    target point.
  - `local_bilinear_j`: index of the lower bracketing node in the second direction, per
    target point; `nothing` in 1D.
"""
struct BilinearRemapping{T12, T13, T14, T15} <: AbstractRemappingMethod
    local_bilinear_s::T12
    local_bilinear_t::T13
    local_bilinear_i::T14
    local_bilinear_j::T15
end

BilinearRemapping() = BilinearRemapping(nothing, nothing, nothing, nothing)

"""
    vertical_indices(space, zcoords)

Return the index of the vertical element of `space` that contains each of `zcoords`.

`zcoords` are interpreted as reference `z` coordinates, i.e., coordinates of the vertical
mesh before any terrain adaption.
"""
function vertical_indices(space, zcoords)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh
    return Meshes.containing_element.(Ref(vert_mesh), zcoords)
end

"""
    vertical_reference_coordinates(space, zcoords)

Return the reference coordinate ξ of each of `zcoords` within its containing vertical
element.

On face spaces, ξ ∈ [-1, 1] within the element. On center spaces, ξ is shifted by ±1 so
that it is the reference coordinate between the two neighboring cell centers: a point in
the lower half of the cell (ξ < 0) maps to (0, 1), a point in the upper half maps to
(-1, 0). The shifted value pairs with the level indices from `vertical_bounding_indices`
for linear interpolation.

`zcoords` are interpreted as reference `z` coordinates.
"""
function vertical_reference_coordinates(space, zcoords)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh

    is_cell_center =
        space isa Spaces.CenterExtrudedFiniteDifferenceSpace ||
        space isa Spaces.CenterFiniteDifferenceSpace ||
        space isa Spaces.CenterMultiColumnFiniteDifferenceSpace

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

Return, for each of `zcoords`, the pair `(v_lo, v_hi)` of vertical level indices between
which the field is interpolated linearly.

On face spaces, these are the two faces of the containing element. On center spaces, they
are the two cell centers nearest to the point. In a non-periodic column, a point in the
upper (lower) half of the top (bottom) cell gets `v_lo == v_hi`, so the interpolation
returns the cell-center value: the interpolation is first-order accurate in the interior of
the column and zeroth-order accurate in the outer half of the boundary cells.
"""
function vertical_bounding_indices end

function vertical_bounding_indices(
    space::Union{
        Spaces.FaceExtrudedFiniteDifferenceSpace,
        Spaces.FaceFiniteDifferenceSpace,
        Spaces.FaceMultiColumnFiniteDifferenceSpace,
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
        Spaces.CenterMultiColumnFiniteDifferenceSpace,
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

Return, for each of `zcoords`, the pair of weights `(A, B)` for linear vertical
interpolation in `space`, `f(zcoord) = A * f_lo + B * f_hi`, where `f_lo` and `f_hi` are
the values at the levels returned by `vertical_bounding_indices`.
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

# Point-cloud columns have no horizontal interpolation
function default_target_hcoords(
    space::Union{
        Spaces.MultiColumnFiniteDifferenceSpace,
        Spaces.MultiPointSpace,
    };
    hresolution = nothing,
)
    return nothing
end

# Point clouds have no vertical extent
function default_target_zcoords(
    space::Spaces.MultiPointSpace;
    zresolution = nothing,
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
    default_target_hcoords(space::Spaces.AbstractSpace; hresolution = 180)

Return an array of `Geometry.Point`s that cover the horizontal domain of `space` uniformly
with `hresolution` points per direction.

On the sphere, the result is a `hresolution × hresolution` array of `LatLongPoint`s, with
latitudes from -90 to 90 along the first dimension and longitudes from -180 to 180 along
the second (in degrees). On a plane, the points have the coordinate type of the horizontal
domain and span its extent. Return `nothing` for spaces without a horizontal direction to
interpolate over (`FiniteDifferenceSpace`, `MultiColumnFiniteDifferenceSpace`,
`MultiPointSpace`).
"""
function default_target_hcoords(space::Spaces.AbstractSpace; hresolution = 180)
    return default_target_hcoords(Spaces.horizontal_space(space); hresolution)
end

"""
    default_target_hcoords_as_vectors(space::Spaces.AbstractSpace; hresolution = 180)

Return the coordinate vectors underlying [`default_target_hcoords`](@ref): a tuple of two
vectors for 2D horizontal spaces (latitudes and longitudes, in degrees, on the sphere) or
a single vector for 1D horizontal spaces, each with `hresolution` uniformly spaced values of
the space's float type.
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
    default_target_zcoords(space::Spaces.AbstractSpace; zresolution = nothing)

Return a vector of `Geometry.ZPoint`s covering the vertical extent of `space`.

When `zresolution` is `nothing`, return the cell-center heights of the model levels, so
that vertical interpolation of a center field reproduces its values. Otherwise, return
`zresolution` uniformly spaced heights between `Domains.z_min(space)` and
`Domains.z_max(space)`. Return `nothing` for spaces without a vertical direction
(`AbstractSpectralElementSpace`, `MultiPointSpace`).
"""
function default_target_zcoords(space; zresolution = nothing)
    return Geometry.ZPoint.(
        default_target_zcoords_as_vectors(space; zresolution),
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

Return the bilinear interpolant at local coordinates `(s, t) ∈ [0, 1]²` of the corner
values `c11 = f(0, 0)`, `c21 = f(1, 0)`, `c22 = f(1, 1)`, and `c12 = f(0, 1)`:
`(1 - s) * (1 - t) * c11 + s * (1 - t) * c21 + (1 - s) * t * c12 + s * t * c22`.
"""
@inline bilinear(c11, c21, c22, c12, s, t) =
    (1 - s) * (1 - t) * c11 + s * (1 - t) * c21 + (1 - s) * t * c12 + s * t * c22

"""
    linear(c1, c2, s)

Return the linear interpolant at local coordinate `s ∈ [0, 1]` between `c1 = f(0)` and
`c2 = f(1)`: `(1 - s) * c1 + s * c2`.
"""
@inline linear(c1, c2, s) = (1 - s) * c1 + s * c2
