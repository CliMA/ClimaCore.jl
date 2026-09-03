"""
    interpolate_slab!(output_array, field, slab_indices, weights)

Interpolate `field` horizontally at one point per entry of `output_array`, writing the
results in place.

`slab_indices[k]` is the `Fields.SlabIndex` (level and element) of the `k`-th target point,
and `weights[k]` is a tuple of one (1D) or two (2D) vectors of interpolation weights over
the quadrature nodes of that element, as returned by `interpolation_weights`.
"""
interpolate_slab!(output_array, field::Fields.Field, slab_indices, weights) =
    interpolate_slab!(
        output_array,
        field::Fields.Field,
        slab_indices,
        weights,
        ClimaComms.device(field),
    )


# CPU kernel for 3D configurations
function interpolate_slab!(
    output_array,
    field::Fields.Field,
    slab_indices,
    weights::AbstractArray{Tuple{A, A}},
    device::ClimaComms.AbstractCPUDevice,
) where {A}
    space = axes(field)
    FT = Spaces.undertype(space)

    @inbounds for index in 1:length(output_array)
        (I1, I2) = weights[index]
        Nq1, Nq2 = length(I1), length(I2)

        output_array[index] = zero(FT)

        for j in 1:Nq2, i in 1:Nq1
            ij = CartesianIndex((i, j))
            output_array[index] +=
                I1[i] *
                I2[j] *
                Operators.get_node(space, field, ij, slab_indices[index])
        end
    end
end

# CPU kernel for 2D configurations
function interpolate_slab!(
    output_array,
    field::Fields.Field,
    slab_indices,
    weights::AbstractArray{Tuple{A}},
    device::ClimaComms.AbstractCPUDevice,
) where {A}
    space = axes(field)
    FT = Spaces.undertype(space)

    @inbounds for index in 1:length(output_array)
        (I1,) = weights[index]
        Nq = length(I1)

        output_array[index] = zero(FT)

        for i in 1:Nq
            ij = CartesianIndex((i,))
            output_array[index] +=
                I1[i] *
                Operators.get_node(space, field, ij, slab_indices[index])
        end
    end
end

"""
    vertical_indices_ref_coordinate(space, zcoord)

Return `(v_lo, v_hi, ξ3)`: the two vertical level indices that bracket `zcoord` and the
reference coordinate `ξ3 ∈ [-1, 1]` of `zcoord` between them, so that linear
interpolation reads `((1 - ξ3) * f_lo + (1 + ξ3) * f_hi) / 2`.

On a `FaceExtrudedFiniteDifferenceSpace`, `v_lo` and `v_hi` are the two faces of the
containing element. On a `CenterExtrudedFiniteDifferenceSpace`, they are the two nearest
cell centers; in a non-periodic column, they coincide for points in the outer half of the
top and bottom cells, so that the cell-center value is returned.
"""
function vertical_indices_ref_coordinate end

function vertical_indices_ref_coordinate(
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    zcoord,
)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh

    velem = Meshes.containing_element(vert_mesh, zcoord)
    ξ3, = Meshes.reference_coordinates(vert_mesh, velem, zcoord)
    v_lo, v_hi = velem - half, velem + half
    return v_lo, v_hi, ξ3
end

function vertical_indices_ref_coordinate(
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    zcoord,
)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh
    Nz = Spaces.nlevels(space)

    velem = Meshes.containing_element(vert_mesh, zcoord)
    ξ3, = Meshes.reference_coordinates(vert_mesh, velem, zcoord)
    if ξ3 < 0
        if Topologies.isperiodic(Spaces.vertical_topology(space))
            v_lo = mod1(velem - 1, Nz)
        else
            v_lo = max(velem - 1, 1)
        end
        v_hi = velem
        ξ3 = ξ3 + 1
    else
        v_lo = velem
        if Topologies.isperiodic(Spaces.vertical_topology(space))
            v_hi = mod1(velem + 1, Nz)
        else
            v_hi = min(velem + 1, Nz)
        end
        ξ3 = ξ3 - 1
    end
    return v_lo, v_hi, ξ3
end

"""
    interpolate_slab_level!(output_array, field, h, Is, vertical_indices_ref_coordinates)

Interpolate `field` at one horizontal point of element `h`, given by the horizontal weights
`Is`, and at each vertical position in `vertical_indices_ref_coordinates`, writing one
value per vertical position into `output_array`.

`Is` is a tuple of one (1D) or two (2D) vectors of horizontal interpolation weights over the
quadrature nodes of the element. Each entry of `vertical_indices_ref_coordinates` is a
`(v_lo, v_hi, ξ3)` triple from `vertical_indices_ref_coordinate`: the field is interpolated
horizontally on levels `v_lo` and `v_hi` and then linearly in `ξ3` between them. On center
spaces, points in the outer half of the top and bottom cells take the cell-center value.
"""
function interpolate_slab_level!(
    output_array,
    field::Fields.Field,
    h::Integer,
    Is::Tuple,
    vertical_indices_ref_coordinates,
)
    device = ClimaComms.device(field)

    interpolate_slab_level!(
        output_array,
        field,
        vertical_indices_ref_coordinates,
        h,
        Is,
        device,
    )
end

# CPU kernel for 3D configurations
function interpolate_slab_level!(
    output_array,
    field::Fields.Field,
    vidx_ref_coordinates,
    h::Integer,
    (I1, I2)::Tuple{<:AbstractArray, <:AbstractArray},
    device::ClimaComms.AbstractCPUDevice,
)
    space = axes(field)
    FT = Spaces.undertype(space)
    Nq1, Nq2 = length(I1), length(I2)

    @inbounds for index in 1:length(vidx_ref_coordinates)
        v_lo, v_hi, ξ3 = vidx_ref_coordinates[index]

        f_lo = zero(FT)
        f_hi = zero(FT)

        for j in 1:Nq2, i in 1:Nq1
            ij = CartesianIndex((i, j))
            f_lo +=
                I1[i] *
                I2[j] *
                Operators.get_node(space, field, ij, Fields.SlabIndex(v_lo, h))
            f_hi +=
                I1[i] *
                I2[j] *
                Operators.get_node(space, field, ij, Fields.SlabIndex(v_hi, h))
        end

        output_array[index] = ((1 - ξ3) * f_lo + (1 + ξ3) * f_hi) / 2
    end
end

# CPU kernel for 2D configurations
function interpolate_slab_level!(
    output_array,
    field::Fields.Field,
    vidx_ref_coordinates,
    h::Integer,
    (I1,)::Tuple{<:AbstractArray},
    device::ClimaComms.AbstractCPUDevice,
)
    space = axes(field)
    FT = Spaces.undertype(space)
    Nq = length(I1)

    @inbounds for index in 1:length(vidx_ref_coordinates)
        v_lo, v_hi, ξ3 = vidx_ref_coordinates[index]

        f_lo = zero(FT)
        f_hi = zero(FT)

        for i in 1:Nq
            ij = CartesianIndex((i,))
            f_lo +=
                I1[i] *
                Operators.get_node(space, field, ij, Fields.SlabIndex(v_lo, h))
            f_hi +=
                I1[i] *
                Operators.get_node(space, field, ij, Fields.SlabIndex(v_hi, h))
        end
        output_array[index] = ((1 - ξ3) * f_lo + (1 + ξ3) * f_hi) / 2
    end
end

"""
    interpolate_array(field, xpts, zpts; horizontal_method = SpectralElementRemapping())
    interpolate_array(field, xpts, ypts, zpts; horizontal_method = SpectralElementRemapping())
    interpolate_array(field, xpts, ypts; horizontal_method = SpectralElementRemapping())

Interpolate `field` pointwise onto the Cartesian product of the given coordinate vectors
and return the values as an `Array` with one dimension per coordinate vector.

The first two methods apply to an `ExtrudedFiniteDifferenceField` with a 1D or 2D
horizontal space; the third applies to a `SpectralElementField2D`. Horizontal interpolation
follows `horizontal_method`: [`SpectralElementRemapping`](@ref) interpolates with the
Lagrange polynomial through all quadrature nodes of the element, [`BilinearRemapping`](@ref)
interpolates bilinearly between the bracketing quadrature nodes. Vertical interpolation is
linear between the two bracketing levels; on center spaces, points in the outer half of the
top and bottom cells take the cell-center value.

`field` must live on a single process (`SingletonCommsContext`). For distributed or
repeated remapping, build a [`Remapper`](@ref) and use [`interpolate`](@ref).

# Arguments

  - `field`: the `Field` to interpolate.
  - `xpts`, `ypts`: vectors of horizontal coordinate points (e.g. `Geometry.LongPoint`,
    `Geometry.LatPoint`, `Geometry.XPoint`); `xpts` and `ypts` are combined with
    `Geometry.product_coordinates`.
  - `zpts`: vector of `Geometry.ZPoint`s, interpreted as reference `z` coordinates.

# Examples

```julia
longpts = range(Geometry.LongPoint(-180.0), Geometry.LongPoint(180.0), length = 21)
latpts = range(Geometry.LatPoint(-80.0), Geometry.LatPoint(80.0), length = 21)
zpts = range(Geometry.ZPoint(0.0), Geometry.ZPoint(1000.0), length = 21)

interpolate_array(field, longpts, latpts, zpts)
```

!!! note

    `zpts` are located in the reference vertical mesh; hypsography (terrain-following
    heights) is ignored.
"""
function interpolate_array end

function interpolate_array(
    field::Fields.ExtrudedFiniteDifferenceField,
    xpts,
    zpts;
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
)
    space = axes(field)
    @assert ClimaComms.context(space) isa ClimaComms.SingletonCommsContext

    horz_topology = Spaces.topology(space)
    horz_mesh = horz_topology.mesh

    T = eltype(field)
    array = zeros(T, length(xpts), length(zpts))

    FT = Spaces.undertype(space)

    vertical_indices_ref_coordinates =
        [vertical_indices_ref_coordinate(space, zcoord) for zcoord in zpts]

    @inbounds for (ix, xcoord) in enumerate(xpts)
        hcoord = xcoord
        helem = Meshes.containing_element(horz_mesh, hcoord)
        quad = Spaces.quadrature_style(space)
        quad_points, _ = Quadratures.quadrature_points(FT, quad)
        weights = interpolation_weights(horz_mesh, hcoord, quad_points, horizontal_method)
        h = helem

        interpolate_slab_level!(
            view(array, ix, :),
            field,
            h,
            weights,
            vertical_indices_ref_coordinates,
        )
    end
    return array
end

function interpolate_array(
    field::Fields.ExtrudedFiniteDifferenceField,
    xpts,
    ypts,
    zpts;
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
)
    space = axes(field)
    @assert ClimaComms.context(space) isa ClimaComms.SingletonCommsContext

    horz_topology = Spaces.topology(space)
    horz_mesh = horz_topology.mesh

    T = eltype(field)
    array = zeros(T, length(xpts), length(ypts), length(zpts))

    FT = Spaces.undertype(space)

    vertical_indices_ref_coordinates =
        [vertical_indices_ref_coordinate(space, zcoord) for zcoord in zpts]

    @inbounds for (iy, ycoord) in enumerate(ypts),
        (ix, xcoord) in enumerate(xpts)

        hcoord = Geometry.product_coordinates(xcoord, ycoord)
        helem = Meshes.containing_element(horz_mesh, hcoord)
        quad = Spaces.quadrature_style(space)
        quad_points, _ = Quadratures.quadrature_points(FT, quad)
        weights = interpolation_weights(horz_mesh, hcoord, quad_points, horizontal_method)
        gidx = horz_topology.orderindex[helem]
        h = gidx

        interpolate_slab_level!(
            view(array, ix, iy, :),
            field,
            h,
            weights,
            vertical_indices_ref_coordinates,
        )
    end
    return array
end

function interpolate_array(
    field::Fields.SpectralElementField2D,
    xpts,
    ypts;
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
)
    space = axes(field)
    @assert ClimaComms.context(space) isa ClimaComms.SingletonCommsContext

    horz_topology = Spaces.topology(space)
    horz_mesh = horz_topology.mesh

    T = eltype(field)
    array = zeros(T, length(xpts), length(ypts))

    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    quad_points, _ = Quadratures.quadrature_points(FT, quad)

    @inbounds for (iy, ycoord) in enumerate(ypts),
        (ix, xcoord) in enumerate(xpts)

        hcoord = Geometry.product_coordinates(xcoord, ycoord)
        helem = Meshes.containing_element(horz_mesh, hcoord)
        weights = interpolation_weights(horz_mesh, hcoord, quad_points, horizontal_method)
        gidx = horz_topology.orderindex[helem]
        h = gidx

        (I1, I2) = weights
        Nq1, Nq2 = length(I1), length(I2)
        val = zero(FT)
        for j in 1:Nq2, i in 1:Nq1
            ij = CartesianIndex((i, j))
            slabidx = Fields.SlabIndex(nothing, h)
            val += I1[i] * I2[j] * Operators.get_node(space, field, ij, slabidx)
        end
        array[ix, iy] = val
    end
    return array
end

"""
    interpolation_weights(horz_mesh, hcoord, quad_points, method::AbstractRemappingMethod)

Return the horizontal interpolation weights for the point `hcoord` as a tuple of one
(`AbstractMesh1D`) or two (`AbstractMesh2D`) vectors, with one weight per quadrature node
of the containing element in each direction.

With `SpectralElementRemapping()`, the weights are the Lagrange basis polynomials through
`quad_points` evaluated at the reference coordinates of `hcoord`. With
`BilinearRemapping()`, only the two nodes that bracket the point in each direction have
nonzero weights, `1 - s` and `s`, where `s ∈ [0, 1]` is the local coordinate between them.
"""
function interpolation_weights end

function interpolation_weights(
    horz_mesh::Meshes.AbstractMesh2D,
    hcoord,
    quad_points,
    ::SpectralElementRemapping,
)
    helem = Meshes.containing_element(horz_mesh, hcoord)
    ξ1, ξ2 = Meshes.reference_coordinates(horz_mesh, helem, hcoord)
    WI1 = Quadratures.interpolation_matrix(SVector(ξ1), quad_points)
    WI2 = Quadratures.interpolation_matrix(SVector(ξ2), quad_points)
    return (WI1, WI2)
end

function interpolation_weights(
    horz_mesh::Meshes.AbstractMesh2D,
    hcoord,
    quad_points,
    ::BilinearRemapping,
)
    helem = Meshes.containing_element(horz_mesh, hcoord)
    ξ1, ξ2 = Meshes.reference_coordinates(horz_mesh, helem, hcoord)
    Nq = length(quad_points)
    FT = promote_type(typeof(ξ1), eltype(quad_points))
    # 2-point cell containing (ξ1, ξ2): linear weights (1-s,s) and (1-t,t).
    i = clamp(searchsortedlast(quad_points, ξ1), 1, Nq - 1)
    j = clamp(searchsortedlast(quad_points, ξ2), 1, Nq - 1)
    s = (ξ1 - quad_points[i]) / (quad_points[i + 1] - quad_points[i])
    t = (ξ2 - quad_points[j]) / (quad_points[j + 1] - quad_points[j])
    WI1 = ntuple(k -> k == i ? 1 - s : (k == i + 1 ? s : zero(FT)), Nq)
    WI2 = ntuple(k -> k == j ? 1 - t : (k == j + 1 ? t : zero(FT)), Nq)
    return (SVector(WI1), SVector(WI2))
end

function interpolation_weights(
    horz_mesh::Meshes.AbstractMesh1D,
    hcoord,
    quad_points,
    ::SpectralElementRemapping,
)
    helem = Meshes.containing_element(horz_mesh, hcoord)
    ξ1, = Meshes.reference_coordinates(horz_mesh, helem, hcoord)
    WI1 = Quadratures.interpolation_matrix(SVector(ξ1), quad_points)
    return (WI1,)
end

function interpolation_weights(
    horz_mesh::Meshes.AbstractMesh1D,
    hcoord,
    quad_points,
    ::BilinearRemapping,
)
    helem = Meshes.containing_element(horz_mesh, hcoord)
    ξ1, = Meshes.reference_coordinates(horz_mesh, helem, hcoord)
    Nq = length(quad_points)
    FT = promote_type(typeof(ξ1), eltype(quad_points))
    i = clamp(searchsortedlast(quad_points, ξ1), 1, Nq - 1)
    s = (ξ1 - quad_points[i]) / (quad_points[i + 1] - quad_points[i])
    (SVector(ntuple(k -> k == i ? 1 - s : (k == i + 1 ? s : zero(FT)), Nq)),)
end
