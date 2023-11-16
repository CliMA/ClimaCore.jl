"""
    interpolate_slab!(output_array, field, slab_indices, weights)

Interpolate horizontal field on the given `slab_indices` using the given interpolation
`weights`.

`interpolate_slab!` interpolates several values at a fixed `z` coordinate. For this reason,
it requires several slab indices and weights.

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

    for index in 1:length(output_array)
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

    for index in 1:length(output_array)
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

# GPU
function interpolate_slab!(
    output_array,
    field::Fields.Field,
    slab_indices,
    weights,
    device::ClimaComms.CUDADevice,
)
    space = axes(field)
    FT = Spaces.undertype(space)

    output_cuarray = CuArray(zeros(FT, length(output_array)))
    cuweights = CuArray(weights)
    cuslab_indices = CuArray(slab_indices)

    nitems = length(output_array)
    nthreads, nblocks = Topologies._configure_threadblock(nitems)

    @cuda threads = (nthreads) blocks = (nblocks) interpolate_slab_kernel!(
        output_cuarray,
        field,
        cuslab_indices,
        cuweights,
    )

    output_array .= Array(output_cuarray)
end

# GPU kernel for 3D configurations
function interpolate_slab_kernel!(
    output_array,
    field,
    slab_indices,
    weights::AbstractArray{Tuple{A, A}},
) where {A}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    space = axes(field)
    FT = Spaces.undertype(space)

    if index <= length(output_array)
        I1, I2 = weights[index]
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
    return nothing
end

# GPU kernel for 2D configurations
function interpolate_slab_kernel!(
    output_array,
    field,
    slab_indices,
    weights::AbstractArray{Tuple{A}},
) where {A}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    space = axes(field)
    FT = Spaces.undertype(space)

    if index <= length(output_array)
        I1, = weights[index]
        Nq = length(I1)

        output_array[index] = zero(FT)

        for i in 1:Nq
            ij = CartesianIndex((i))
            output_array[index] +=
                I1[i] *
                Operators.get_node(space, field, ij, slab_indices[index])
        end
    end
    return nothing
end

"""
    vertical_indices_ref_coordinate(space, zcoord)

Return the vertical indices of the elements below and above `zcoord`.

Return also the correct reference coordinate `zcoord` for vertical interpolation.
"""
function vertical_indices end

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
    interpolate_slab_level(
                           field::Fields.Field,
                           h::Integer,
                           Is::Tuple,
                           zpts;
                           fill_value = eltype(field)(NaN)
                           )

Vertically interpolate the given `field` on `zpts`.

`interpolate_slab_level!` interpolates several values at a fixed horizontal coordinate.

The field is linearly interpolated across two neighboring vertical elements.

For centered-valued fields, if `zcoord` is in the top (bottom) half of a top (bottom)
element in a column, no interpolation is performed and the value at the cell center is
returned. Effectively, this means that the interpolation is first-order accurate across the
column, but zeroth-order accurate close to the boundaries.

Return `fill_value` when the vertical coordinate is negative.

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

    for index in 1:length(vidx_ref_coordinates)
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

    for index in 1:length(vidx_ref_coordinates)
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

# GPU
function interpolate_slab_level!(
    output_array,
    field::Fields.Field,
    vidx_ref_coordinates,
    h::Integer,
    Is::Tuple,
    device::ClimaComms.CUDADevice,
)
    cuvidx_ref_coordinates = CuArray(vidx_ref_coordinates)

    output_cuarray = CuArray(
        zeros(Spaces.undertype(axes(field)), length(vidx_ref_coordinates)),
    )

    nitems = length(vidx_ref_coordinates)
    nthreads, nblocks = Topologies._configure_threadblock(nitems)
    @cuda threads = (nthreads) blocks = (nblocks) interpolate_slab_level_kernel!(
        output_cuarray,
        field,
        cuvidx_ref_coordinates,
        h,
        Is,
    )
    output_array .= Array(output_cuarray)
end

# GPU kernel for 3D configurations
function interpolate_slab_level_kernel!(
    output_array,
    field,
    vidx_ref_coordinates,
    h,
    (I1, I2)::Tuple{<:AbstractArray, <:AbstractArray},
)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    space = axes(field)
    FT = Spaces.undertype(space)
    Nq1, Nq2 = length(I1), length(I2)

    if index <= length(vidx_ref_coordinates)
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
    return nothing
end

# GPU kernel for 2D configurations
function interpolate_slab_level_kernel!(
    output_array,
    field,
    vidx_ref_coordinates,
    h,
    (I1,)::Tuple{<:AbstractArray},
)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    space = axes(field)
    FT = Spaces.undertype(space)
    Nq = length(I1)

    if index <= length(vidx_ref_coordinates)
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
    return nothing
end


"""
    interpolate_array(field, xpts, ypts)
    interpolate_array(field, xpts, ypts, zpts)

Interpolate a field to a regular array using pointwise interpolation.

This is primarily used for plotting and diagnostics.

# Examples

```julia
longpts = range(Geometry.LongPoint(-180.0), Geometry.LongPoint(180.0), length = 21)
latpts = range(Geometry.LatPoint(-80.0), Geometry.LatPoint(80.0), length = 21)
zpts = range(Geometry.ZPoint(0.0), Geometry.ZPoint(1000.0), length = 21)

interpolate_array(field, longpts, latpts, zpts)
```

!!! note
    Hypsography is not currently handled correctly.
"""
function interpolate_array end

function interpolate_array(
    field::Fields.ExtrudedFiniteDifferenceField,
    xpts,
    zpts,
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

    for (ix, xcoord) in enumerate(xpts)
        hcoord = xcoord
        helem = Meshes.containing_element(horz_mesh, hcoord)
        quad = Spaces.quadrature_style(space)
        quad_points, _ = Spaces.Quadratures.quadrature_points(FT, quad)
        weights = interpolation_weights(horz_mesh, hcoord, quad_points)
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
    zpts,
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

    for (iy, ycoord) in enumerate(ypts), (ix, xcoord) in enumerate(xpts)
        hcoord = Geometry.product_coordinates(xcoord, ycoord)
        helem = Meshes.containing_element(horz_mesh, hcoord)
        quad = Spaces.quadrature_style(space)
        quad_points, _ = Spaces.Quadratures.quadrature_points(FT, quad)
        weights = interpolation_weights(horz_mesh, hcoord, quad_points)
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

"""
    interpolation_weights(horz_mesh, hcoord, quad_points)

Return the weights (tuple of arrays) to interpolate fields onto `hcoord` on the
given mesh and quadrature points.
"""
function interpolation_weights end

function interpolation_weights(
    horz_mesh::Meshes.AbstractMesh2D,
    hcoord,
    quad_points,
)
    helem = Meshes.containing_element(horz_mesh, hcoord)
    ξ1, ξ2 = Meshes.reference_coordinates(horz_mesh, helem, hcoord)
    WI1 = Spaces.Quadratures.interpolation_matrix(SVector(ξ1), quad_points)
    WI2 = Spaces.Quadratures.interpolation_matrix(SVector(ξ2), quad_points)
    return (WI1, WI2)
end

function interpolation_weights(
    horz_mesh::Meshes.AbstractMesh1D,
    hcoord,
    quad_points,
)
    helem = Meshes.containing_element(horz_mesh, hcoord)
    ξ1, = Meshes.reference_coordinates(horz_mesh, helem, hcoord)
    WI1 = Spaces.Quadratures.interpolation_matrix(SVector(ξ1), quad_points)
    return (WI1,)
end

"""
    interpolate_column(field, zpts, weights, gidx)

Interpolate the given `field` on the given points assuming the given interpolation_matrix
and global index in the topology.

The coefficients `weights` are computed with `Spaces.Quadratures.interpolation_matrix`.
See also `interpolate_array`.

Keyword arguments
==================

- `physical_z`: When `true`, the given `zpts` are interpreted as "from mean sea
                            level" (instead of "from surface") and hypsography is
                            interpolated. `NaN`s are returned for values that are below the
                            surface.

"""
function interpolate_column(
    field::Fields.ExtrudedFiniteDifferenceField,
    zpts,
    Is,
    gidx;
    physical_z = false,
    fill_value = Spaces.undertype(axes(field))(NaN),
)
    space = axes(field)

    # When we don't have hypsography, there is no notion of "interpolating hypsography". In
    # this case, the reference vertical points coincide with the physical ones. Setting
    # physical_z = false ensures that zpts_ref = zpts
    if space.hypsography isa Grids.Flat
        physical_z = false
    end

    output_array = zeros(Spaces.undertype(space), length(zpts))

    # If we have physical_z, we have to move the z coordinates from physical to reference
    # ones. We also have to deal with the fact that the output_array is not going to be
    # fully obtained through interpolation, and the head of it will be simply a collection
    # of `fill_values` (up to the surface). When we have topography, we compute the
    # reference z and take the positive ones (above the surface). Then, we the top of
    # `output_array` with `fill_value` and the rest with interpolated ones. To achieve this,
    # we have to make sure that we are passing around views of the same array (as opposed to
    # copied of it).
    if physical_z
        # We are hardcoding the transformation from Hypsography.LinearAdaption
        space.hypsography isa Hypsography.LinearAdaption ||
            error("Cannot interpolate $(space.hypsography) hypsography")

        FT = Spaces.undertype(axes(field))

        # interpolate_slab! takes a vector
        z_surface = [zero(FT)]

        interpolate_slab!(
            z_surface,
            space.hypsography.surface,
            [Fields.SlabIndex(nothing, gidx)],
            [Is],
        )
        z_surface = z_surface[]
        z_top = Spaces.vertical_topology(space).mesh.domain.coord_max.z

        zpts_ref = [
            Geometry.ZPoint((z.z - z_surface) / (1 - z_surface / z_top)) for
            z in zpts if z.z > z_surface
        ]

        # When zpts = zpts_ref, all the points are above the surface
        num_points_below_surface = length(zpts) - length(zpts_ref)

        fill!(
            (@view output_array[1:(1 + num_points_below_surface)]),
            fill_value,
        )
    else
        zpts_ref = zpts
        num_points_below_surface = 0
    end

    vertical_indices_ref_coordinates =
        [vertical_indices_ref_coordinate(space, zcoord) for zcoord in zpts_ref]

    interpolate_slab_level!(
        (@view output_array[(1 + num_points_below_surface):end]),
        field,
        gidx,
        Is,
        vertical_indices_ref_coordinates,
    )

    return output_array
end
