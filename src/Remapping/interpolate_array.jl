"""
    interpolate_slab!(output_array, field, slab_indices, weights)

Interpolate horizontal field on the given `slab_indices` using the given interpolation
`weights`.

"""
interpolate_slab!(output_array, field::Fields.Field, slab_indices, weights) =
    interpolate_slab!(
        output_array,
        field::Fields.Field,
        slab_indices,
        weights,
        ClimaComms.device(field),
    )

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
    nthreads, nblocks = Spaces._configure_threadblock(nitems)

    @cuda threads = (nthreads) blocks = (nblocks) interpolate_slab_kernel!(
        output_cuarray,
        field,
        cuslab_indices,
        cuweights,
    )

    output_array .= Array(output_cuarray)
end

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

        for i in 1:Nq1
            ij = CartesianIndex((i))
            output_array[index] +=
                I1[i] *
                Operators.get_node(space, field, ij, slab_indices[index])
        end
    end
    return nothing
end

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

function interpolate_slab!(
    output_array,
    field::Fields.Field,
    slabidx::Fields.SlabIndex,
    weights::AbstractArray{Tuple{A}},
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
                           zcoord;
                           fill_value = eltype(field)(NaN)
                           )

Vertically interpolate the given `field` on `zcoord`.

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
    zpts;
    fill_value = eltype(field)(NaN),
)
    device = ClimaComms.device(field)
    interpolate_slab_level!(
        output_array,
        field,
        h,
        Is,
        zpts,
        device;
        fill_value,
    )
end

function interpolate_slab_level!(
    output_array,
    field::Fields.Field,
    h::Integer,
    (I1, I2)::Tuple{<:AbstractArray, <:AbstractArray},
    zpts,
    device::ClimaComms.AbstractCPUDevice;
    fill_value = Spaces.undertype(axes(field))(NaN),
)
    space = axes(field)
    FT = Spaces.undertype(space)
    Nq1, Nq2 = length(I1), length(I2)

    output_array .= map(zpts) do (zcoord)
        zcoord.z < 0 && return fill_value

        v_lo, v_hi, ξ3 =
            vertical_indices_ref_coordinate(axes(field), zcoord)

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

        return ((1 - ξ3) * f_lo + (1 + ξ3) * f_hi) / 2
    end
end

function interpolate_slab_level!(
    output_array,
    field::Fields.Field,
    h::Integer,
    (I1,)::Tuple{<:AbstractArray},
    zpts,
    device::ClimaComms.AbstractCPUDevice;
    fill_value = Spaces.undertype(axes(field))(NaN),
)
    space = axes(field)
    FT = Spaces.undertype(space)
    Nq = length(I1)

    output_array .= map(zpts) do (zcoord)
        zcoord.z < 0 && return fill_value

        v_lo, v_hi, ξ3 =
            vertical_indices_ref_coordinate(axes(field), zcoord)

        f_lo = zero(FT)
        f_hi = zero(FT)

        for i in 1:Nq
            ij = CartesianIndex((i,))
            f_lo +=
                I1[i] * Operators.get_node(
                    space,
                    field,
                    ij,
                    Fields.SlabIndex(v_lo, h),
                )
            f_hi +=
                I1[i] * Operators.get_node(
                    space,
                    field,
                    ij,
                    Fields.SlabIndex(v_hi, h),
                )
        end

        return ((1 - ξ3) * f_lo + (1 + ξ3) * f_hi) / 2
    end
end

function interpolate_slab_level!(
    output_array,
    field::Fields.Field,
    h::Integer,
    Is::Tuple,
    zpts,
    device::ClimaComms.CUDADevice;
    fill_value = Spaces.undertype(axes(field))(NaN),
)
    # We have to deal with topography and NaNs. For that, we select the points that have z
    # >= 0 (above the surface) and interpolate only those on the GPU. Then, we fill the
    # output array with fill_value and overwrite only those values that we computed with
    # interpolation. This is a simple way to avoid having branching on the GPU to check if
    # z>0.

    positive_zcoords_indices = [z.z >= 0 for z in zpts]

    vertical_indices_ref_coordinates = CuArray([
        vertical_indices_ref_coordinate(axes(field), zcoord) for
        zcoord in zpts[positive_zcoords_indices]
    ])

    output_cuarray = CuArray(
        zeros(
            Spaces.undertype(axes(field)),
            length(vertical_indices_ref_coordinates),
        ),
    )

    nitems = length(zpts)
    nthreads, nblocks = Spaces._configure_threadblock(nitems)
    @cuda threads = (nthreads) blocks = (nblocks) interpolate_slab_level_kernel!(
        output_cuarray,
        field,
        vertical_indices_ref_coordinates,
        h,
        Is,
    )
    output_array .= fill_value
    output_array[positive_zcoords_indices] .= Array(output_cuarray)
end

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

    if index <= length(output_array)
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

    if index <= length(output_array)
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
    for (ix, xcoord) in enumerate(xpts)
        hcoord = xcoord
        helem = Meshes.containing_element(horz_mesh, hcoord)
        quad = Spaces.quadrature_style(space)
        quad_points, _ = Spaces.Quadratures.quadrature_points(FT, quad)
        weights = interpolation_weights(horz_mesh, hcoord, quad_points)
        h = helem

        interpolate_slab_level!(view(array, ix, :), field, h, weights, zpts)
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

    for (iy, ycoord) in enumerate(ypts), (ix, xcoord) in enumerate(xpts)
        hcoord = Geometry.product_coordinates(xcoord, ycoord)
        helem = Meshes.containing_element(horz_mesh, hcoord)
        quad = Spaces.quadrature_style(space)
        quad_points, _ = Spaces.Quadratures.quadrature_points(FT, quad)
        weights = interpolation_weights(horz_mesh, hcoord, quad_points)
        gidx = horz_topology.orderindex[helem]
        h = gidx

        interpolate_slab_level!(view(array, ix, iy, :), field, h, weights, zpts)
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
    weights,
    gidx;
    physical_z = false,
)

    space = axes(field)

    # When we don't have hypsography, there is no notion of "interpolating hypsography". In
    # this case, the reference vertical points coincide with the physical ones. Setting
    # physical_z = false ensures that zpts_ref = zpts
    if space.hypsography isa Spaces.Flat
        physical_z = false
    end

    # If we physical_z, we have to move the z coordinates from physical to
    # reference ones.
    if physical_z
        # We are hardcoding the transformation from Hypsography.LinearAdaption
        space.hypsography isa Hypsography.LinearAdaption ||
            error("Cannot interpolate $(space.hypsography) hypsography")

        z_surface = interpolate_slab(
            space.hypsography.surface,
            Fields.SlabIndex(nothing, gidx),
            weights,
        )
        z_top = Spaces.vertical_topology(space).mesh.domain.coord_max.z
        zpts_ref = [
            Geometry.ZPoint((z.z - z_surface) / (1 - z_surface / z_top)) for
            z in zpts
        ]
    else
        zpts_ref = zpts
    end

    output_array = zeros(Spaces.undertype(space), length(zpts))

    interpolate_slab_level!(output_array, field, gidx, weights, zpts_ref)

    return output_array
end
