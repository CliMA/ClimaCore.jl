function interpolate_slab(
    field::Fields.Field,
    slabidx::Fields.SlabIndex,
    (I1,)::Tuple{<:AbstractArray},
)
    space = axes(field)
    x = zero(eltype(field))
    QS = Spaces.quadrature_style(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(QS)

    for i in 1:Nq
        ij = CartesianIndex((i,))
        x += I1[i] * Operators.get_node(space, field, ij, slabidx)
    end
    return x
end

function interpolate_slab(
    field::Fields.Field,
    slabidx::Fields.SlabIndex,
    (I1, I2)::Tuple{<:AbstractArray, <:AbstractArray},
)
    space = axes(field)
    x = zero(eltype(field))
    QS = Spaces.quadrature_style(space)
    Nq = Spaces.Quadratures.degrees_of_freedom(QS)

    for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        x += I1[i] * I2[j] * Operators.get_node(space, field, ij, slabidx)
    end
    return x
end

function interpolate_slab_level(
    field::Fields.Field,
    h::Integer,
    Is::Tuple,
    zcoord,
)
    space = axes(field)
    vert_topology = Spaces.vertical_topology(space)
    vert_mesh = vert_topology.mesh
    Nz = Spaces.nlevels(space)

    velem = Meshes.containing_element(vert_mesh, zcoord)
    ξ3, = Meshes.reference_coordinates(vert_mesh, velem, zcoord)
    if space isa Spaces.FaceExtrudedFiniteDifferenceSpace
        v_lo = velem - half
        v_hi = velem + half
    elseif space isa Spaces.CenterExtrudedFiniteDifferenceSpace
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
    end
    f_lo = interpolate_slab(field, Fields.SlabIndex(v_lo, h), Is)
    f_hi = interpolate_slab(field, Fields.SlabIndex(v_hi, h), Is)
    return ((1 - ξ3) * f_lo + (1 + ξ3) * f_hi) / 2
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
        helem = Meshes.containing_element(horz_mesh, hcoord)
        hcoord = xcoord
        quad = Spaces.quadrature_style(space)
        quad_points, _ = Spaces.Quadratures.quadrature_points(FT, quad)
        weights = interpolation_weights(horz_mesh, hcoord, quad_points)
        h = helem

        for (iz, zcoord) in enumerate(zpts)
            array[ix, iz] = interpolate_slab_level(field, h, weights, zcoord)
        end
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
        helem = Meshes.containing_element(horz_mesh, hcoord)
        hcoord = Geometry.product_coordinates(xcoord, ycoord)
        quad = Spaces.quadrature_style(space)
        quad_points, _ = Spaces.Quadratures.quadrature_points(FT, quad)
        weights = interpolation_weights(horz_mesh, hcoord, quad_points)
        gidx = horz_topology.orderindex[helem]
        h = gidx

        for (iz, zcoord) in enumerate(zpts)
            array[ix, iy, iz] =
                interpolate_slab_level(field, h, weights, zcoord)
        end
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
"""
function interpolate_column(
    field::Fields.ExtrudedFiniteDifferenceField,
    zpts,
    weights,
    gidx,
)
    return [interpolate_slab_level(field, gidx, weights, z) for z in zpts]
end
