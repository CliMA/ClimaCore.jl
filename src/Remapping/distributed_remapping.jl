# This file provides function that perform Lagrange interpolation of fields onto pre-defined
# grids of points. These functions are particularly useful to map the computational grid
# onto lat-long-z/xyz grids.
#
# We perform interpolation as described in Berrut2004. In most simulations, the points where
# to interpolate is fixed and the field changes. So, we design our functions with the
# assumption that we have a fixed `remapping` matrix (computed from the evaluation points
# and the nodes), and a variable field. The `remapping` is essentially Equation (4.2) in
# Berrut2004 without the fields f_j:
#
# interpolated(x) = sum_j [w_j/(x - x_j) f_j] / sum_j [w_j / (x - x_j)]
#
# with j = 1, ..., n_nodes, and w weights defined in Berrut2004.
#
# Fixed x, we can write this as a vector-vector multiplication
#
# V_j = [w_j/(x - x_j) / sum_k [w_k / (x - x_k)]]
#
# interpolated(x) = sum_j v_j f_j          (*)
#
# We could take this one step further, and evaluate multiple points at the same time. Now,
# we have a matrix M, with each row being V in the formula above for the corresponding x.
# (This is not done in the current implementation)
#
# As we can see from (*), v_j are fixed as long as we always interpolate on the same points.
# So, it is convenient to precompute V and store it somewhere. The challenge in all of this
# is for distributed runs, where each process only contains a subset of all the points.
#
# Here, we define a `Remapper` object where we store the information needed to perform the
# interpolation. A `Remapper` contains the list of points that it will interpolate on, the
# coefficients for the interpolation, and other technical information. Each MPI process has
# its own remapper. When we interpolate, we prepare a big matrix with all the points we want
# to interpolate on, then, each process computes the points that belong to that process an
# writes them into the matrix. The root process will reduce all the various intermediate
# steps by summing all the matrices and returning a matrix with the result.
#
# To keep track of which points belong to each process, we define bitmasks that are
# process-local. The bitmask is also important because it allows us to reconstruct the
# desired output shape. This is because in the process we lose spatial information and all
# the target points are in a flat list. This is a necessary evil because general simulations
# spread points across processes in a non uniform way (think: cubed sphere).
#
# By virtue of our parallelization strategy, we can guarantee that columns are in the same
# process. So, we split the target interpolation points into horizontal and vertical, and
# focus mostly on the horizontal ones. Vertical points are handled by increasing the rank of
# the intermediate matrices by one (ie, each "point" becomes a vector).

"""
    containing_pid(target_point, topology)

Return the process id that contains the `target_point` in the given `topology`.
"""
function containing_pid(
    target_point::P,
    topology::T,
) where {
    P <: Union{Geometry.LatLongPoint, Geometry.XYPoint},
    T <: Topologies.Topology2D,
}
    containing_elem = Meshes.containing_element(topology.mesh, target_point)
    gidx = topology.orderindex[containing_elem]
    return topology.elempid[gidx]
end

"""
    target_hcoords_pid_bitmask(target_hcoords, topology, pid)

Return a bitmask for points in `target_hcoords` in the given (horizontal) `topology` that
belong to the process with process number `pid`.

This mask can be used to extract the `target_hcoords` relevant to the given `pid`. The mask
is the same shape and size as the input `target_hcoords`, which makes it particularly useful.
"""
function target_hcoords_pid_bitmask(target_hcoords, topology, pid)
    pid_hcoord = hcoord -> containing_pid(hcoord, topology)
    return pid_hcoord.(target_hcoords) .== pid
end

struct Remapper{
    T <: ClimaComms.AbstractCommsContext,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6 <: Spaces.AbstractSpace,
}
    comms_ctx::T

    # Target points that are on the process where this object is defined
    # local_target_hcoords is stored as a 1D array (we store it as 1D array because in
    # general there is no structure to this object, especially for cubed sphere, which have
    # points spread all over the place)
    local_target_hcoords::T1

    # Target coordinates in the vertical direction. zcoords are the same for all the processes
    target_zcoords::T2

    # bitmask that identifies which target points are on process where the object is
    # defined. In general, local_target_points_bitmask is a 2D matrix and it is used to fill
    # the correct values in the final output. Every time we find a 1, we are going to stick
    # the vertical column of the interpolated data.
    local_target_hcoords_bitmask::T3

    # Coefficients (WI1[, WI2]) used for the interpolation. Array of tuples.
    interpolation_coeffs::T4

    # Local indices the element of each local_target_hcoords in the given topology
    local_indices::T5

    # Space over which the remapper is defined
    space::T6
end

"""
   Remapper(target_hcoords, target_zcoords, space)

Return a `Remapper` responsible for interpolating any `Field` defined on the given `space`
to the Cartesian product of `target_hcoords` with `target_zcoords`.

`target_zcoords` can be `nothing` for interpolation on horizontal spaces.

The `Remapper` is designed to not be tied to any particular `Field`. You can use the same
`Remapper` for any `Field` as long as they are all defined on the same `topology`.

`Remapper` is the main argument to the `interpolate` function.

"""
function Remapper(target_hcoords, target_zcoords, space)

    comms_ctx = ClimaComms.context(space)
    pid = ClimaComms.mypid(comms_ctx)
    FT = Spaces.undertype(space)
    horizontal_topology = Spaces.topology(space)

    is_1d = typeof(horizontal_topology) <: Topologies.IntervalTopology

    # For IntervalTopology, all the points belong to the same process and there's no notion
    # of containing pid
    if is_1d
        # a .== a is an easy way to make a bitmask of the same shape as `a` filled with true
        local_target_hcoords_bitmask = target_hcoords .== target_hcoords
    else
        local_target_hcoords_bitmask =
            target_hcoords_pid_bitmask(target_hcoords, horizontal_topology, pid)
    end

    # Extract the coordinate we own (as a MPI process). This will flatten the matrix.
    local_target_hcoords = target_hcoords[local_target_hcoords_bitmask]

    horz_mesh = horizontal_topology.mesh

    interpolation_coeffs = map(local_target_hcoords) do hcoord
        quad = Spaces.quadrature_style(space)
        quad_points, _ = Spaces.Quadratures.quadrature_points(FT, quad)
        return interpolation_weights(horz_mesh, hcoord, quad_points)
    end

    # For 2D meshes, we have a notion of local and global indices. This is not the case for
    # 1D meshes, which are much simpler. For 1D meshes, the "index" is the same as the
    # element number, for 2D ones, we have to do some work.
    if is_1d
        local_indices = map(local_target_hcoords) do hcoord
            return Meshes.containing_element(horz_mesh, hcoord)
        end
    else
        # We need to obtain the local index from the global, so we prepare a lookup table
        global_elem_lidx = Dict{Int, Int}() # inverse of local_elem_gidx: lidx = global_elem_lidx[gidx]
        for (lidx, gidx) in enumerate(horizontal_topology.local_elem_gidx)
            global_elem_lidx[gidx] = lidx
        end

        local_indices = map(local_target_hcoords) do hcoord
            helem = Meshes.containing_element(horz_mesh, hcoord)
            return global_elem_lidx[horizontal_topology.orderindex[helem]]
        end
    end

    # We represent interpolation onto an horizontal slab as an empty list of zcoords
    isnothing(target_zcoords) && (target_zcoords = [])

    return Remapper(
        comms_ctx,
        local_target_hcoords,
        target_zcoords,
        local_target_hcoords_bitmask,
        interpolation_coeffs,
        local_indices,
        space,
    )
end


"""
   interpolate(remapper, field)

Interpolate the given `field` as prescribed by `remapper`.

Example
========

Given `field1`,`field2`, two `Field` defined on a cubed sphere.

```julia
longpts = range(-180.0, 180.0, 21)
latpts = range(-80.0, 80.0, 21)
zpts = range(0.0, 1000.0, 21)

hcoords = [Geometry.LatLongPoint(lat, long) for long in longpts, lat in latpts]
zcoords = [Geometry.ZPoint(z) for z in zpts]

space = axes(field1)

remapper = Remapper(hcoords, zcoords, space)

int1 = interpolate(remapper, field1)
int2 = interpolate(remapper, field2)

```
"""
function interpolate(remapper::Remapper, field::T) where {T <: Fields.Field}

    axes(field) == remapper.space ||
        error("Field is defined on a different space than remapper")

    FT = eltype(field)

    if length(remapper.target_zcoords) == 0
        out_local_array = zeros(FT, size(remapper.local_target_hcoords_bitmask))
        interpolated_values = [
            interpolate_slab(field, Fields.SlabIndex(nothing, gidx), weights) for (gidx, weights) in
            zip(remapper.local_indices, remapper.interpolation_coeffs)
        ]

        # out_local_array[remapper.local_target_hcoords_bitmask] returns a view on space we
        # want to write on
        out_local_array[remapper.local_target_hcoords_bitmask] .=
            interpolated_values
    else
        # We have to add one extra dimension with respect to the bitmask because we are going to store
        # the values for the columns
        out_local_array = zeros(
            FT,
            (
                size(remapper.local_target_hcoords_bitmask)...,
                length(remapper.target_zcoords),
            ),
        )

        # interpolated_values is an array of arrays properly ordered according to the bitmask

        # `stack` stacks along the first dimension, so we need to transpose (') to make sure
        # that we have the correct shape
        interpolated_values =
            stack(
                interpolate_column(
                    field,
                    remapper.target_zcoords,
                    weights,
                    gidx,
                ) for (gidx, weights) in
                zip(remapper.local_indices, remapper.interpolation_coeffs)
            )'

        # out_local_array[remapper.local_target_hcoords_bitmask, :] returns a view on space we
        # want to write on
        out_local_array[remapper.local_target_hcoords_bitmask, :] .=
            interpolated_values
    end

    # Next, we have to send all the out_arrays to root and sum them up to obtain the final
    # answer. Only the root will return something
    return ClimaComms.reduce(remapper.comms_ctx, out_local_array, +)
end
