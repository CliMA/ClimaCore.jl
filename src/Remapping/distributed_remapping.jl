# This file provides functions that perform Lagrange interpolation of fields onto pre-defined
# grids of points. These functions are particularly useful to map the computational grid
# onto lat-long-z/xyz grids.
#
# We perform interpolation as described in Berrut2004. Let us start by focusing on the 1D case.
#
# In most simulations, the points where to interpolate are fixed and the field changes. So,
# we design our functions with the assumption that we have a fixed `remapping` matrix
# (computed from the evaluation points and the nodes), and a variable field. The `remapping`
# matrix is essentially Equation (4.2) in Berrut2004:
#
# interpolated(x) = sum_j [w_j/(x - x_j) f_j] / sum_j [w_j / (x - x_j)]
#
# with j = 1, ..., n_nodes, and w weights defined in Berrut2004.
#
# Fixed x, we can write this as a vector-vector multiplication
#
# V_j = [w_j/(x - x_j) / sum_k [w_k / (x - x_k)]]
#
# V_j = N(x) * w_j/(x - x_j)    where    N(x) = 1 / sum_k [w_k / (x - x_k)]
#
# interpolated(x) = sum_j V_j f_j          (*)
#
#
# In the 1D case, this is nice and efficient. Now, let us move to the 2D case. In this case,
# we will have two weights V1 and V2 and
#
# interpolated(x) = sum_i sum_j V1_i V2_j f_ij
#
# In other words, this is
#
# interpolated(x) = V1^T * F * V2
#
# where V1 and V2 depend on x.
#
# As we can see, the interpolation matrices are fixed, so we should compute them once and
# store them. The challenge in all of this is for distributed runs, where each process only
# contains a subset of all the points.
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
#
# We perform linear vertical interpolation. So, for each point, we need to find vertical
# which element contains that point and what are the neighboring elements needed to perform
# linear interpolation. Once again, this only depends on the target grid, which is fixed in
# a `Remapper`. For this reason, we can compute and cache the list of elements that contain
# and are neighboring each target point. Similarly, we can also compute the vertical
# interpolation weights once for all and store them.
#
# So, the `Remapper` will contain:
# - The target horizontal and vertical points H and V. We interpolate over HxV.
# - The horizontal interpolation weights. One or two (depending on the dimension of the
#   horizontal space) per horizontal point.
# - The bitmask that determines which points belong to the process
# - The list of indices in the horizontal mesh that belong to the process
# - The space where the remapper is defined
# - The vertical interpolation weights, a vector two numbers per each vertical point.
# - The bounding indices that identify the vertical elements neighboring all the target
#   vertical points.
# - Some scratch spaces, one where to save field values F, one where to save the
#   process-local interpolated points as a linear array, and one where to save the
#   process-local interpolate points in the correct shape with respect to the global
#   interpolation (and where to collect results)
#
# For GPU runs, we store all the vectors as CuArrays on the GPU.

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
    CC <: ClimaComms.AbstractCommsContext,
    SPACE <: Spaces.AbstractSpace,
    T1 <: AbstractArray,
    TARG_Z <: Union{Nothing, AA1} where {AA1 <: AbstractArray},
    T3 <: AbstractArray,
    T4 <: Tuple,
    T5 <: AbstractArray,
    VERT_W <: Union{Nothing, AA2} where {AA2 <: AbstractArray},
    VERT_IND <: Union{Nothing, AA3} where {AA3 <: AbstractArray},
    T8 <: AbstractArray,
    T9 <: AbstractArray,
    T10 <: AbstractArray,
}

    comms_ctx::CC

    # Space over which the remapper is defined
    space::SPACE

    # Target points that are on the process where this object is defined
    # local_target_hcoords is stored as a 1D array (we store it as 1D array because in
    # general there is no structure to this object, especially for cubed sphere, which have
    # points spread all over the place)
    local_target_hcoords::T1

    # Target coordinates in the vertical direction. zcoords are the same for all the processes.
    # target_zcoords are always assumed to be "reference z"s.
    target_zcoords::TARG_Z

    # bitmask that identifies which target points are on process where the object is
    # defined. In general, local_target_points_bitmask is a 2D matrix and it is used to fill
    # the correct values in the final output. Every time we find a 1, we are going to stick
    # the vertical column of the interpolated data.
    local_target_hcoords_bitmask::T3

    # Tuple of arrays of weights that performs horizontal interpolation (fixed the
    # horizontal and target spaces). It contains 1 element for grids with one horizontal
    # dimension, and 2 elements for grids with two horizontal dimensions.
    local_horiz_interpolation_weights::T4

    # Local indices the element of each local_target_hcoords in the given topology. This is
    # a linear array with the same length as local_target_hcoords.
    local_horiz_indices::T5

    # Given the target_zcoords, vert_reference_coordinates contains the reference coordinate
    # in the element. For center spaces, the reference coordinate is shifted to be in (0, 1)
    # when the point is in the lower half of the cell, and in (-1, 0) when it is in the
    # upper half. This shift is needed to directly use the reference coordinate in linear
    # vertical interpolation. Array of tuples or Nothing.
    vert_interpolation_weights::VERT_W

    # Given the target_zcoords, vert_bounding_indices contain the vertical indices of the
    # neighboring elements that are required for vertical interpolation.
    # Array of tuples or Nothing.
    vert_bounding_indices::VERT_IND

    # Scratch space where we save the process-local interpolated values. We keep overwriting
    # this to avoid extra allocations. This is a linear array with the same length as
    # local_horiz_indices.
    _local_interpolated_values::T8

    # Scratch space where we save the process-local field value. We keep overwriting this to
    # avoid extra allocations. Ideally, we wouldn't need this and we would use views for
    # everything. This has dimensions (Nq) or (Nq, Nq) depending if the horizontal space is
    # 1D or 2D.
    _field_values::T9

    # Storage area where the interpolated values are saved. This is meaningful only for the
    # root process and gets filled by a interpolate call. This has dimensions (H, V), where
    # H is the size of target_hcoords and V of target_zcoords. In other words, this is the
    # expected output array.
    _interpolated_values::T10
end

"""
   Remapper(space, target_hcoords, target_zcoords)
   Remapper(space, target_hcoords)

Return a `Remapper` responsible for interpolating any `Field` defined on the given `space`
to the Cartesian product of `target_hcoords` with `target_zcoords`.

`target_zcoords` can be `nothing` for interpolation on horizontal spaces.

The `Remapper` is designed to not be tied to any particular `Field`. You can use the same
`Remapper` for any `Field` as long as they are all defined on the same `topology`.

`Remapper` is the main argument to the `interpolate` function.
"""
function Remapper(
    space::Spaces.AbstractSpace,
    target_hcoords::AbstractArray,
    target_zcoords::Union{AbstractArray, Nothing},
)

    comms_ctx = ClimaComms.context(space)
    pid = ClimaComms.mypid(comms_ctx)
    FT = Spaces.undertype(space)
    ArrayType = ClimaComms.array_type(space)
    horizontal_topology = Spaces.topology(space)
    horizontal_mesh = horizontal_topology.mesh

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

    # Extract the coordinates we own (as an MPI process). This will flatten the matrix.
    local_target_hcoords = target_hcoords[local_target_hcoords_bitmask]

    # Compute interpolation matrices
    helems =
        Meshes.containing_element.(Ref(horizontal_mesh), local_target_hcoords)
    ξs_combined =
        Meshes.reference_coordinates.(
            Ref(horizontal_mesh),
            helems,
            local_target_hcoords,
        )
    num_hdims = length(ξs_combined[begin])
    # ξs is a Vector of SVector{1, Float64} or SVector{2, Float64}
    # Here we split the two dimensions because we want to compute the two interpolation matrices.
    ξs_split = Tuple([ξ[i] for ξ in ξs_combined] for i in 1:num_hdims)

    # Compute the interpolation matrices
    quad = Spaces.quadrature_style(space)
    quad_points, _ = Quadratures.quadrature_points(FT, quad)
    Nq = Quadratures.degrees_of_freedom(quad)

    local_horiz_interpolation_weights = map(
        ξs -> ArrayType(Quadratures.interpolation_matrix(ξs, quad_points)),
        ξs_split,
    )

    # For 2D meshes, we have a notion of local and global indices. This is not the case for
    # 1D meshes, which are much simpler. For 1D meshes, the "index" is the same as the
    # element number, for 2D ones, we have to do some work.
    if is_1d
        local_horiz_indices =
            Meshes.containing_element.(
                Ref(horizontal_mesh),
                local_target_hcoords,
            )
    else
        # We need to obtain the local index from the global, so we prepare a lookup table
        global_elem_lidx = Dict{Int, Int}() # inverse of local_elem_gidx: lidx = global_elem_lidx[gidx]
        for (lidx, gidx) in enumerate(horizontal_topology.local_elem_gidx)
            global_elem_lidx[gidx] = lidx
        end

        local_horiz_indices = map(local_target_hcoords) do hcoord
            helem = Meshes.containing_element(horizontal_mesh, hcoord)
            return global_elem_lidx[horizontal_topology.orderindex[helem]]
        end
    end

    local_horiz_indices = ArrayType(local_horiz_indices)

    field_values_size = ntuple(_ -> Nq, num_hdims)
    field_values = ArrayType(zeros(FT, field_values_size))

    # We represent interpolation onto an horizontal slab as an empty list of zcoords
    if isnothing(target_zcoords) || isempty(target_zcoords)
        target_zcoords = nothing
        vert_interpolation_weights = nothing
        vert_bounding_indices = nothing
        local_interpolated_values =
            ArrayType(zeros(FT, size(local_horiz_indices)))
        interpolated_values =
            ArrayType(zeros(FT, size(local_target_hcoords_bitmask)))
    else
        vert_interpolation_weights =
            ArrayType(vertical_interpolation_weights(space, target_zcoords))
        vert_bounding_indices =
            ArrayType(vertical_bounding_indices(space, target_zcoords))

        # We have to add one extra dimension with respect to the bitmask/local_horiz_indices
        # because we are going to store the values for the columns
        local_interpolated_values = ArrayType(
            zeros(FT, (size(local_horiz_indices)..., length(target_zcoords))),
        )
        interpolated_values = ArrayType(
            zeros(
                FT,
                (size(local_target_hcoords_bitmask)..., length(target_zcoords)),
            ),
        )
    end

    return Remapper(
        comms_ctx,
        space,
        local_target_hcoords,
        target_zcoords,
        local_target_hcoords_bitmask,
        local_horiz_interpolation_weights,
        local_horiz_indices,
        vert_interpolation_weights,
        vert_bounding_indices,
        local_interpolated_values,
        field_values,
        interpolated_values,
    )
end

Remapper(space::Spaces.AbstractSpace, target_hcoords::AbstractArray) =
    Remapper(space, target_hcoords, nothing)

"""
    _set_interpolated_values!(remapper, field)

Change the local state of `remapper` by performing interpolation of `Fields` on the vertical
and horizontal points.
"""
function _set_interpolated_values!(remapper::Remapper, field::Fields.Field)
    _set_interpolated_values!(
        remapper._local_interpolated_values,
        field,
        remapper._field_values,
        remapper.local_horiz_indices,
        remapper.local_horiz_interpolation_weights,
        remapper.vert_interpolation_weights,
        remapper.vert_bounding_indices,
    )
end

function set_interpolated_values_cpu_kernel!(
    out::AbstractArray,
    field::Fields.Field,
    (I1, I2)::NTuple{2},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    scratch_field_values,
    field_values,
)
    space = axes(field)
    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quad)
    field_values = Fields.field_values(field)

    # Reading values from field_values is expensive, so we try to limit the number of reads. We can do
    # this because multiple target points might be all contained in the same element.
    prev_vindex, prev_lidx = -1, -1
    @inbounds for (vindex, (A, B)) in enumerate(vert_interpolation_weights)
        (v_lo, v_hi) = vert_bounding_indices[vindex]
        for (out_index, h) in enumerate(local_horiz_indices)
            # If we are no longer in the same element, read the field values again
            if prev_lidx != h || prev_vindex != vindex
                for j in 1:Nq, i in 1:Nq
                    scratch_field_values[i, j] = (
                        A * field_values[i, j, nothing, v_lo, h] +
                        B * field_values[i, j, nothing, v_hi, h]
                    )
                end
                prev_vindex, prev_lidx = vindex, h
            end

            tmp = zero(FT)

            for j in 1:Nq, i in 1:Nq
                tmp +=
                    I1[out_index, i] *
                    I2[out_index, j] *
                    scratch_field_values[i, j]
            end
            out[out_index, vindex] = tmp
        end
    end
end

function set_interpolated_values_kernel!(
    out::AbstractArray,
    (I1, I2)::NTuple{2},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
)

    hindex = blockIdx().x
    vindex = threadIdx().x
    index = vindex + (hindex - 1) * blockDim().x
    index > length(out) && return nothing

    h = local_horiz_indices[hindex]
    v_lo, v_hi = vert_bounding_indices[vindex]
    A, B = vert_interpolation_weights[vindex]

    _, Nq = size(I1)

    out[hindex, vindex] = 0
    for j in 1:Nq, i in 1:Nq
        out[hindex, vindex] +=
            I1[hindex, i] *
            I2[hindex, j] *
            (
                A * field_values[i, j, nothing, v_lo, h] +
                B * field_values[i, j, nothing, v_hi, h]
            )
    end

    return nothing
end

function set_interpolated_values_kernel!(
    out::AbstractArray,
    (I,)::NTuple{1},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    field_values,
)

    hindex = blockIdx().x
    vindex = threadIdx().x
    index = vindex + (hindex - 1) * blockDim().x
    index > length(out) && return nothing

    h = local_horiz_indices[hindex]
    v_lo, v_hi = vert_bounding_indices[vindex]
    A, B = vert_interpolation_weights[vindex]

    _, Nq = size(I)

    out[hindex, vindex] = 0
    for i in 1:Nq
        out[hindex, vindex] +=
            I[hindex, i] * (
                A * field_values[i, nothing, nothing, v_lo, h] +
                B * field_values[i, nothing, nothing, v_hi, h]
            )
    end

    return nothing
end

function set_interpolated_values_cpu_kernel!(
    out::AbstractArray,
    field::Fields.Field,
    (I,)::NTuple{1},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    scratch_field_values,
    field_values,
)
    space = axes(field)
    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quad)
    field_values = Fields.field_values(field)

    # Reading values from field_values is expensive, so we try to limit the number of reads. We can do
    # this because multiple target points might be all contained in the same element.
    prev_vindex, prev_lidx = -1, -1
    @inbounds for (vindex, (A, B)) in enumerate(vert_interpolation_weights)
        (v_lo, v_hi) = vert_bounding_indices[vindex]

        for (out_index, h) in enumerate(local_horiz_indices)
            # If we are no longer in the same element, read the field values again
            if prev_lidx != h || prev_vindex != vindex
                for i in 1:Nq
                    scratch_field_values[i] = (
                        A * field_values[i, nothing, nothing, v_lo, h] +
                        B * field_values[i, nothing, nothing, v_hi, h]
                    )
                end
                prev_vindex, prev_lidx = vindex, h
            end

            tmp = zero(FT)

            for i in 1:Nq
                tmp += I[out_index, i] * scratch_field_values[i]
            end
            out[out_index, vindex] = tmp
        end
    end
end

function _set_interpolated_values!(
    out::AbstractArray,
    field::Fields.Field,
    scratch_field_values,
    local_horiz_indices,
    interpolation_matrix,
    vert_interpolation_weights::AbstractArray,
    vert_bounding_indices::AbstractArray,
)

    field_values = Fields.field_values(field)

    if ClimaComms.device(field) isa ClimaComms.CUDADevice
        nblocks, _ = size(interpolation_matrix[1])
        nthreads = length(vert_interpolation_weights)
        @cuda always_inline = true threads = (nthreads) blocks = (nblocks) set_interpolated_values_kernel!(
            out,
            interpolation_matrix,
            local_horiz_indices,
            vert_interpolation_weights,
            vert_bounding_indices,
            field_values,
        )
    else
        set_interpolated_values_cpu_kernel!(
            out,
            field,
            interpolation_matrix,
            local_horiz_indices,
            vert_interpolation_weights,
            vert_bounding_indices,
            scratch_field_values,
            field_values,
        )
    end
end

function _set_interpolated_values!(
    out::AbstractArray,
    field::Fields.Field,
    _scratch_field_values,
    local_horiz_indices,
    local_horiz_interpolation_weights,
    ::Nothing,
    ::Nothing,
)

    space = axes(field)
    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quad)
    field_values = Fields.field_values(field)

    hdims = length(local_horiz_interpolation_weights)
    hdims in (1, 2) || error("Cannot handle $hdims horizontal dimensions")

    if ClimaComms.device(space) isa ClimaComms.CUDADevice
        nitems = length(out)
        nthreads, nblocks = Topologies._configure_threadblock(nitems)
        @cuda always_inline = true threads = (nthreads) blocks = (nblocks) set_interpolated_values_kernel!(
            out,
            local_horiz_interpolation_weights,
            local_horiz_indices,
            field_values,
        )
    else
        for (out_index, h) in enumerate(local_horiz_indices)
            out[out_index] = zero(FT)
            if hdims == 2
                for j in 1:Nq, i in 1:Nq
                    out[out_index] +=
                        local_horiz_interpolation_weights[1][out_index, i] *
                        local_horiz_interpolation_weights[2][out_index, j] *
                        field_values[i, j, nothing, nothing, h]
                end
            elseif hdims == 1
                for i in 1:Nq
                    out[out_index] +=
                        local_horiz_interpolation_weights[1][out_index, i] *
                        field_values[i, nothing, nothing, nothing, h]
                end
            end
        end
    end
end

function set_interpolated_values_kernel!(
    out,
    (I1, I2)::NTuple{2},
    local_horiz_indices,
    field_values,
)

    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    index > length(out) && return nothing

    h = local_horiz_indices[index]
    _, Nq = size(I1)

    out[index] = 0
    for j in 1:Nq, i in 1:Nq
        out[index] +=
            I1[index, i] *
            I2[index, j] *
            field_values[i, j, nothing, nothing, h]
    end

    return nothing
end

function set_interpolated_values_kernel!(
    out::AbstractArray,
    (I,)::NTuple{1},
    local_horiz_indices,
    field_values,
)

    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    index > length(out) && return nothing

    h = local_horiz_indices[index]
    _, Nq = size(I)

    out[index] = 0
    for i in 1:Nq
        out[index] +=
            I[index, i] * field_values[i, nothing, nothing, nothing, h]
    end

    return nothing
end

"""
    _apply_mpi_bitmask!(remapper::Remapper)

Change to local (private) state of the `remapper` by applying the MPI bitmask and reconstructing
the correct shape for the interpolated values.

Internally, `remapper` performs interpolation on a flat list of points, this function moves points
around according to MPI-ownership and the expected output shape.
"""
function _apply_mpi_bitmask!(remapper::Remapper)
    if isnothing(remapper.target_zcoords)
        # _interpolated_values[remapper.local_target_hcoords_bitmask] returns a view on
        # space we want to write on
        remapper._interpolated_values[remapper.local_target_hcoords_bitmask] .=
            remapper._local_interpolated_values
    else
        # interpolated_values is an array of arrays properly ordered according to the bitmask

        # _interpolated_values[remapper.local_target_hcoords_bitmask, :] returns a
        # view on space we want to write on
        remapper._interpolated_values[
            remapper.local_target_hcoords_bitmask,
            :,
        ] .= remapper._local_interpolated_values
    end
end

"""
    _reset_interpolated_values!(remapper::Remapper)

Reset the local (private) state in `remapper`. This function has to be called before performing
interpolation.
"""
function _reset_interpolated_values!(remapper::Remapper)
    fill!(remapper._interpolated_values, 0)
end

"""
    _collect_and_return_interpolated_values!(remapper::Remapper)

Perform an MPI call to aggregate the interpolated points from all the MPI processes and save
the result in the local state of the `remapper`. Only the root process will return the
interpolated data.

`_collect_and_return_interpolated_values!` is type-unstable and allocates new return arrays.
"""
function _collect_and_return_interpolated_values!(remapper::Remapper)
    ClimaComms.reduce!(remapper.comms_ctx, remapper._interpolated_values, +)
    return ClimaComms.iamroot(remapper.comms_ctx) ?
           Array(remapper._interpolated_values) : nothing
end

function _collect_interpolated_values!(dest, remapper::Remapper)
    # MPI.reduce! seems to behave nicely with respect to CPU/GPU. In particular,
    # if the destination is on the CPU, but the source is on the GPU, the values
    # are automatically moved.
    ClimaComms.reduce!(
        remapper.comms_ctx,
        remapper._interpolated_values,
        dest,
        +,
    )
    return nothing
end

"""
   interpolate(remapper::Remapper, field)
   interpolate!(dest, remapper::Remapper, field)

Interpolate the given `field` as prescribed by `remapper`.

This call mutates the internal (private) state of the `remapper`.

Horizontally, interpolation is performed with the barycentric formula in
[Berrut2004](@cite), equation (3.2). Vertical interpolation is linear except
in the boundary elements where it is 0th order.

`interpolate!` writes the output to the given `dest`iniation. `dest` is expected
to be defined on the root process and to be `nothing` for the other processes.

Note: `interpolate` allocates new arrays and has some internal type-instability,
`interpolate!` is non-allocating and type-stable.

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

remapper = Remapper(space, hcoords, zcoords)

int1 = interpolate(remapper, field1)
int2 = interpolate(remapper, field2)
```
"""
function interpolate(remapper::Remapper, field::T) where {T <: Fields.Field}

    axes(field) == remapper.space ||
        error("Field is defined on a different space than remapper")

    # Reset interpolated_values. This is needed because we collect distributed results with
    # a + reduction.
    _reset_interpolated_values!(remapper)
    # Perform the interpolations (horizontal and vertical)
    _set_interpolated_values!(remapper, field)
    # Reshape the output so that it is a nice grid.
    _apply_mpi_bitmask!(remapper)
    # Finally, we have to send all the _interpolated_values to root and sum them up to
    # obtain the final answer. Only the root will contain something useful. This also moves
    # the data off the GPU
    return _collect_and_return_interpolated_values!(remapper)
end

function interpolate!(
    dest::Union{Nothing, <:AbstractArray},
    remapper::Remapper,
    field::T,
) where {T <: Fields.Field}

    axes(field) == remapper.space ||
        error("Field is defined on a different space than remapper")

    if !isnothing(dest)
        # !isnothing(dest) means that this is the root process, in this case, the size have
        # to match
        size(dest) == size(remapper._interpolated_values) || error(
            "Destination array is not compatible with remapper (size mismatch)",
        )
    end

    # Reset interpolated_values. This is needed because we collect distributed results with
    # a + reduction.
    _reset_interpolated_values!(remapper)
    # Perform the interpolations (horizontal and vertical)
    _set_interpolated_values!(remapper, field)
    # Reshape the output so that it is a nice grid.
    _apply_mpi_bitmask!(remapper)
    # Finally, we have to send all the _interpolated_values to root and sum them
    # up to obtain the final answer. This also moves the data off the GPU. The
    # output is written to the given destination
    _collect_interpolated_values!(dest, remapper)
    return nothing
end
