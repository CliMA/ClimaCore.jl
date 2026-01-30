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
# Horizontal and vertical interpolation can be switch off, so that we interpolate purely
# horizontal/vertical Fields.
#
# To process multiple Fields at the same time, some of the scratch spaces gain an extra
# dimension (buffer_length). With this extra dimension, we can batch the work and process up
# to buffer_length fields at the same time. This reduces the number of kernel launches and
# MPI calls.
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


# TODO: Define an inner construct and restrict types, as was done in
#       https://github.com/CliMA/RRTMGP.jl/pull/352
#       to avoid potential compilation issues.
struct Remapper{
    CC <: ClimaComms.AbstractCommsContext,
    SPACE <: Spaces.AbstractSpace,
    T1, # <: Union{AbstractArray, Nothing},
    TARG_Z <: Union{Nothing, AA1} where {AA1 <: AbstractArray},
    T3, # <: Union{AbstractArray, Nothing},
    T4, # <: Union{Tuple, Nothing},
    T5, # <: Union{AbstractArray, Nothing},
    VERT_W <: Union{Nothing, AA2} where {AA2 <: AbstractArray},
    VERT_IND <: Union{Nothing, AA3} where {AA3 <: AbstractArray},
    T8, # <: AbstractArray,
    T9, # <: AbstractArray,
    T10 <: AbstractArray,
    T11 <: Union{Tuple{Colon}, Tuple{Colon, Colon}, Tuple{Colon, Colon, Colon}},
}
    # The ClimaComms context
    comms_ctx::CC

    # Space over which the remapper is defined
    space::SPACE

    # Target points that are on the process where this object is defined.
    # local_target_hcoords is stored as a 1D array (we store it as 1D array because in
    # general there is no structure to this object, especially for cubed sphere, which have
    # points spread all over the place). This is nothing when remapping purely vertical
    # spaces.
    local_target_hcoords::T1

    # Target coordinates in the vertical direction. zcoords are the same for all the
    # processes. target_zcoords are always assumed to be "reference z"s. is nothing when
    # remapping purely horizontal spaces.
    target_zcoords::TARG_Z

    # bitmask that identifies which target points are on process where the object is
    # defined. In general, local_target_points_bitmask is a 2D matrix and it is used to fill
    # the correct values in the final output. Every time we find a 1, we are going to stick
    # the vertical column of the interpolated data. This is nothing when remapping purely
    # vertical spaces.
    local_target_hcoords_bitmask::T3

    # Tuple of arrays of weights that performs horizontal interpolation (fixed the
    # horizontal and target spaces). It contains 1 element for grids with one horizontal
    # dimension, and 2 elements for grids with two horizontal dimensions. This is nothing
    # when remapping purely vertical spaces.
    local_horiz_interpolation_weights::T4

    # Local indices the element of each local_target_hcoords in the given topology. This is
    # a linear array with the same length as local_target_hcoords. This is nothing when
    # remapping purely vertical spaces.
    local_horiz_indices::T5

    # Given the target_zcoords, vert_reference_coordinates contains the reference coordinate
    # in the element. For center spaces, the reference coordinate is shifted to be in (0, 1)
    # when the point is in the lower half of the cell, and in (-1, 0) when it is in the
    # upper half. This shift is needed to directly use the reference coordinate in linear
    # vertical interpolation. Array of tuples or Nothing. This is nothing when remapping
    # purely horizontal spaces.
    vert_interpolation_weights::VERT_W

    # Given the target_zcoords, vert_bounding_indices contain the vertical indices of the
    # neighboring elements that are required for vertical interpolation. Array of tuples or
    # Nothing. This is nothing when remapping purely horizontal spaces.
    vert_bounding_indices::VERT_IND

    # Scratch space where we save the process-local interpolated values. We keep overwriting
    # this to avoid extra allocations. This is a linear array with the same length as
    # local_horiz_indices with an extra dimension of length buffer_length added.
    _local_interpolated_values::T8

    # Scratch space where we save the process-local field value. We keep overwriting this to
    # avoid extra allocations. Ideally, we wouldn't need this and we would use views for
    # everything. This has dimensions (Nq, ) or (Nq, Nq, ) depending if the horizontal space
    # is 1D or 2D. This is nothing when remapping purely vertical spaces.
    _field_values::T9

    # Storage area where the interpolated values are saved. This is meaningful only for the
    # root process and gets filled by a interpolate call. This has dimensions
    # (H, V, buffer_length), where H is the size of target_hcoords and V of target_zcoords.
    # In other words, this is the expected output array.
    _interpolated_values::T10

    # Maximum number of Fields that can be interpolated at any given time
    buffer_length::Int

    # A tuple of Colons (1, 2, or 3), used to more easily get views into arrays with unknown
    # dimension (1-3D)
    colons::T11

    # Horizontal remapping method. BilinearRemapping holds precomputed (s, t) and (i, j).
    horiz_method::AbstractRemappingMethod
end

"""
   Remapper(space, target_hcoords, target_zcoords, buffer_length = 1, horizontal_method = SpectralElementRemapping())
   Remapper(space; target_hcoords, target_zcoords, buffer_length = 1, horizontal_method = SpectralElementRemapping())
   Remapper(space, target_hcoords; buffer_length = 1, horizontal_method = SpectralElementRemapping())
   Remapper(space, target_zcoords; buffer_length = 1)

Return a `Remapper` responsible for interpolating any `Field` defined on the given `space`
to the Cartesian product of `target_hcoords` with `target_zcoords`.

`target_zcoords` can be `nothing` for interpolation on horizontal spaces. Similarly,
`target_hcoords` can be `nothing` for interpolation on vertical spaces.

The `Remapper` is designed to not be tied to any particular `Field`. You can use the same
`Remapper` for any `Field` as long as they are all defined on the same `topology`.

`Remapper` is the main argument to the `interpolate` function.

If you want to quickly remap something, you can call directly `interpolate`.

By default, [`default_target_zcoords`](@ref) [`default_target_hcoords`](@ref) are used to
determine the coordinates.

Keyword arguments
=================

`buffer_length` is size of the internal buffer in the Remapper to store intermediate values
for interpolation. Effectively, this controls how many fields can be remapped simultaneously
in `interpolate`. When more fields than `buffer_length` are passed, the remapper will batch
the work in sizes of `buffer_length`.

`horizontal_method`: `SpectralElementRemapping()` (default; uses spectral element quadrature weights)
or `BilinearRemapping()` (1D: linear on 2-point cell; 2D: bilinear on 2×2 cell).
"""
function Remapper end

# General case
#
# We have Union{AbstractArray, Nothing} because we want to allow for a single interface that
# capture every case.
Remapper(
    space::Spaces.AbstractSpace;
    target_hcoords::Union{AbstractArray, Nothing} = default_target_hcoords(
        space,
    ),
    target_zcoords::Union{AbstractArray, Nothing} = default_target_zcoords(
        space,
    ),
    buffer_length::Int = 1,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
) = _Remapper(space; target_zcoords, target_hcoords, buffer_length, horizontal_method)

# General case, everything passed as positional
Remapper(
    space::Spaces.AbstractSpace,
    target_hcoords::Union{AbstractArray, Nothing},
    target_zcoords::Union{AbstractArray, Nothing};
    buffer_length::Int = 1,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
) = _Remapper(space; target_zcoords, target_hcoords, buffer_length, horizontal_method)

# Purely vertical case (horizontal_method accepted for uniform API, ignored)
Remapper(
    space::Spaces.FiniteDifferenceSpace;
    target_zcoords::AbstractArray = default_target_zcoords(space),
    buffer_length::Int = 1,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
) = _Remapper(space; target_zcoords, target_hcoords = nothing, buffer_length)

# Purely vertical, positional
Remapper(
    space::Spaces.FiniteDifferenceSpace,
    target_zcoords::AbstractArray;
    buffer_length::Int = 1,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
) = _Remapper(space; target_zcoords, target_hcoords = nothing, buffer_length)

# Purely horizontal case
Remapper(
    space::Spaces.AbstractSpectralElementSpace;
    target_hcoords::AbstractArray = default_target_hcoords(space),
    buffer_length::Int = 1,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
) = _Remapper(
    space;
    target_zcoords = nothing,
    target_hcoords,
    buffer_length,
    horizontal_method,
)

# Purely horizontal case, positional
Remapper(
    space::Spaces.AbstractSpectralElementSpace,
    target_hcoords::AbstractArray;
    buffer_length::Int = 1,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
) = _Remapper(
    space;
    target_zcoords = nothing,
    target_hcoords,
    buffer_length,
    horizontal_method,
)

# Constructor for the case with horizontal spaces
function _Remapper(
    space::Spaces.AbstractSpace;
    target_zcoords::Union{AbstractArray, Nothing},
    target_hcoords::AbstractArray,
    buffer_length::Int = 1,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
)
    horiz_method = horizontal_method
    comms_ctx = ClimaComms.context(space)
    pid = ClimaComms.mypid(comms_ctx)
    FT = Spaces.undertype(space)
    ArrayType = ClimaComms.array_type(space)
    horizontal_topology = Spaces.topology(space)
    horizontal_mesh = horizontal_topology.mesh
    quad = Spaces.quadrature_style(space)

    space_has_mask = !(Spaces.get_mask(space) isa DataLayouts.NoMask)
    element_has_more_than_one_node = Quadratures.degrees_of_freedom(quad) > 1

    # Spectral remapping with a mask makes sense only if there is only one node
    # in the element (the code will go through, but the results will be
    # incorrect)
    if space_has_mask && element_has_more_than_one_node
        error(
            "Remapping does not support masks, unless each element contains exactly one nodal point",
        )
    end

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

    # Compute the interpolation matrices (or bilinear objects when BilinearRemapping)
    quad_points, _ = Quadratures.quadrature_points(FT, quad)
    Nq = Quadratures.degrees_of_freedom(quad)

    if horiz_method isa BilinearRemapping
        quad_pts = quad_points
        if num_hdims == 1
            # 1D: linear on 2-point cell. 
            ξ1s = ξs_split[1]
            i_arr = [clamp(searchsortedlast(quad_pts, ξ1), 1, Nq - 1) for ξ1 in ξ1s]
            s_arr = [
                (ξ1 - quad_pts[i]) / (quad_pts[i + 1] - quad_pts[i]) for
                (ξ1, i) in zip(ξ1s, i_arr)
            ]
            local_bilinear_i = ArrayType(i_arr)
            local_bilinear_s = ArrayType(s_arr)
            local_bilinear_t = local_bilinear_j = nothing
            local_horiz_interpolation_weights = nothing
        else
            # 2D: bilinear on 2×2 cell. 
            n = length(ξs_split[1])
            s_arr = Vector{FT}(undef, n)
            t_arr = Vector{FT}(undef, n)
            i_arr = Vector{Int}(undef, n)
            j_arr = Vector{Int}(undef, n)
            for (idx, (ξ1, ξ2)) in enumerate(zip(ξs_split[1], ξs_split[2]))
                i = clamp(searchsortedlast(quad_pts, ξ1), 1, Nq - 1)
                j = clamp(searchsortedlast(quad_pts, ξ2), 1, Nq - 1)
                s_arr[idx] = (ξ1 - quad_pts[i]) / (quad_pts[i + 1] - quad_pts[i])
                t_arr[idx] = (ξ2 - quad_pts[j]) / (quad_pts[j + 1] - quad_pts[j])
                i_arr[idx] = i
                j_arr[idx] = j
            end
            local_bilinear_s = ArrayType(s_arr)
            local_bilinear_t = ArrayType(t_arr)
            local_bilinear_i = ArrayType(i_arr)
            local_bilinear_j = ArrayType(j_arr)
            local_horiz_interpolation_weights = nothing  # bilinear uses horiz_method, not weights
        end
    else # SpectralElementRemapping
        local_bilinear_s = local_bilinear_t = local_bilinear_i = local_bilinear_j = nothing
        local_horiz_interpolation_weights = map(
            ξs -> ArrayType(Quadratures.interpolation_matrix(ξs, quad_points)),
            ξs_split,
        )
    end

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

    # For bilinear: 1D needs 2-point scratch; 2D needs 2×2. Spectral uses Nq×...×Nq.
    field_values_size = if horiz_method isa BilinearRemapping
        num_hdims == 1 ? (2,) : (2, 2)
    else
        ntuple(_ -> Nq, num_hdims)
    end
    field_values = ArrayType(zeros(FT, field_values_size...))

    # We represent interpolation onto an horizontal slab as an empty list of zcoords
    if isnothing(target_zcoords) || isempty(target_zcoords)
        target_zcoords = nothing
        vert_interpolation_weights = nothing
        vert_bounding_indices = nothing
        local_interpolated_values =
            ArrayType(zeros(FT, (size(local_horiz_indices)..., buffer_length)))
        interpolated_values = ArrayType(
            zeros(FT, (size(local_target_hcoords_bitmask)..., buffer_length)),
        )
        num_dims = num_hdims
    else
        vert_interpolation_weights =
            ArrayType(vertical_interpolation_weights(space, target_zcoords))
        vert_bounding_indices =
            ArrayType(vertical_bounding_indices(space, target_zcoords))

        # We have to add one extra dimension with respect to the bitmask/local_horiz_indices
        # because we are going to store the values for the columns
        local_interpolated_values = ArrayType(
            zeros(
                FT,
                (
                    size(local_horiz_indices)...,
                    length(target_zcoords),
                    buffer_length,
                ),
            ),
        )
        interpolated_values = ArrayType(
            zeros(
                FT,
                (
                    size(local_target_hcoords_bitmask)...,
                    length(target_zcoords),
                    buffer_length,
                ),
            ),
        )
        num_dims = num_hdims + 1
    end

    # We don't know how many dimensions an array might have, so we define a colons object
    # that we can use to index with array[colons...]

    colons = ntuple(_ -> Colon(), num_dims)

    # Reconstruct BilinearRemapping with computed arrays to preserve interface BilinearRemapping().
    if horiz_method isa BilinearRemapping
        horiz_method = BilinearRemapping(
            local_bilinear_s,
            local_bilinear_t,
            local_bilinear_i,
            local_bilinear_j,
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
        buffer_length,
        colons,
        horiz_method,
    )
end

# Constructor for the case with vertical spaces (horizontal_method accepted, ignored)
function _Remapper(
    space::Spaces.FiniteDifferenceSpace;
    target_zcoords::AbstractArray,
    target_hcoords::Nothing,
    buffer_length::Int = 1,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
)
    comms_ctx = ClimaComms.context(space)
    FT = Spaces.undertype(space)
    ArrayType = ClimaComms.array_type(space)

    vert_interpolation_weights =
        ArrayType(vertical_interpolation_weights(space, target_zcoords))
    vert_bounding_indices =
        ArrayType(vertical_bounding_indices(space, target_zcoords))

    local_interpolated_values =
        ArrayType(zeros(FT, (length(target_zcoords), buffer_length)))
    interpolated_values =
        ArrayType(zeros(FT, (length(target_zcoords), buffer_length)))
    colons = (:,)

    return Remapper(
        comms_ctx,
        space,
        nothing, # local_target_hcoords,
        target_zcoords,
        nothing, # local_target_hcoords_bitmask,
        nothing, # local_horiz_interpolation_weights,
        nothing, # local_horiz_indices,
        vert_interpolation_weights,
        vert_bounding_indices,
        local_interpolated_values,
        nothing, # field_values,
        interpolated_values,
        buffer_length,
        colons,
        SpectralElementRemapping(), # no horizontal interpolation
    )
end

"""
    _set_interpolated_values!(remapper, field)

Change the local state of `remapper` by performing interpolation of `fields` on the vertical
and horizontal points.
"""
_set_interpolated_values!(remapper::Remapper, fields) =
    _set_interpolated_values!(remapper.horiz_method, remapper, fields)

function _set_interpolated_values!(
    horiz_method::BilinearRemapping,
    remapper::Remapper,
    fields,
)
    _set_interpolated_values_bilinear!(
        remapper._local_interpolated_values,
        fields,
        remapper._field_values,
        remapper.local_horiz_indices,
        remapper.vert_interpolation_weights,
        remapper.vert_bounding_indices,
        horiz_method.local_bilinear_s,
        horiz_method.local_bilinear_t,
        horiz_method.local_bilinear_i,
        horiz_method.local_bilinear_j,
    )
end

function _set_interpolated_values!(::SpectralElementRemapping, remapper::Remapper, fields)
    _set_interpolated_values!(
        remapper._local_interpolated_values,
        fields,
        remapper._field_values,
        remapper.local_horiz_indices,
        remapper.local_horiz_interpolation_weights,
        remapper.vert_interpolation_weights,
        remapper.vert_bounding_indices,
    )
end

# 1D linear (2D extruded): horizontal linear at v_lo and v_hi, then vertical blend.
# (bilinear and linear are defined in remapping_utils.jl)
function _set_interpolated_values_bilinear!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_corners,
    local_horiz_indices,
    vert_interpolation_weights::AbstractArray,
    vert_bounding_indices::AbstractArray,
    local_bilinear_s,
    ::Nothing,
    local_bilinear_i,
    ::Nothing,
)
    CI = CartesianIndex
    for (field_index, field) in enumerate(fields)
        fv = Fields.field_values(field)
        # out_index = horizontal target point
        # vindex = vertical target level 
        # h = element index
        # (i, s) = 1D linear stencil.
        @inbounds for (vindex, (A, B)) in enumerate(vert_interpolation_weights)
            (v_lo, v_hi) = vert_bounding_indices[vindex]
            for (out_index, h) in enumerate(local_horiz_indices)
                i, s = local_bilinear_i[out_index], local_bilinear_s[out_index]
                out[out_index, vindex, field_index] =
                    A * linear(fv[CI(i, 1, 1, v_lo, h)], fv[CI(i + 1, 1, 1, v_lo, h)], s) +
                    B * linear(fv[CI(i, 1, 1, v_hi, h)], fv[CI(i + 1, 1, 1, v_hi, h)], s)
            end
        end
    end
end

# Bilinear path (3D): horizontal bilinear level-by-level (at v_lo and v_hi), then vertical blend.
# Same structure as spectral: horizontal interpolation at each level, then linear vertical.
function _set_interpolated_values_bilinear!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_corners,
    local_horiz_indices,
    vert_interpolation_weights::AbstractArray,
    vert_bounding_indices::AbstractArray,
    local_bilinear_s,
    local_bilinear_t,
    local_bilinear_i,
    local_bilinear_j,
)
    CI = CartesianIndex
    for (field_index, field) in enumerate(fields)
        field_values = Fields.field_values(field)
        @inbounds for (vindex, (A, B)) in enumerate(vert_interpolation_weights)
            (v_lo, v_hi) = vert_bounding_indices[vindex]
            for (out_index, h) in enumerate(local_horiz_indices)
                i, j = local_bilinear_i[out_index], local_bilinear_j[out_index]
                s, t = local_bilinear_s[out_index], local_bilinear_t[out_index]
                # Horizontal bilinear at v_lo (level by level, no vertical yet)
                scratch_corners[1, 1] = field_values[CI(i, j, 1, v_lo, h)]
                scratch_corners[2, 1] = field_values[CI(i + 1, j, 1, v_lo, h)]
                scratch_corners[2, 2] = field_values[CI(i + 1, j + 1, 1, v_lo, h)]
                scratch_corners[1, 2] = field_values[CI(i, j + 1, 1, v_lo, h)]
                f_lo = bilinear(
                    scratch_corners[1, 1],
                    scratch_corners[2, 1],
                    scratch_corners[2, 2],
                    scratch_corners[1, 2],
                    s,
                    t,
                )
                # Horizontal bilinear at v_hi
                scratch_corners[1, 1] = field_values[CI(i, j, 1, v_hi, h)]
                scratch_corners[2, 1] = field_values[CI(i + 1, j, 1, v_hi, h)]
                scratch_corners[2, 2] = field_values[CI(i + 1, j + 1, 1, v_hi, h)]
                scratch_corners[1, 2] = field_values[CI(i, j + 1, 1, v_hi, h)]
                f_hi = bilinear(
                    scratch_corners[1, 1],
                    scratch_corners[2, 1],
                    scratch_corners[2, 2],
                    scratch_corners[1, 2],
                    s,
                    t,
                )
                # Vertical linear blend (same as spectral)
                out[out_index, vindex, field_index] = A * f_lo + B * f_hi
            end
        end
    end
end

# 1D linear: horizontal-only.
function _set_interpolated_values_bilinear!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_corners,
    local_horiz_indices,
    ::Nothing,
    ::Nothing,
    local_bilinear_s,
    ::Nothing,
    local_bilinear_i,
    ::Nothing,
)
    CI = CartesianIndex
    for (field_index, field) in enumerate(fields)
        fv = Fields.field_values(field)
        @inbounds for (out_index, h) in enumerate(local_horiz_indices)
            i, s = local_bilinear_i[out_index], local_bilinear_s[out_index]
            out[out_index, field_index] =
                linear(fv[CI(i, 1, 1, 1, h)], fv[CI(i + 1, 1, 1, 1, h)], s)
        end
    end
end

# Bilinear path (2D horizontal-only): horizontal-only (no vertical).
function _set_interpolated_values_bilinear!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_corners,
    local_horiz_indices,
    ::Nothing,
    ::Nothing,
    local_bilinear_s,
    local_bilinear_t,
    local_bilinear_i,
    local_bilinear_j,
)
    CI = CartesianIndex
    for (field_index, field) in enumerate(fields)
        field_values = Fields.field_values(field)
        @inbounds for (out_index, h) in enumerate(local_horiz_indices)
            i, j = local_bilinear_i[out_index], local_bilinear_j[out_index]
            c11 = field_values[CI(i, j, 1, 1, h)]
            c21 = field_values[CI(i + 1, j, 1, 1, h)]
            c22 = field_values[CI(i + 1, j + 1, 1, 1, h)]
            c12 = field_values[CI(i, j + 1, 1, 1, h)]
            s, t = local_bilinear_s[out_index], local_bilinear_t[out_index]
            out[out_index, field_index] = bilinear(c11, c21, c22, c12, s, t)
        end
    end
end

# CPU, 3D case
function set_interpolated_values_cpu_kernel!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    (I1, I2)::NTuple{2},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    scratch_field_values,
)
    space = axes(first(fields))
    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quad)
    for (field_index, field) in enumerate(fields)
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
                            A * field_values[CartesianIndex(i, j, 1, v_lo, h)] +
                            B * field_values[CartesianIndex(i, j, 1, v_hi, h)]
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
                out[out_index, vindex, field_index] = tmp
            end
        end
    end
end

# CPU, vertical case
function set_interpolated_values_cpu_kernel!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    ::Nothing,
    ::Nothing,
    vert_interpolation_weights,
    vert_bounding_indices,
    ::Nothing,
)
    space = axes(first(fields))
    FT = Spaces.undertype(space)
    for (field_index, field) in enumerate(fields)
        field_values = Fields.field_values(field)

        # Reading values from field_values is expensive, so we try to limit the number of reads. We can do
        # this because multiple target points might be all contained in the same element.
        prev_vindex = -1
        @inbounds for (vindex, (A, B)) in enumerate(vert_interpolation_weights)
            (v_lo, v_hi) = vert_bounding_indices[vindex]
            # If we are no longer in the same element, read the field values again
            if prev_vindex != vindex
                out[vindex, field_index] = (
                    A * field_values[CartesianIndex(1, 1, 1, v_lo, 1)] +
                    B * field_values[CartesianIndex(1, 1, 1, v_hi, 1)]
                )
                prev_vindex = vindex
            end
        end
    end
end

# CPU, 2D case
function set_interpolated_values_cpu_kernel!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    (I,)::NTuple{1},
    local_horiz_indices,
    vert_interpolation_weights,
    vert_bounding_indices,
    scratch_field_values,
)
    space = axes(first(fields))
    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quad)
    for (field_index, field) in enumerate(fields)
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
                            A * field_values[CartesianIndex(i, 1, 1, v_lo, h)] +
                            B * field_values[CartesianIndex(i, 1, 1, v_hi, h)]
                        )
                    end
                    prev_vindex, prev_lidx = vindex, h
                end

                tmp = zero(FT)

                for i in 1:Nq
                    tmp += I[out_index, i] * scratch_field_values[i]
                end
                out[out_index, vindex, field_index] = tmp
            end
        end
    end
end

function _set_interpolated_values!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_field_values,
    local_horiz_indices,
    interpolation_matrix,
    vert_interpolation_weights::AbstractArray,
    vert_bounding_indices::AbstractArray,
)
    _set_interpolated_values_device!(
        out,
        fields,
        scratch_field_values,
        local_horiz_indices,
        interpolation_matrix,
        vert_interpolation_weights,
        vert_bounding_indices,
        ClimaComms.device(first(fields)),
    )
end

function _set_interpolated_values_device!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    scratch_field_values,
    local_horiz_indices,
    interpolation_matrix,
    vert_interpolation_weights::AbstractArray,
    vert_bounding_indices::AbstractArray,
    ::ClimaComms.AbstractDevice,
)
    set_interpolated_values_cpu_kernel!(
        out,
        fields,
        interpolation_matrix,
        local_horiz_indices,
        vert_interpolation_weights,
        vert_bounding_indices,
        scratch_field_values,
    )
end

# Horizontal
function _set_interpolated_values!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    _scratch_field_values,
    local_horiz_indices,
    local_horiz_interpolation_weights,
    ::Nothing,
    ::Nothing,
)
    _set_interpolated_values_device!(
        out,
        fields,
        _scratch_field_values,
        local_horiz_indices,
        local_horiz_interpolation_weights,
        nothing,
        nothing,
        ClimaComms.device(axes(first(fields))),
    )
end

function _set_interpolated_values_device!(
    out::AbstractArray,
    fields::AbstractArray{<:Fields.Field},
    _scratch_field_values,
    local_horiz_indices,
    local_horiz_interpolation_weights,
    ::Nothing,
    ::Nothing,
    ::ClimaComms.AbstractDevice,
)
    space = axes(first(fields))
    FT = Spaces.undertype(space)
    quad = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(quad)

    hdims = length(local_horiz_interpolation_weights)
    hdims in (1, 2) || error("Cannot handle $hdims horizontal dimensions")

    for (field_index, field) in enumerate(fields)
        field_values = Fields.field_values(field)
        @inbounds for (out_index, h) in enumerate(local_horiz_indices)
            out[out_index, field_index] = zero(FT)
            if hdims == 2
                for j in 1:Nq, i in 1:Nq
                    out[out_index, field_index] +=
                        local_horiz_interpolation_weights[1][out_index, i] *
                        local_horiz_interpolation_weights[2][out_index, j] *
                        field_values[CartesianIndex(i, j, 1, 1, h)]
                end
            elseif hdims == 1
                for i in 1:Nq
                    out[out_index, field_index] +=
                        local_horiz_interpolation_weights[1][out_index, i] *
                        field_values[CartesianIndex(i, 1, 1, 1, h)]
                end
            end
        end
    end
end

"""
    _apply_mpi_bitmask!(remapper::Remapper, num_fields::Int)

Change to local (private) state of the `remapper` by applying the MPI bitmask and reconstructing
the correct shape for the interpolated values.

Internally, `remapper` performs interpolation on a flat list of points, this function moves points
around according to MPI-ownership and the expected output shape.

`num_fields` is the number of fields that have been processed and have to be moved in the
`interpolated_values`. We assume that it is always the first `num_fields` that have to be moved.
"""
function _apply_mpi_bitmask!(remapper::Remapper, num_fields::Int)
    if isnothing(remapper.target_zcoords)
        view(
            remapper._interpolated_values,
            remapper.local_target_hcoords_bitmask,
            1:num_fields,
        ) .= view(remapper._local_interpolated_values, :, 1:num_fields)
    else
        view(
            remapper._interpolated_values,
            remapper.local_target_hcoords_bitmask,
            :,
            1:num_fields,
        ) .= view(remapper._local_interpolated_values, :, :, 1:num_fields)
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

function _collect_interpolated_values!(
    dest,
    remapper::Remapper,
    index_field_begin::Int,
    index_field_end::Int;
    only_one_field,
)
    cuda_synchronize(ClimaComms.device(remapper.comms_ctx))   # Sync streams before MPI calls
    if only_one_field
        ClimaComms.reduce!(
            remapper.comms_ctx,
            view(remapper._interpolated_values, remapper.colons..., 1),
            dest,
            +,
        )
    else
        num_fields = 1 + index_field_end - index_field_begin
        ClimaComms.reduce!(
            remapper.comms_ctx,
            view(
                remapper._interpolated_values,
                remapper.colons...,
                1:num_fields,
            ),
            view(dest, remapper.colons..., index_field_begin:index_field_end),
            +,
        )
    end
    return nothing
end

"""
   interpolate(remapper::Remapper, fields)
   interpolate!(dest, remapper::Remapper, fields)

Interpolate the given `field`(s) as prescribed by `remapper`.

The optimal number of fields passed is the `buffer_length` of the `remapper`. If
more fields are passed, the `remapper` will batch work with size up to its
`buffer_length`.

This call mutates the internal (private) state of the `remapper`.

Horizontally, interpolation is performed with the barycentric formula in
[Berrut2004](@cite), equation (3.2). Vertical interpolation is linear except
in the boundary elements where it is 0th order.

`interpolate!` writes the output to the given `dest`iniation. `dest` is expected
to be defined on the root process and to be `nothing` for the other processes.

Note: `interpolate` allocates new arrays and has some internal type-instability,
`interpolate!` is non-allocating and type-stable.

When using `interpolate!`, the `dest`ination has to be the same array type as the
device in use (e.g., `CuArray` for CUDA runs).

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

# Or
int12 = interpolate(remapper, [field1, field2])
# With int1 = int12[1, :, :, :]
```
"""
function interpolate(remapper::Remapper, fields)
    ArrayType = ClimaComms.array_type(remapper.space)
    FT = Spaces.undertype(remapper.space)
    only_one_field = fields isa Fields.Field

    interpolated_values_dim..., _buffer_length =
        size(remapper._interpolated_values)

    allocate_extra = only_one_field ? () : (length(fields),)
    dest = ArrayType(zeros(FT, interpolated_values_dim..., allocate_extra...))

    # interpolate! has an MPI call, so it is important to return after it is
    # called, not before!
    interpolate!(dest, remapper, fields)
    ClimaComms.iamroot(remapper.comms_ctx) || return nothing
    return dest
end

# dest has to be allowed to be nothing because interpolation happens only on the root
# process
function interpolate!(
    dest::Union{Nothing, <:AbstractArray},
    remapper::Remapper,
    fields,
)
    only_one_field = fields isa Fields.Field
    if only_one_field
        fields = [fields]
    end
    isa_vertical_space = remapper.space isa Spaces.FiniteDifferenceSpace

    for field in fields
        axes(field) == remapper.space ||
            error("Field is defined on a different space than remapper")
    end

    if !isnothing(dest)
        # !isnothing(dest) means that this is the root process, in this case, the size have
        # to match (ignoring the buffer_length)
        dest_size = only_one_field ? size(dest) : size(dest)[1:(end - 1)]

        dest_size == size(remapper._interpolated_values)[1:(end - 1)] || error(
            "Destination array is not compatible with remapper (size mismatch)",
        )

        expected_array_type =
            ClimaComms.array_type(ClimaComms.device(remapper.comms_ctx))

        found_type = nameof(typeof(dest))

        parent(dest) isa expected_array_type ||
            error("dest is a $found_type, expected $expected_array_type")
    end
    index_field_begin, index_field_end =
        1, min(length(fields), remapper.buffer_length)

    while true
        num_fields = 1 + index_field_end - index_field_begin

        # Reset interpolated_values. This is needed because we collect distributed results
        # with a + reduction.
        _reset_interpolated_values!(remapper)
        # Perform the interpolations (horizontal and vertical)
        _set_interpolated_values!(
            remapper,
            view(fields, index_field_begin:index_field_end),
        )

        if !isa_vertical_space
            # For spaces with an horizontal component, reshape the output so that it is a nice grid.
            _apply_mpi_bitmask!(remapper, num_fields)
        else
            # For purely vertical spaces, just move to _interpolated_values
            remapper._interpolated_values .= remapper._local_interpolated_values
        end

        # Finally, we have to send all the _interpolated_values to root and sum them up to
        # obtain the final answer.
        _collect_interpolated_values!(
            dest,
            remapper,
            index_field_begin,
            index_field_end;
            only_one_field,
        )

        index_field_end != length(fields) || break
        index_field_begin = index_field_begin + remapper.buffer_length
        index_field_end =
            min(length(fields), index_field_end + remapper.buffer_length)
    end
    return nothing
end

"""
    interpolate(field; hresolution=180, zresolution=nothing, target_hcoords=..., target_zcoords=..., horizontal_method=SpectralElementRemapping())

Interpolate `field` onto the Cartesian product of `target_hcoords` and `target_zcoords`.
`zresolution = nothing` disables vertical interpolation. `horizontal_method`: `SpectralElementRemapping()` or `BilinearRemapping()`.
For performance, use a `Remapper` and `interpolate(remapper, fields)` instead.

Example
========

Given `field`, a `Field` defined on a cubed sphere.

By default, a target uniform grid is chosen (with resolution `hresolution` and
`zresolution`), so remapping is simply
```julia
julia> interpolate(field)
```
This will return an array of interpolated values.

Resolution can be specified
```julia
julia> interpolate(field; hresolution = 100, zresolution = 50)
```
Coordinates can be also specified directly:
```julia
julia> longpts = range(-180.0, 180.0, 21)
julia> latpts = range(-80.0, 80.0, 21)
julia> zpts = range(0.0, 1000.0, 21)

julia> hcoords = [Geometry.LatLongPoint(lat, long) for long in longpts, lat in latpts]
julia> zcoords = [Geometry.ZPoint(z) for z in zpts]

julia> interpolate(field, target_hcoords, target_zcoords)
```

If you need the array of coordinates, you can call `default_target_hcoords` (or
`default_target_zcoords`) passing `axes(field)`. This will return an array of
`Geometry.Point`s. The functions `Geometry.Components` and `Geometry.Component`
can be used to extract the components as numeric values. For example,
```julia
julia> Geometry.components.(Geometry.components.([
           Geometry.LatLongPoint(x, y) for x in range(-180.0, 180.0, length = 180),
           y in range(-90.0, 90.0, length = 180)
       ]))
180×180 Matrix{StaticArraysCore.SVector{2, Float64}}:
 [-180.0, -90.0]    [-180.0, -88.9944]    …  [-180.0, 88.9944]    [-180.0, 90.0]
  ⋮                                        ⋱
 [180.0, -90.0]     [180.0, -88.9944]        [180.0, 88.9944]     [180.0, 90.0]
```
To extract only long or lat, one can broadcast `getindex`
```julia
julia> lats = getindex.(Geometry.components.([Geometry.LatLongPoint(x, y)
                                              for x in range(-180.0, 180.0, length = 180),
                                                  y in range(-90.0, 90.0, length = 180)
                                             ]),
                        1)
```
This can be used directly for plotting.
"""
function interpolate(
    field::Fields.Field;
    zresolution = nothing,
    hresolution = 180,
    target_hcoords = default_target_hcoords(axes(field); hresolution),
    target_zcoords = default_target_zcoords(axes(field); zresolution),
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
)
    return interpolate(
        field,
        axes(field);
        target_hcoords,
        target_zcoords,
        horizontal_method,
    )
end

# interpolate, positional
function interpolate(
    field::Fields.Field,
    target_hcoords,
    target_zcoords;
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
)
    return interpolate(
        field,
        axes(field);
        target_hcoords,
        target_zcoords,
        horizontal_method,
    )
end

function interpolate(
    field::Fields.Field,
    space::Spaces.AbstractSpace;
    target_hcoords,
    target_zcoords,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
)
    remapper = Remapper(space; target_hcoords, target_zcoords, horizontal_method)
    return interpolate(remapper, field)
end

function interpolate(
    field::Fields.Field,
    space::Spaces.FiniteDifferenceSpace;
    target_zcoords,
    target_hcoords,
    kwargs...,  # e.g. horizontal_method; accepted for uniform API, ignored for vertical-only
)
    remapper = Remapper(space; target_zcoords)
    return interpolate(remapper, field)
end

function interpolate(
    field::Fields.Field,
    space::Spaces.AbstractSpectralElementSpace;
    target_hcoords,
    target_zcoords,
    horizontal_method::AbstractRemappingMethod = SpectralElementRemapping(),
)
    remapper = Remapper(space; target_hcoords, horizontal_method)
    return interpolate(remapper, field)
end
