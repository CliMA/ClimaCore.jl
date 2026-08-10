import ClimaCore: Spaces, Quadratures, Topologies, Operators
import Base.Broadcast: Broadcasted
import ClimaCore.Fields: Field, field_values, AbstractFieldStyle
import ClimaComms
import ClimaCore.Utilities: half, new, unsafe_eltype
import ClimaCore.Operators
import ClimaCore.Geometry: project
import ClimaCore.Operators:
    StencilBroadcasted, setidx!, getidx, reconstruct_placeholder_space
import ClimaCore.MatrixFields: FaceToCenter, CenterToFace, CenterToCenter,
    FaceToFace, FDOperatorMatrix, MultiplyColumnwiseBandMatrixField,
    op_matrix_row_type, BandMatrixRow, band_matrix_d
import ClimaCore.Utilities
import ClimaCore
using ClimaCore.MatrixFields
using ClimaCore.Geometry
import UnrolledUtilities


include("column_matrix_helpers.jl")

"""
    max_eager_shmem_per_thread(bc)

Return the maximum number of bytes that any single sub-expression of `bc` needs to
cache its result in shared memory, per thread.

Only `MultiplyColumnwiseBandMatrixField` stencil operations cache a result (the
projected row of their second argument) in shared memory; every other node needs no
shared memory of its own but is still traversed for nested multiplications. The launch
configuration multiplies this value by the number of threads per block to size the
dynamic shared memory, so that the result of any single multiplication is guaranteed to
fit and can always be cached.
"""
max_eager_shmem_per_thread(x) = 0
max_eager_shmem_per_thread(bc::Union{Broadcasted, StencilBroadcasted}) =
    _max_eager_shmem_over_args(bc.args)
max_eager_shmem_per_thread(
    bc::StencilBroadcasted{S, <:MultiplyColumnwiseBandMatrixField},
) where {S} =
    max(sizeof(cached_operand_type(bc)), _max_eager_shmem_over_args(bc.args))

_max_eager_shmem_over_args(::Tuple{}) = 0
_max_eager_shmem_over_args(args::Tuple) = max(
    max_eager_shmem_per_thread(first(args)),
    _max_eager_shmem_over_args(Base.tail(args)),
)

"""
    cached_operand_type(bc)

The type that `calc_level_val` writes into shared memory for the multiplication `bc`,
i.e. `typeof(project_row2_for_mul(mat1_row, mat2_row, hidx, mat2_space))`.

The size cannot be read off the second operand directly, because
`project_row2_for_mul` projects every tensor leaf of that operand onto the axis dual
to the first operand's entries, which changes its size in either direction: a
`Covariant1Vector` widens to a `Contravariant123Vector`, while a `Covariant12Vector`
narrows to a `Contravariant1Vector`. For a matrix-matrix product those leaves are also
nested inside a `BandMatrixRow`, so no property of the operand's outermost type
bounds the projected size. Mirror `project_row2_for_mul`'s type-level logic instead
and infer the projected type, so the buffer is always big enough for what the kernel
writes into it.
"""
function cached_operand_type(bc)
    mat1_type = unsafe_eltype(bc.args[1i32])
    mat2_type = unsafe_eltype(bc.args[2i32])
    mat1_et = mat1_type <: BandMatrixRow ? eltype(mat1_type) : mat1_type
    project_onto =
        ClimaCore.Geometry.recursively_find_dual_axes_for_projection(mat1_et)
    isnothing(project_onto) && return mat2_type
    lg_type = Spaces.local_geometry_type(typeof(axes(bc.args[2i32])))
    projected_type = ClimaCore.Utilities.return_type(
        recursively_project,
        Tuple{Tuple{typeof(project_onto), lg_type}, mat2_type},
    )
    isconcretetype(projected_type) || error(
        "Unable to size the eager finite difference kernel's shared memory: \
         inference gave the non-concrete type $projected_type for the \
         projection of a $mat2_type operand onto $project_onto",
    )
    return projected_type
end


ClimaCore.Utilities.unsafe_eltype(::CUDA.CuRefType{T}) where {T} = T

"""
    has_padding_thread(space)

The eager kernel launches one thread per face level. For a non-periodic center-output
space there is one fewer center level than face level, so the last x-thread
(`v == blockDim().x`) does not map to a valid output level and must be skipped. For face
output, or for periodic spaces (where the center and face level counts are equal), every
thread maps to a valid level and none must be skipped.

Both the staggering and `isperiodic` are encoded in the space type, so this is a
compile-time constant.
"""
@inline has_padding_thread(space) =
    space.staggering isa Spaces.CellCenter && !Topologies.isperiodic(space)

"""
    eager_copyto_stencil_kernel!(out, bc::BC, mask, space)

CUDA kernel to compute the value of a `Broadcasted` or `StencilBroadcasted` at a single index.
This calls `calc_level_val(bc, hidx, space)`, which computes the value of the broadcasted
expression at the given index, and then copies the result into `out`.
"""
Base.@propagate_inbounds function eager_copyto_stencil_kernel!(
    out,
    bc::BC,
    mask,
    space,
) where {BC}
    v = threadIdx().x
    col_idx = threadIdx().y + (blockIdx().x - 1) * blockDim().y
    (i, j, h) = if mask isa NoMask
        # `Ni` and `Nj` are read off the output layout's type parameters (see
        # `vijh_params`), so they are compile-time constants and the `CartesianIndices`
        # decomposition below is a fixed-divisor `divrem`. Only `Nh` is a runtime value,
        # and being the last extent it is never divided by.
        size_params = ClimaCore.DataLayouts.vijh_params(ClimaCore.Fields.field_values(out))
        Nj = size_params.Nj
        Ni = size_params.Ni
        Nh = ClimaCore.DataLayouts.nelems(ClimaCore.Fields.field_values(out))
        cart_inds = CartesianIndices((Ni, Nj, Nh))
        col_idx > length(cart_inds) && return nothing
        @inbounds cart_inds[col_idx].I
    else
        (; N, i_map, j_map, h_map) = mask
        # Bound by the active-column count `N`, not `length(i_map)`: the maps
        # are allocated with one entry per column of the layout, but
        # `set_mask_maps!` only writes the first `N` entries, and the launch
        # rounds the grid up to a multiple of `blockDim().y` columns.
        @inbounds col_idx > N[1] && return nothing
        @inbounds i = i_map[col_idx]
        @inbounds j = j_map[col_idx]
        @inbounds h = h_map[col_idx]
        (i, j, h)
    end
    hidx = (i, j, h)
    val = @inbounds @inline calc_level_val(bc, hidx, space)
    if space.staggering isa ClimaCore.Grids.CellFace
        @inbounds @inline setidx!(space, out, v - half, hidx, val)
    elseif !(has_padding_thread(space) && v == CUDA.blockDim().x)
        @inbounds @inline setidx!(space, out, v, hidx, val)
    end
    return nothing
end

# All the functions below this line should not be used outside of this file

"""
    calc_level_val(bc, hidx, space)

Call `calc_level_val` on all the arguments of `bc`, and then apply the function `bc.f` to the results.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    hidx,
    space,
) where {BC <: Base.Broadcast.Broadcasted}
    resolved_args = @inbounds @inline UnrolledUtilities.unrolled_map(
        Base.Fix2(reconstruct_space_and_call_calc_level_val, (hidx, space)),
        bc.args,
    )
    return @inline @inbounds bc.f(resolved_args...)
end

"""
    reconstruct_space_and_call_calc_level_val(arg, (hidx, space))

If `arg` is a `Broadcasted`, `StencilBroadcasted`, or `Field`,
reconstruct the space for the argument and call `calc_level_val` on it. This allows
us to use Base.Fix2.
"""
Base.@propagate_inbounds reconstruct_space_and_call_calc_level_val(
    arg::A,
    space_idx_tpl::S,
) where {
    A <: Union{Base.Broadcast.Broadcasted{<:AbstractFieldStyle}, StencilBroadcasted, Field},
    S,
} = @inbounds @inline calc_level_val(
    arg,
    space_idx_tpl[1],
    reconstruct_placeholder_space(axes(arg), space_idx_tpl[2]),
)
Base.@propagate_inbounds reconstruct_space_and_call_calc_level_val(
    arg::A,
    space_idx_tpl::S,
) where {A, S} = @inbounds @inline calc_level_val(arg, space_idx_tpl[1], space_idx_tpl[2])

"""
    calc_level_val(val::T, hidx, space)

If `val` is not a `Broadcasted`, `StencilBroadcasted`, or `Field`, just return `val`.
If it is a `Ref`, return `val[]`. If it is a one element tuple, return the element.
"""
Base.@propagate_inbounds calc_level_val(val::T, hidx, space) where {T <: Ref} = val[]
Base.@propagate_inbounds calc_level_val(val::T, hidx, space) where {V, T <: Tuple{V}} =
    first(val)
Base.@propagate_inbounds calc_level_val(arg::S, hidx, space) where {S} = arg

"""
    calc_level_val(bc::StencilBroadcasted{<:Any, <: MultiplyColumnwiseBandMatrixField}, hidx, space)

Call `calc_level_val` on both args of `bc`, place the result of the second arg into shared memory,
and then perform the multiplication.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    hidx,
    space,
) where {
    S,
    Op <: MultiplyColumnwiseBandMatrixField,
    BC <: StencilBroadcasted{S, Op},
}
    # The launch configuration sizes the dynamic shared memory to fit the largest single
    # expression result in the broadcasted tree (see `max_eager_shmem_per_thread`), so the
    # result of every multiplication is guaranteed to fit and can always be cached.
    v = threadIdx().x
    block_col_idx = threadIdx().y
    # Whether the vertical topology is periodic. `row_mul_*!` uses this (a compile-time
    # constant) to wrap operand reads at the column ends instead of zero-padding them.
    periodic = Topologies.isperiodic(space)
    mat1_space = reconstruct_placeholder_space(axes(bc.args[1i32]), space)
    mat2_space = reconstruct_placeholder_space(axes(bc.args[2i32]), space)

    mat2_row = calc_level_val(bc.args[2i32], hidx, mat2_space)
    mat1_row = calc_level_val(bc.args[1i32], hidx, mat1_space)
    # project before placing in shared memory to avoid projecting multiple times
    mat2_row_converted =
        @inbounds @inline project_row2_for_mul(mat1_row, mat2_row, hidx, mat2_space)
    # It should be possible to use static shared memory here, but it allocates new shared memory
    # for each layer of recursion
    CUDA.sync_threads()
    # it should be possible to use a multi dim shared array here as well, but it seems to
    # cause some weird issues with the indexing, so I'm just using a 1D array and indexing manually
    mat2 = CUDA.CuDynamicSharedArray(
        typeof(mat2_row_converted),
        CUDA.blockDim().x * CUDA.blockDim().y,
    )
    @inbounds mat2[v + (block_col_idx - 1) * CUDA.blockDim().x] = mat2_row_converted
    CUDA.sync_threads()
    # if the output is on centers, the padding CUDA.blockDim().xth thread can just return 0
    has_padding_thread(mat1_space) && v == CUDA.blockDim().x &&
        return new(eltype(bc))
    if mat1_space.staggering isa Spaces.CellCenter
        mat1_shape =
            eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat1_row))) <:
            ClimaCore.Utilities.PlusHalf ? FaceToCenter() : CenterToCenter()
    else
        mat1_shape =
            eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat1_row))) <:
            ClimaCore.Utilities.PlusHalf ? CenterToFace() : FaceToFace()
    end

    if mat2_row_converted isa ClimaCore.MatrixFields.BandMatrixRow
        # mat * mat case
        if mat2_space.staggering isa Spaces.CellCenter
            mat2_shape =
                eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat2_row))) <:
                ClimaCore.Utilities.PlusHalf ? FaceToCenter() : CenterToCenter()
        else
            mat2_shape =
                eltype(ClimaCore.MatrixFields.outer_diagonals(typeof(mat2_row))) <:
                ClimaCore.Utilities.PlusHalf ? CenterToFace() : FaceToFace()
        end
        out = @inbounds @inline row_mul_mat!(
            eltype(bc),
            mat1_row,
            mat2,
            mat1_shape,
            mat2_shape,
            periodic,
        )
        out isa eltype(bc) || return convert(eltype(bc), out)
        return out
    else
        # mat * vec case
        out =
            @inbounds @inline row_mul_vec!(eltype(bc), mat1_row, mat2, mat1_shape, periodic)
        out isa eltype(bc) || return convert(eltype(bc), out)
        return out
    end
end

"""
    calc_level_val(bc::StencilBroadcasted{<:Any, <: SetBoundaryOperator}, hidx, space)

A `SetBoundaryOperator` only modifies the two boundary levels of the space it is applied
to, and is the identity in the interior. At the boundaries we dispatch to
`stencil_left_boundary` / `stencil_right_boundary` (which extract and project the
boundary value), and in the interior we reuse the eagerly-computed value of the argument.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    hidx,
    space,
) where {
    S,
    Op <: Operators.SetBoundaryOperator,
    BC <: StencilBroadcasted{S, Op},
}
    op = bc.op
    v = threadIdx().x
    val_no_bcs = @inline @inbounds calc_level_val(bc.args[1i32], hidx, space)
    # A `SetBoundaryOperator` is space-preserving (`return_space(op, space) = space`), so
    # this method is compiled for both staggerings: the automatic conversion puts one on a
    # face output (InterpolateC2F + SetValue), on a center output (DivergenceF2C +
    # SetDivergence), and on a face input (GradientF2C + SetValue). Deriving `idx` from
    # the compile-time staggering type keeps the two apart, so the `PlusHalf` face index
    # only reaches `should_call_*_boundary` when compiling for a face space and its
    # `idx < left_interior_idx` comparison never mixes a `PlusHalf` with an integer center
    # index -- which would pull in non-GPU-compatible error-formatting code.
    idx = space.staggering isa Spaces.CellFace ? (v - half) : v
    if Operators.should_call_left_boundary(idx, space, op, bc.args...)
        lbw = Operators.left_boundary_window(space)
        return @inbounds @inline Operators.stencil_left_boundary(
            op,
            Operators.get_boundary(op, lbw),
            space,
            idx,
            hidx,
            bc.args...,
        )
    elseif !(has_padding_thread(space) && v == CUDA.blockDim().x) &&
           Operators.should_call_right_boundary(idx, space, op, bc.args...)
        rbw = Operators.right_boundary_window(space)
        return @inbounds @inline Operators.stencil_right_boundary(
            op,
            Operators.get_boundary(op, rbw),
            space,
            idx,
            hidx,
            bc.args...,
        )
    end

    return val_no_bcs
end

"""
    calc_level_val(bc::StencilBroadcasted, hidx, space)

Fallback case of `calc_level_val` that calls `Operators.getidx`. This is used for
affine BCs or values that won't fit in shmmem.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    hidx,
    space,
) where {BC <: StencilBroadcasted}
    v = threadIdx().x
    if has_padding_thread(space)
        v == CUDA.blockDim().x && return @inline @inbounds new(eltype(bc))
    end
    li = space.staggering isa Spaces.CellCenter ? 1i32 : half
    idx = v - 1i32 + li
    return @inbounds @inline getidx(space, bc, idx, hidx)
end

"""
    calc_level_val(arg::Field, hidx, space)

Returns the value of the field `f` at the thread's index.
When the staggering of `space` is `CellCenter`, the thread with `v == CUDA.blockDim().x` returns `new(eltype(f))`
"""
Base.@propagate_inbounds function calc_level_val(
    arg::F,
    hidx,
    space,
) where {F <: Field}
    data = field_values(arg)
    v = threadIdx().x
    (i, j, h) = hidx
    if space isa
       Union{Spaces.ExtrudedFiniteDifferenceSpace, Spaces.FiniteDifferenceSpace} &&
       has_padding_thread(space)
        v == CUDA.blockDim().x && return @inline @inbounds new(eltype(data))
    end
    return @inline @inbounds data[v, i, j, h]
end

"""
    calc_level_val(bc::StencilBroadcasted{<:Any, <: FDOperatorMatrix}, hidx, space)

Return the correct row of the operator matrix for the current thread
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    hidx,
    space,
) where {
    S,
    BC <:
    StencilBroadcasted{S, <:FDOperatorMatrix},
}
    op = bc.op.op
    args = bc.args
    val = @inbounds @inline get_op_row(op, args, hidx, space)
    return val
end

"""
    get_op_row(op, args, hidx, space)

Get the correct row of the operator matrix for the current thread, taking into account boundary conditions.
"""

Base.@propagate_inbounds function get_op_row(op, args, hidx, space)
    FT = Spaces.undertype(space)
    v = threadIdx().x

    outputs_to_face = space.staggering isa ClimaCore.Grids.CellFace
    row_type = @inbounds @inline op_matrix_row_type(op, FT, args[1:(end - 1)]...)
    if has_padding_thread(space) && v == CUDA.blockDim().x
        return new(row_type)
    end
    v_half = outputs_to_face ? v - half : v
    in_left_bnd = Operators.should_call_left_boundary(v_half, space, op, nothing)
    in_right_bnd =
        Operators.should_call_right_boundary(v_half, space, op, nothing)
    op_matrix = FDOperatorMatrix(op)
    if in_left_bnd
        lloc = Operators.left_boundary_window(space)
        left_bndry = Operators.get_boundary(op, lloc)
        val = @inbounds @inline Operators.stencil_left_boundary(
            op_matrix,
            left_bndry,
            space,
            v_half,
            hidx,
            args...,
        )
    elseif in_right_bnd
        rroc = Operators.right_boundary_window(space)
        right_bndry = Operators.get_boundary(op, rroc)
        val = @inbounds @inline Operators.stencil_right_boundary(
            op_matrix,
            right_bndry,
            space,
            v_half,
            hidx,
            args...,
        )
    else
        val =
            @inbounds @inline Operators.stencil_interior(
                op_matrix,
                space,
                v_half,
                hidx,
                args...,
            )
    end
    return val
end


"""
    project_row2_for_mul(mat1_row, mat2_row, hidx, space)

Projects `mat2_row` onto the correct axis for multiplication with `mat1_row` if necessary, and returns the projected row.
"""
Base.@propagate_inbounds function project_row2_for_mul(mat1_row, mat2_row, hidx, space)
    mat1_et = mat1_row isa BandMatrixRow ? eltype(mat1_row) : typeof(mat1_row)
    project_onto =
        ClimaCore.Geometry.recursively_find_dual_axes_for_projection(mat1_et)
    isnothing(project_onto) && return mat2_row
    v = threadIdx().x
    if has_padding_thread(space) && v == CUDA.blockDim().x
        lg = new(Spaces.local_geometry_type(typeof(space)))
    else
        v_maybe_half = space.staggering isa Spaces.CellFace ? v - half : v
        @inbounds lg = Geometry.LocalGeometry(space, v_maybe_half, hidx)
    end
    # put needed info into tuple so we can use Base.Fix2
    projection_tuple = (project_onto, lg)
    return @inbounds @inline recursively_project(
        projection_tuple,
        mat2_row,
    )
end

"""
    recursively_project(projection_tuple, y)

Recursively project `y` onto the axes in `projection_tuple[1]` using the local geometry
in `projection_tuple[2]`. The axes are either a single axis, which projects every tensor
leaf of `y`, or (for multi-component entries like Tuples and AutoBroadcasters) a Tuple
that pairs componentwise with `y`, with `nothing` marking components that need no
projection (see `Geometry.recursively_find_dual_axes_for_projection`).
"""
Base.@propagate_inbounds recursively_project(projection_tuple::T, y::Y) where {T, Y} =
    project_or_map(projection_tuple[1], projection_tuple[2], y)

# `nothing` marks a component that needs no projection (with disambiguating
# methods for the leaf types below).
@inline project_or_map(::Nothing, lg, y) = y
@inline project_or_map(::Nothing, lg, y::Number) = y
@inline project_or_map(::Nothing, lg, y::AbstractTensor) = y
# A Tuple of axes pairs componentwise with a multi-component entry.
Base.@propagate_inbounds project_or_map(axes_per_component::Tuple, lg, y::Tuple) =
    paired_projection(axes_per_component, y, lg)
Base.@propagate_inbounds project_or_map(
    axes_per_component::Tuple,
    lg,
    y::NamedTuple{names},
) where {names} =
    NamedTuple{names}(paired_projection(axes_per_component, values(y), lg))
Base.@propagate_inbounds project_or_map(
    axes_per_component::Tuple,
    lg,
    y::ClimaCore.Utilities.AutoBroadcaster,
) = ClimaCore.Utilities.AutoBroadcaster(
    project_or_map(axes_per_component, lg, ClimaCore.Utilities.unwrap(y)),
)
# A single axis projects every tensor leaf below it (a container, like a
# BandMatrixRow, maps the whole projection over its entries).
Base.@propagate_inbounds project_or_map(axis, lg, y) =
    map(Base.Fix1(recursively_project, (axis, lg)), y)
@inline project_or_map(axis, lg, y::Number) = y
Base.@propagate_inbounds project_or_map(axis, lg, y::AbstractTensor) =
    @inbounds @inline project(axis, y, lg)

Base.@propagate_inbounds paired_projection(::Tuple{}, ::Tuple{}, lg) = ()
Base.@propagate_inbounds paired_projection(axes_per_component, ys, lg) = (
    project_or_map(first(axes_per_component), lg, first(ys)),
    paired_projection(Base.tail(axes_per_component), Base.tail(ys), lg)...,
)

if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(recursively_project)
        m.recursion_relation = dont_limit
    end
    for m in methods(project_or_map)
        m.recursion_relation = dont_limit
    end
    for m in methods(paired_projection)
        m.recursion_relation = dont_limit
    end
    for m in methods(calc_level_val)
        m.recursion_relation = dont_limit
    end
end
