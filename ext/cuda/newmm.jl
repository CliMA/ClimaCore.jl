import ClimaCore: Spaces, Quadratures, Topologies, Operators
import Base.Broadcast: Broadcasted
import ClimaCore.Fields: Field, field_values
import ClimaComms
import ClimaCore.Utilities: half
import ClimaCore.Operators
import ClimaCore.Geometry: ⊗
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠, rmuladd, rmap
import ClimaCore.Operators: StencilBroadcasted, setidx!, getidx, reconstruct_placeholder_space
import ClimaCore.MatrixFields: FaceToCenter, CenterToFace, Square, CenterToCenter,
    FaceToFace, TwoArgFDOperator, OneArgFDOperator, has_affine_bc, FDOperatorMatrix,
    MultiplyColumnwiseBandMatrixField, operator_input_space, op_matrix_row_type, BandMatrixRow
using ClimaCore.MatrixFields
import ClimaCore.Utilities
import ClimaCore
using ClimaCore.MatrixFields
using ClimaCore.Geometry
using LinearAlgebra
import UnrolledUtilities


include("matmul.jl")

"""
    check_if_fits_in_shmem(x)

Check if `x`, or the `eltype(x)` can fit in shared memory with the current config.
"""
check_if_fits_in_shmem(bc::Union{StencilBroadcasted, Broadcasted, Field}) =
    sizeof(eltype(bc)) <= 36
check_if_fits_in_shmem(val) = sizeof(typeof(val)) <= 36


"""
    recursively_replace_fd_ops(val)

Recursively replace any `OneArgFDOperator` or `TwoArgFDOperator` in `val` with a
`MultiplyColumnwiseBandMatrixField` with the corresponding `FDOperatorMatrix`, if the operator
does not have affine BCs and the operator matrix fits in shared memory.

`OneArgFDOperator`s with affine BCs are also replaced with `MultiplyColumnwiseBandMatrixField`s
if all the BCs are `SetValue`s

"""
recursively_replace_fd_ops(val) = val

function recursively_replace_fd_ops(
    bc::Base.Broadcast.Broadcasted{Style, Axes, F, ARGS},
) where {Style, Axes, F, ARGS}
    new_args = UnrolledUtilities.unrolled_map(recursively_replace_fd_ops, bc.args)
    return Base.Broadcast.Broadcasted{Style}(bc.f, new_args, bc.axes)
end

recursively_replace_fd_ops(
    bc::StencilBroadcasted{Style, Op, Args, Axes, Work},
) where {Style, Op <: FDOperatorMatrix, Args, Axes, Work} = bc
function recursively_replace_fd_ops(
    bc::StencilBroadcasted{Style, Op, Args, Axes, Work},
) where {Style, Op, Args, Axes, Work}
    new_args = UnrolledUtilities.unrolled_map(recursively_replace_fd_ops, bc.args)
    return StencilBroadcasted{Style, Op, typeof(new_args), Axes, Work}(
        bc.op,
        new_args,
        bc.axes,
        bc.work,
    )
end

function recursively_replace_fd_ops(
    bc::StencilBroadcasted{Style, Op, Args, Axes, Work},
) where {Style, Op <: TwoArgFDOperator, Args, Axes, Work}
    if !has_affine_bc(bc.op) && check_if_fits_in_shmem(bc.args[end])
        opmat = Base.Broadcast.broadcasted(
            FDOperatorMatrix(bc.op),
            recursively_replace_fd_ops(bc.args[1]),
        )
        new_args = (opmat, recursively_replace_fd_ops(bc.args[end]))
        newop = MultiplyColumnwiseBandMatrixField()
        return StencilBroadcasted{Style, typeof(newop), typeof(new_args), Axes, Work}(
            newop,
            new_args,
            bc.axes,
            bc.work,
        )
    else
        # TODO: add SetValue bc support for this case as well
        return bc
    end
end

function recursively_replace_fd_ops(
    bc::StencilBroadcasted{Style, Op, Args, Axes, Work},
) where {Style, Op <: OneArgFDOperator, Args, Axes, Work}
    if !has_affine_bc(bc.op) && check_if_fits_in_shmem(bc.args[1])
        opmat = Base.Broadcast.broadcasted(
            FDOperatorMatrix(bc.op),
            Fields.local_geometry_field(operator_input_space(bc.op, axes(bc.args[end]))),
        )
        new_args = (opmat, recursively_replace_fd_ops(bc.args[1]))
        newop = MultiplyColumnwiseBandMatrixField()
        return StencilBroadcasted{Style, typeof(newop), typeof(new_args), Axes, Work}(
            newop,
            new_args,
            bc.axes,
            bc.work,
        )
    elseif check_if_fits_in_shmem(bc.args[1]) && UnrolledUtilities.unrolled_all(
        Base.Fix2(isa, Operators.SetValue),
        values(bc.op.bcs),
    )
        # SetBoundaryOperator is either the inner or outer depending on if the operator takes input from faces or centers
        if bc.op isa MatrixFields.OneArgFDOperatorWithCenterInput
            opmat = Base.Broadcast.broadcasted(
                FDOperatorMatrix(bc.op),
                Fields.local_geometry_field(
                    operator_input_space(bc.op, axes(bc.args[end])),
                ),
            )
            inner_args = (opmat, recursively_replace_fd_ops(bc.args[1]))
            inner_op = MultiplyColumnwiseBandMatrixField()
            inner_bc =
                StencilBroadcasted{Style, typeof(inner_op), typeof(inner_args), Axes, Work}(
                    inner_op,
                    inner_args,
                    bc.axes,
                    bc.work,
                )
            outer_op = Operators.SetBoundaryOperator(; bc.op.bcs...)
            return StencilBroadcasted{
                Style,
                typeof(outer_op),
                Tuple{typeof(inner_bc)},
                Axes,
                Work,
            }(
                outer_op,
                (inner_bc,),
                bc.axes,
                bc.work,
            )
        else
            opmat = Base.Broadcast.broadcasted(
                FDOperatorMatrix(Utilities.unionall_type(typeof(bc.op))()),
                Fields.local_geometry_field(
                    operator_input_space(bc.op, axes(bc.args[end])),
                ),
            )
            inner_args = (recursively_replace_fd_ops(bc.args[1]),)
            inner_op = Operators.SetBoundaryOperator(; bc.op.bcs...)
            inner_axes = axes(inner_args[1])
            inner_bc = StencilBroadcasted{
                Style,
                typeof(inner_op),
                typeof(inner_args),
                typeof(inner_axes),
                Work,
            }(
                inner_op,
                inner_args,
                inner_axes,
                bc.work,
            )
            outer_args = (opmat, inner_bc)
            outer_op = MultiplyColumnwiseBandMatrixField()
            return StencilBroadcasted{
                Style,
                typeof(outer_op),
                typeof(outer_args),
                Axes,
                Work,
            }(
                outer_op,
                outer_args,
                bc.axes,
                bc.work,
            )
        end
    else
        # affine BCs with non-SetValue BCs, or values that won't fit in shmmem
        return bc
    end
end

"""
    new_stencil_entry!(out, bc::BC, space)

CUDA kernel to compute the value of a `Broadcasted` or `StencilBroadcasted` at a single index.
This should only be used when there are 63 vertical elements.
This calls `calc_level_val(bc, space)`, which  computes the value of the broadcasted expression at the given index,
and then copies the result into `out`.
"""
Base.@propagate_inbounds function new_stencil_entry!(out, bc::BC, space) where {BC}
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    val = @inbounds @inline calc_level_val(bc, space)
    if space.staggering isa ClimaCore.Grids.CellFace
        @inline @inbounds setidx!(space, out, v - half, hidx, val)
    else
        if v != Int32(64)
            @inline @inbounds setidx!(space, out, v, hidx, val)
        end
    end
    return nothing
end

# All the functions below this line should not be used outside of this file

"""
    calc_level_val(bc, space)

Call `calc_level_val` on all the arguments of `bc`, and then apply the function `bc.f` to the results.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    space,
) where {BC <: Base.Broadcast.Broadcasted}
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    @inline @inbounds resolved_args = UnrolledUtilities.unrolled_map(
        Base.Fix2(reconstruct_space_and_call_calc_level_val, space),
        bc.args,
    )
    return @inline @inbounds bc.f(resolved_args...)
end

"""
    reconstruct_space_and_call_calc_level_val(arg, space)

If `arg` is a `Broadcasted`, `StencilBroadcasted`, or `Field`,
reconstruct the space for the argument and call `calc_level_val` on it. This allows
us to use Base.Fix2.
"""
Base.@propagate_inbounds reconstruct_space_and_call_calc_level_val(
    arg::A,
    space::S,
) where {
    A <: Union{Base.Broadcast.Broadcasted, StencilBroadcasted, Field},
    S,
} = calc_level_val(arg, reconstruct_placeholder_space(axes(arg), space))
Base.@propagate_inbounds reconstruct_space_and_call_calc_level_val(
    arg::A,
    space::S,
) where {A, S} = calc_level_val(arg, space)

"""
    calc_level_val(val::T, space)

If `val` is not a `Broadcasted`, `StencilBroadcasted`, or `Field`, just return `val`.
If it is a `Ref`, return `val[]`. If it is a one element tuple, return the element.
"""
Base.@propagate_inbounds calc_level_val(val::T, space) where {T <: Ref} = val[]
Base.@propagate_inbounds calc_level_val(val::T, space) where {V, T <: Tuple{V}} =
    val[Int32(1)]
calc_level_val(arg::S, space) where {S} = arg

"""
    calc_level_val(bc::StencilBroadcasted{<:Any, <: MultiplyColumnwiseBandMatrixField}, space)

Call `calc_level_val` on both args of `bc`, place the result of the second arg into shared memory,
and then perform the multiplication.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    space,
) where {
    S,
    Op <: MultiplyColumnwiseBandMatrixField,
    BC <: StencilBroadcasted{S, Op},
}
    v = threadIdx().x
    i = threadIdx().y
    mat1_space =
        reconstruct_placeholder_space(axes(bc.args[Int32(1)]), space)
    @inline @inbounds mat2_space =
        reconstruct_placeholder_space(axes(bc.args[Int32(2)]), space)

    mat2_row = @inline @inbounds calc_level_val(bc.args[Int32(2)], mat2_space)
    mat1_row = @inline @inbounds calc_level_val(bc.args[Int32(1)], mat1_space)
    # project before placing in shared memory to avoid projecting multiple times
    mat2_row_converted =
        @inline @inbounds project_row2_for_mul(mat1_row, mat2_row, mat2_space)
    # It should be possible to use static shared memory here, but it allocates new shared memory
    # for each layer of recursion
    CUDA.sync_threads()
    # it should be possible to use a multi dim shared array here as well, but it seems to
    # cause some weird issues with the indexing, so I'm just using a 1D array and indexing manually
    mat2 = CUDA.CuDynamicSharedArray(typeof(mat2_row_converted), Int32(256))
    @inbounds mat2[v  + (i - 1) * 64] = mat2_row_converted
    CUDA.sync_threads()
    # if the output is on centers, the 64th thread can just return 0
    mat1_space.staggering isa Spaces.CellCenter && v == Int32(64) &&
        return rzero(eltype(bc))
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
        return @inline @inbounds row_mul_mat!(
            eltype(bc),
            mat1_row,
            mat2,
            mat1_shape,
            mat2_shape,
        )
    else
        # mat * vec case
        out = @inline @inbounds row_mul_vec!(eltype(bc), mat1_row, mat2, mat1_shape)
        # v==64 && blockIdx().z == 1 && i == 1 && blockIdx().y == 1  && @cushow out
        # if !(-0.001f0 < out - mat2_row_converted < 0.001f0)
        #     blockIdx().z == 1 && i == 1 && blockIdx().y == 1  && @cushow v
        #     blockIdx().z == 1 && i == 1 && blockIdx().y == 1  && @cushow out
        #     # blockIdx().z == 1 && i == 1 && blockIdx().y == 1  && @cushow out - mat2_row_converted
        # end
        return out
    end
end

"""
    calc_level_val(bc::StencilBroadcasted{<:Any, <: SetBoundaryOperator}, space)

Special case of `calc_level_val` for `SetBoundaryOperator`s, which just applies the BC.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    space,
) where {S, Op <: Operators.SetBoundaryOperator, BC <: StencilBroadcasted{S, Op}}
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    # we know it is on face staggered space
    v_half = v - half
    inner_val = @inline @inbounds calc_level_val(bc.args[Int32(1)], space)
    if Operators.should_call_left_boundary(v_half, space, bc.op, nothing)
        lloc = Operators.left_boundary_window(space)
        left_bndry = Operators.get_boundary(bc.op, lloc)
        return @inbounds @inline calc_level_val(left_bndry.val, space)
    elseif Operators.should_call_right_boundary(v_half, space, bc.op, nothing)
        rroc = Operators.right_boundary_window(space)
        right_bndry = Operators.get_boundary(bc.op, rroc)
        return @inbounds @inline calc_level_val(right_bndry.val, space)
    else
        return @inbounds @inline inner_val
    end
end

"""
    calc_level_val(bc::StencilBroadcasted{<:Any, <: LinVanLeerC2F}, space)

Special case of `calc_level_val` for `LinVanLeerC2F`s, which makes the
top and bottom face values not use the fallback `Operators.getidx`, since that
will error if the operator is eagerly evaluated at the boundaries.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    space,
) where {S, Op <: Operators.LinVanLeerC2F, BC <: StencilBroadcasted{S, Op}}
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    if v == Int32(1)  || v == Int32(64)
        return zero(eltype(bc))
    end
    idx = v - half
    return @inline @inbounds getidx(space, bc, idx, hidx)
end

"""
    calc_level_val(bc::StencilBroadcasted, space)

Fallback case of `calc_level_val` that calls `Operators.getidx`. This is used for
affine BCs with non-SetValue BCs, or values that won't fit in shmmem.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    space,
) where {BC <: StencilBroadcasted}
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    if space.staggering isa Spaces.CellCenter
        v == Int32(64) && return @inline @inbounds rzero(eltype(bc))
    end
    li = space.staggering isa Spaces.CellCenter ? Int32(1) : half
    idx = v - Int32(1) + li
    return @inline @inbounds getidx(space, bc, idx, hidx)
end

"""
    calc_level_val(f::Field, space)

Returns the value of the field `f` at the thread's index.
When the staggering of `space` is `CellCenter`, the thread with `v == 64` returns `rzero(eltype(f))`
"""
Base.@propagate_inbounds function calc_level_val(
    arg::F,
    space,
) where {F <: Field}
    data = field_values(arg)
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    if space.staggering isa Spaces.CellCenter
        v == Int32(64) && return @inline @inbounds rzero(eltype(data))
    end
    return @inline @inbounds data[CartesianIndex(i, j, Int32(1), v, h)]
end

"""
    calc_level_val(bc::StencilBroadcasted{<:Any, <: FDOperatorMatrix}, space)

Return the correct row of the operator matrix for the current thread
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    space,
) where {
    S,
    BC <:
    StencilBroadcasted{S, <:FDOperatorMatrix},
}
    op = bc.op.op
    args = bc.args
    val = @inline @inbounds get_op_row(op, args, space)
    CUDA.sync_warp()
    return val
end

"""
    get_op_row(op, args, space)

Get the correct row of the operator matrix for the current thread, taking into account boundary conditions.
"""

Base.@propagate_inbounds function get_op_row(op, args, space)
    FT = Spaces.undertype(space)
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)

    outputs_to_face = space.staggering isa ClimaCore.Grids.CellFace
    row_type = op_matrix_row_type(op, FT, args[1:(end - 1)]...)
    if !outputs_to_face && v == Int32(64)
        return rzero(row_type)
    end
    v_half = outputs_to_face ? v - half : v
    in_left_bnd = Operators.should_call_left_boundary(v_half, space, op, nothing)
    in_right_bnd =
        Operators.should_call_right_boundary(v_half, space, op, nothing)
    op_matrix = FDOperatorMatrix(op)
    # boundaries can return different row types
    if in_left_bnd
        lloc = Operators.left_boundary_window(space)
        left_bndry = Operators.get_boundary(op, lloc)
        raw_val = Operators.stencil_left_boundary(
            op_matrix,
            left_bndry,
            space,
            v_half,
            hidx,
            args...,
        )
        val = convert(row_type, raw_val)
    elseif in_right_bnd
        rroc = Operators.right_boundary_window(space)
        right_bndry = Operators.get_boundary(op, rroc)
        raw_val = Operators.stencil_right_boundary(
            op_matrix,
            right_bndry,
            space,
            v_half,
            hidx,
            args...,
        )
        val = convert(row_type, raw_val)
    else
        raw_val =
            Operators.stencil_interior(op_matrix, space, v_half, hidx, args...)
        val = convert(row_type, raw_val)
    end
    return val
end


"""
    project_row2_for_mul

Project's `mat2_row` onto the correct axis for multiplication with `mat1_row` if necessary, and returns the projected row.
"""
Base.@propagate_inbounds function project_row2_for_mul(mat1_row, mat2_row, space)
    if !ClimaCore.Geometry.needs_projection(typeof(mat1_row), typeof(mat2_row))
        return mat2_row
    end
    v = threadIdx().x
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    project_onto =
        ClimaCore.Geometry.recursively_find_dual_axes_for_projection(typeof(mat1_row))
    if space.staggering isa Spaces.CellCenter && v == Int32(64)
        lg = rzero(Spaces.local_geometry_type(typeof(space)))
    else
        v_maybe_half = space.staggering isa Spaces.CellFace ? v - half : v
        @inbounds lg = Geometry.LocalGeometry(space, v_maybe_half, hidx)
    end
    # put needed info into tuple so we can use Base.Fix2
    projection_tuple = (project_onto, lg)
    return @inline @inbounds ClimaCore.MatrixFields.recursively_project(
        projection_tuple,
        mat2_row,
    )
end
