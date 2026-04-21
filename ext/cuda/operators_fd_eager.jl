import ClimaCore: Spaces, Quadratures, Topologies, Operators
import Base.Broadcast: Broadcasted
import ClimaCore.Fields: Field, field_values, AbstractFieldStyle
import ClimaComms
import ClimaCore.Utilities: half
import ClimaCore.Operators
import ClimaCore.Geometry: ⊗, project
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠, rmuladd, rmap
import ClimaCore.Operators:
    StencilBroadcasted, setidx!, getidx, reconstruct_placeholder_space
import ClimaCore.MatrixFields: FaceToCenter, CenterToFace, Square, CenterToCenter,
    FaceToFace, TwoArgFDOperator, OneArgFDOperator, has_affine_bc, FDOperatorMatrix,
    MultiplyColumnwiseBandMatrixField, operator_input_space, op_matrix_row_type,
    BandMatrixRow
using ClimaCore.MatrixFields
import ClimaCore.Utilities
import ClimaCore
using ClimaCore.MatrixFields
using ClimaCore.Geometry
using LinearAlgebra
import UnrolledUtilities


include("column_matrix_helpers.jl")

"""
    check_if_fits_in_shmem(x)

Check if `x`, or the `eltype(x)` can fit in shared memory with the current config.

The limit is currently set to 36 bytes. On an A100, each thread can use 32 bytes of shared
memory per thread before theoretical occupancy is limited. We use 36 bytes to allow caching of
3x3 axis tensors
"""
check_if_fits_in_shmem(bc::Union{StencilBroadcasted, Broadcasted, Field}) =
    sizeof(eltype(bc)) <= 36
check_if_fits_in_shmem(val) = sizeof(typeof(val)) <= 36

"""
    has_type_arg(x)
Check if `x` is a `Type`, or any of its arguments has a `Type` argument.
This is needed because both the shmem matrix multiplication and the getidx fallback rely on
`eltype`, and `eltype(::CudaRefType) = Any`
"""
has_type_arg(_) = false
has_type_arg(::Type) = true
has_type_arg(::Base.RefValue{<:Type}) = true
has_type_arg(bc::Union{StencilBroadcasted, Broadcasted}) =
    UnrolledUtilities.unrolled_any(has_type_arg, bc.args)

"""
    replace_fd_ops(val)

Recursively replace any `OneArgFDOperator` or `TwoArgFDOperator` in `val` with a
`MultiplyColumnwiseBandMatrixField` with the corresponding `FDOperatorMatrix`, if the operator
does not have affine BCs and the operator matrix fits in shared memory.
"""
replace_fd_ops(val) = val

function replace_fd_ops(
    bc::Base.Broadcast.Broadcasted,
)
    new_args = UnrolledUtilities.unrolled_map(replace_fd_ops, bc.args)
    return Base.Broadcast.Broadcasted{typeof(bc.style)}(bc.f, new_args, bc.axes)
end

replace_fd_ops(
    bc::StencilBroadcasted{Style, Op},
) where {Style, Op <: FDOperatorMatrix} = bc
function replace_fd_ops(
    bc::StencilBroadcasted{Style},
) where {Style}
    new_args = UnrolledUtilities.unrolled_map(replace_fd_ops, bc.args)
    return StencilBroadcasted{
        Style,
        typeof(bc.op),
        typeof(new_args),
        typeof(bc.axes),
        typeof(bc.work),
    }(
        bc.op,
        new_args,
        bc.axes,
        bc.work,
    )
end

function replace_fd_ops(
    bc::StencilBroadcasted{Style, Op},
) where {Style, Op <: TwoArgFDOperator}
    if !has_affine_bc(bc.op) && check_if_fits_in_shmem(bc.args[end]) &&
       !has_type_arg(bc.args[end])
        opmat = Base.Broadcast.broadcasted(
            FDOperatorMatrix(bc.op),
            replace_fd_ops(bc.args[1]),
        )
        new_args = (opmat, replace_fd_ops(bc.args[end]))
        newop = MultiplyColumnwiseBandMatrixField()
        return StencilBroadcasted{
            Style,
            typeof(newop),
            typeof(new_args),
            typeof(bc.axes),
            typeof(bc.work),
        }(
            newop,
            new_args,
            bc.axes,
            bc.work,
        )
    else
        # affine BCs or values that won't fit in shmmem
        return bc
    end
end

function replace_fd_ops(
    bc::StencilBroadcasted{Style, Op},
) where {Style, Op <: OneArgFDOperator}
    if !has_affine_bc(bc.op) && check_if_fits_in_shmem(bc.args[1]) &&
       !has_type_arg(bc.args[1])
        opmat = Base.Broadcast.broadcasted(
            FDOperatorMatrix(bc.op),
            Fields.local_geometry_field(operator_input_space(bc.op, axes(bc.args[end]))),
        )
        new_args = (opmat, replace_fd_ops(bc.args[1]))
        newop = MultiplyColumnwiseBandMatrixField()
        return StencilBroadcasted{
            Style,
            typeof(newop),
            typeof(new_args),
            typeof(bc.axes),
            typeof(bc.work),
        }(
            newop,
            new_args,
            bc.axes,
            bc.work,
        )
    else
        # affine BCs or values that won't fit in shmmem
        return bc
    end
end

"""
    eager_copyto_stencil_kernel!(out, bc::BC, space)

CUDA kernel to compute the value of a `Broadcasted` or `StencilBroadcasted` at a single index.
This calls `calc_level_val(bc, space)`, which  computes the value of the broadcasted
expression at the given index, and then copies the result into `out`.
"""
Base.@propagate_inbounds function eager_copyto_stencil_kernel!(
    out,
    bc::BC,
    space,
) where {BC}
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    val = @inbounds @inline calc_level_val(bc, space)
    if space.staggering isa ClimaCore.Grids.CellFace
        @inbounds @inline setidx!(space, out, v - half, hidx, val)
    else
        if v != CUDA.blockDim().x
            @inbounds @inline setidx!(space, out, v, hidx, val)
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
    resolved_args = @inbounds @inline UnrolledUtilities.unrolled_map(
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
    A <: Union{Base.Broadcast.Broadcasted{<:AbstractFieldStyle}, StencilBroadcasted, Field},
    S,
} = @inbounds @inline calc_level_val(arg, reconstruct_placeholder_space(axes(arg), space))
Base.@propagate_inbounds reconstruct_space_and_call_calc_level_val(
    arg::A,
    space::S,
) where {A, S} = @inbounds @inline calc_level_val(arg, space)

"""
    calc_level_val(val::T, space)

If `val` is not a `Broadcasted`, `StencilBroadcasted`, or `Field`, just return `val`.
If it is a `Ref`, return `val[]`. If it is a one element tuple, return the element.
"""
Base.@propagate_inbounds calc_level_val(val::T, space) where {T <: Ref} = val[]
Base.@propagate_inbounds calc_level_val(val::T, space) where {V, T <: Tuple{V}} =
    first(val)
Base.@propagate_inbounds calc_level_val(arg::S, space) where {S} = arg

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
    if check_if_fits_in_shmem(bc.args[2i32])
        v = threadIdx().x
        i = threadIdx().y
        mat1_space =
            reconstruct_placeholder_space(axes(bc.args[1i32]), space)
        mat2_space =
            reconstruct_placeholder_space(axes(bc.args[2i32]), space)

        mat2_row = calc_level_val(bc.args[2i32], mat2_space)
        mat1_row = calc_level_val(bc.args[1i32], mat1_space)
        # project before placing in shared memory to avoid projecting multiple times
        mat2_row_converted =
            @inbounds @inline project_row2_for_mul(mat1_row, mat2_row, mat2_space)
        # It should be possible to use static shared memory here, but it allocates new shared memory
        # for each layer of recursion
        CUDA.sync_threads()
        # it should be possible to use a multi dim shared array here as well, but it seems to
        # cause some weird issues with the indexing, so I'm just using a 1D array and indexing manually
        mat2 = CUDA.CuDynamicSharedArray(
            typeof(mat2_row_converted),
            CUDA.blockDim().x * CUDA.blockDim().y,
        )
        @inbounds mat2[v + (i - 1) * CUDA.blockDim().x] = mat2_row_converted
        CUDA.sync_threads()
        # if the output is on centers, the CUDA.blockDim().xth thread can just return 0
        mat1_space.staggering isa Spaces.CellCenter && v == CUDA.blockDim().x &&
            return new_struct(eltype(bc))
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
            )
            out isa eltype(bc) || return convert(eltype(bc), out)
            return out
        else
            # mat * vec case
            out = @inbounds @inline row_mul_vec!(eltype(bc), mat1_row, mat2, mat1_shape)
            out isa eltype(bc) || return convert(eltype(bc), out)
            return out
        end
    else
        # values that won't fit in shmmem should just call getidx
        i = threadIdx().y
        j = blockIdx().y
        v = threadIdx().x
        h = blockIdx().z
        hidx = (i, j, h)
        if space.staggering isa Spaces.CellCenter
            v == CUDA.blockDim().x && return @inline @inbounds new_struct(eltype(bc))
        end
        li = space.staggering isa Spaces.CellCenter ? 1i32 : half
        idx = v - 1i32 + li
        return @inbounds @inline getidx(space, bc, idx, hidx)
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
    if v == 1i32 || v == CUDA.blockDim().x
        return zero(eltype(bc))
    end
    idx = v - half
    return @inbounds @inline getidx(space, bc, idx, hidx)
end

"""
    calc_level_val(bc::StencilBroadcasted, space)

Fallback case of `calc_level_val` that calls `Operators.getidx`. This is used for
affine BCs or values that won't fit in shmmem.
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
        v == CUDA.blockDim().x && return @inline @inbounds new_struct(eltype(bc))
    end
    li = space.staggering isa Spaces.CellCenter ? 1i32 : half
    idx = v - 1i32 + li
    return @inbounds @inline getidx(space, bc, idx, hidx)
end

"""
    calc_level_val(f::Field, space)

Returns the value of the field `f` at the thread's index.
When the staggering of `space` is `CellCenter`, the thread with `v == CUDA.blockDim().x` returns `new_struct(eltype(f))`
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
    if space isa
       Union{Spaces.ExtrudedFiniteDifferenceSpace, Spaces.FiniteDifferenceSpace} &&
       space.staggering isa Spaces.CellCenter
        v == CUDA.blockDim().x && return @inline @inbounds new_struct(eltype(data))
    end
    return @inline @inbounds data[CartesianIndex(i, j, 1i32, v, h)]
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
    val = @inbounds @inline get_op_row(op, args, space)
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
    row_type = @inbounds @inline op_matrix_row_type(op, FT, args[1:(end - 1)]...)
    if !outputs_to_face && v == CUDA.blockDim().x
        return new_struct(row_type)
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
    project_row2_for_mul

Projects `mat2_row` onto the correct axis for multiplication with `mat1_row` if necessary, and returns the projected row.
"""
Base.@propagate_inbounds function project_row2_for_mul(mat1_row, mat2_row, space)
    mat1_et = mat1_row isa BandMatrixRow ? eltype(mat1_row) : typeof(mat1_row)
    mat2_et = mat2_row isa BandMatrixRow ? eltype(mat2_row) : typeof(mat2_row)
    if !ClimaCore.Geometry.needs_projection(mat1_et, mat2_et)
        return mat2_row
    end
    v = threadIdx().x
    i = threadIdx().y
    j = blockIdx().y
    v = threadIdx().x
    h = blockIdx().z
    hidx = (i, j, h)
    project_onto =
        ClimaCore.Geometry.recursively_find_dual_axes_for_projection(mat1_et)
    if space.staggering isa Spaces.CellCenter && v == CUDA.blockDim().x
        lg = new_struct(Spaces.local_geometry_type(typeof(space)))
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

Recursively project `y` onto the axes in `projection_tuple[1]` using the local geometry in
`projection_tuple[2]`.
"""
Base.@propagate_inbounds recursively_project(
    projection_tuple::T,
    y::Y,
) where {T, Y <: BandMatrixRow} = map(Base.Fix1(recursively_project, projection_tuple), y)
Base.@propagate_inbounds recursively_project(projection_tuple::T, y::Y) where {T, Y} =
    rmap(Base.Fix1(recursively_project, projection_tuple), y)
Base.@propagate_inbounds recursively_project(
    projection_tuple::T,
    y::Y,
) where {T, Y <: AxisTensor} =
    @inbounds @inline project(projection_tuple[1], y, projection_tuple[2])

@generated new_struct(::Type{T}) where {T} = Expr(:new, :T)

if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(recursively_project)
        m.recursion_relation = dont_limit
    end
    for m in methods(calc_level_val)
        m.recursion_relation = dont_limit
    end
    for m in methods(outer_or_mul)
        m.recursion_relation = dont_limit
    end
end
