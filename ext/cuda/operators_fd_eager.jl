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

Two kinds of node cache a result in shared memory: a
`MultiplyColumnwiseBandMatrixField` stencil operation caches the projected row of its
second argument (see `cached_operand_type`), and an `AdvectionOperator` caches its
level's advected value, and velocity where it needs one (see
`advection_shmem_entry_type`). Every other node needs no shared memory of its own but
is still traversed for nested nodes that do. The launch configuration multiplies this
value by the number of threads per block to size the dynamic shared memory, so that any
single node's result is guaranteed to fit and can always be cached. Each node allocates
that memory at offset 0, so nodes reuse it and must synchronize before writing.

Returns `nothing` when any cached entry type cannot be sized (its inference gave a
non-concrete type; see `cached_operand_type`), in which case the caller must fall back
to the lazy `copyto_stencil_kernel!` instead of launching the eager kernel.
"""
max_eager_shmem_per_thread(x) = 0
max_eager_shmem_per_thread(bc::Union{Broadcasted, StencilBroadcasted}) =
    _max_eager_shmem_over_args(bc.args)
max_eager_shmem_per_thread(
    bc::StencilBroadcasted{S, <:MultiplyColumnwiseBandMatrixField},
) where {S} =
    _shmem_max(
        _sizeof_or_nothing(cached_operand_type(bc)),
        _max_eager_shmem_over_args(bc.args),
    )
max_eager_shmem_per_thread(
    bc::StencilBroadcasted{S, <:Operators.AdvectionOperator},
) where {S} = _shmem_max(
    _sizeof_or_nothing(advection_shmem_entry_type(bc)),
    _max_eager_shmem_over_args(bc.args),
)

_max_eager_shmem_over_args(args::Tuple) = UnrolledUtilities.unrolled_mapreduce(
    max_eager_shmem_per_thread,
    _shmem_max,
    args;
    init = 0,
)

# `max` over byte counts, with `nothing` (unsizeable) as an absorbing element.
_shmem_max(bytes1, bytes2) =
    isnothing(bytes1) || isnothing(bytes2) ? nothing : max(bytes1, bytes2)

_sizeof_or_nothing(::Nothing) = nothing
_sizeof_or_nothing(::Type{T}) where {T} = isconcretetype(T) ? sizeof(T) : nothing

"""
    cached_operand_type(bc)

Return the type that `calc_level_val` writes into shared memory for the multiplication
`bc`, i.e. `typeof(project_row2_for_mul(mat1_row, mat2_row, hidx, mat2_space))`.

The size cannot be read off the second operand directly, because
`project_row2_for_mul` projects every tensor leaf of that operand onto the axis dual
to the first operand's entries, which changes its size in either direction: a
`Covariant1Vector` widens to a `Contravariant123Vector`, while a `Covariant12Vector`
narrows to a `Contravariant1Vector`. For a matrix-matrix product those leaves are also
nested inside a `BandMatrixRow`, so no property of the operand's outermost type
bounds the projected size. This function mirrors the type-level logic of
`project_row2_for_mul` and infers the projected type, so the buffer is always big enough
for what the kernel writes into it.

The kernel's multiply handler allocates its shared memory buffer with this same
function, so the launch-time sizing and the device-side element type cannot go out
of sync (see `advection_shmem_entry_type` for the same pattern).

Returns `nothing` when the type cannot be determined (an operand's eltype is the
inference-failure sentinel `Union{}`, or inference of the projection gave a
non-concrete type); the launch then falls back to the lazy `copyto_stencil_kernel!`,
which needs no shared memory, instead of erroring.

The single-argument form is used by the launch-side sizing, where `bc` is not yet
space-stripped; the kernel passes the reconstructed operand space explicitly, since
a stripped argument's `axes` is a placeholder space with no local geometry type.
"""
@inline cached_operand_type(bc) = cached_operand_type(bc, axes(bc.args[2i32]))
@inline function cached_operand_type(bc, mat2_space)
    mat1_type = unsafe_eltype(bc.args[1i32])
    mat2_type = unsafe_eltype(bc.args[2i32])
    (isconcretetype(mat1_type) && isconcretetype(mat2_type)) || return nothing
    mat1_et = mat1_type <: BandMatrixRow ? eltype(mat1_type) : mat1_type
    project_onto =
        ClimaCore.Geometry._dual_axes_for_projection(mat1_et)
    isnothing(project_onto) && return mat2_type
    lg_type = Spaces.local_geometry_type(typeof(mat2_space))
    # Core.Compiler.return_type rather than Utilities.return_type: the latter
    # throws an InferenceError on a non-concrete result, and the caller falls
    # back to the lazy kernel instead.
    projected_type = Core.Compiler.return_type(
        recursively_project,
        Tuple{Tuple{typeof(project_onto), lg_type}, mat2_type},
    )
    isconcretetype(projected_type) || return nothing
    return projected_type
end


"""
    advection_shmem_entry_type(bc)

Return the type that the `AdvectionOperator` method of `calc_level_val` writes into
shared memory for `bc`, per thread: the advected field's value at the thread's center
level, prepended with the contravariant3 velocity component at the thread's face when
the operator also needs the velocity at neighboring faces
(`advection_velocity_width(op) == Val(:neighboring)`). The velocity slot holds
`Geometry.contravariant3(velocity_val, lg)`, whose type is
`Operators.velocity_component_type` of the velocity's element type — for a tuple-valued
velocity that is an `AutoBroadcaster` of the components' scalars, not the tuple's single
component type (which is all `Base.eltype` would give). The kernel allocates its shared
memory buffer with this same function, so the launch-time sizing and the device-side
element type cannot go out of sync.
"""
@inline function advection_shmem_entry_type(bc)
    x_type = unsafe_eltype(bc.args[2i32])
    v_type = unsafe_eltype(bc.args[1i32])
    # unsafe_eltype may return the inference-failure sentinel Union{}, which would
    # dispatch into the AutoBroadcaster method of velocity_component_type (Union{} is
    # a subtype of everything); propagate it so the launch falls back to the lazy
    # kernel (`_sizeof_or_nothing` treats any non-concrete type as unsizeable).
    v_type == Union{} && return Union{}
    v3_type = Operators.velocity_component_type(v_type)
    return Operators.advection_velocity_width(bc.op) isa Val{:neighboring} ?
           Tuple{v3_type, x_type} : x_type
end

ClimaCore.Utilities.unsafe_eltype(::CUDA.CuRefType{T}) where {T} = T

"""
    has_padding_thread(space)

Return whether the last x-thread of the eager kernel maps to no output level of `space`.

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

Compute the value of the `Broadcasted` or `StencilBroadcasted` expression `bc` at the
current thread's index and copy it into `out`; this is the CUDA kernel of the eager
finite-difference path. The value is computed by `calc_level_val(bc, hidx, space)`.
"""
Base.@propagate_inbounds function eager_copyto_stencil_kernel!(
    out,
    bc::BC,
    mask,
    space,
) where {BC}
    v = threadIdx().x
    col_idx = threadIdx().y + (blockIdx().x - 1) * blockDim().y
    # Out-of-range columns must not exit early: the shmem handlers in `calc_level_val`
    # contain `sync_threads()` barriers, and a barrier that only part of a block reaches
    # is undefined behavior before sm_70 (exited threads only implicitly satisfy
    # `bar.sync` on Volta and later). Instead, out-of-range threads compute a valid
    # dummy column -- all x-threads of a y-row share `col_idx`, so a dummy column's
    # shared-memory slice is written and read only by its own threads -- and the result
    # is discarded at the store below.
    (in_range, (i, j, h)) = if mask isa NoMask
        # `Ni` and `Nj` are read off the output layout's type parameters (see
        # `vijh_params`), so they are compile-time constants and the `CartesianIndices`
        # decomposition below is a fixed-divisor `divrem`. Only `Nh` is a runtime value,
        # and being the last extent it is never divided by.
        size_params = ClimaCore.DataLayouts.vijh_params(ClimaCore.Fields.field_values(out))
        Nj = size_params.Nj
        Ni = size_params.Ni
        Nh = ClimaCore.DataLayouts.nelems(ClimaCore.Fields.field_values(out))
        cart_inds = CartesianIndices((Ni, Nj, Nh))
        in_range = col_idx <= length(cart_inds)
        (in_range, @inbounds(cart_inds[in_range ? col_idx : one(col_idx)].I))
    else
        (; N, i_map, j_map, h_map) = mask
        # Bound by the active-column count `N`, not `length(i_map)`: the maps
        # are allocated with one entry per column of the layout, but
        # `set_mask_maps!` only writes the first `N` entries, and the launch
        # rounds the grid up to a multiple of `blockDim().y` columns.
        in_range = @inbounds col_idx <= N[1]
        ijh = if in_range
            @inbounds (i_map[col_idx], j_map[col_idx], h_map[col_idx])
        else
            # Column (1, 1, 1) always exists in the allocated layout (even when
            # masked out, its memory is allocated); deriving the dummy from
            # constants avoids reading map entries beyond `N` that
            # `set_mask_maps!` never wrote.
            (one(eltype(i_map)), one(eltype(j_map)), one(eltype(h_map)))
        end
        (in_range, ijh)
    end
    hidx = (i, j, h)
    val = @inbounds @inline calc_level_val(bc, hidx, space)
    if in_range
        if space.staggering isa ClimaCore.Grids.CellFace
            @inbounds @inline setidx!(space, out, v - half, hidx, val)
        elseif !(has_padding_thread(space) && v == CUDA.blockDim().x)
            @inbounds @inline setidx!(space, out, v, hidx, val)
        end
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

If `arg` is a `Broadcasted`, `StencilBroadcasted`, or `Field`, reconstruct the space for
the argument and call `calc_level_val` on it. The tuple argument allows the function to
be used with `Base.Fix2`.
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

If `val` is not a `Broadcasted`, `StencilBroadcasted`, or `Field`, return `val`. If it
is a `Ref`, return `val[]`. If it is a one-element tuple, return the element.
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
    # sync before writing so that no thread is still reading a previous user of
    # the shared memory region (every handler allocates it at offset 0)
    CUDA.sync_threads()
    # The region is dynamic because static shared memory would allocate a new
    # one for each layer of recursion, and it is a 1D array indexed manually
    # because a multi-dimensional shared array indexes incorrectly here.
    # Allocate with the same `cached_operand_type` the launch-side sizing used, so
    # the buffer's element size cannot go out of sync with the sizing; writing
    # `mat2_row_converted` into it converts (a no-op when inference matched the
    # runtime type, and a loud conversion error -- rather than silent shared-memory
    # corruption -- if the two ever diverge).
    mat2 = CUDA.CuDynamicSharedArray(
        cached_operand_type(bc, mat2_space),
        CUDA.blockDim().x * CUDA.blockDim().y,
    )
    @inbounds mat2[v + (block_col_idx - 1) * CUDA.blockDim().x] = mat2_row_converted
    CUDA.sync_threads()
    # If the output is on centers, the padding thread (index CUDA.blockDim().x) returns 0.
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

Compute the value of a `SetBoundaryOperator` at the current thread's level. The operator
modifies only the two boundary levels of the space it is applied to and is the identity
in the interior: at the boundaries the value comes from `stencil_left_boundary` or
`stencil_right_boundary` (which extract and project the boundary value), and in the
interior the eagerly computed value of the argument is reused.
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
    # the compile-time staggering type keeps the two staggerings apart, so the `PlusHalf`
    # face index only reaches `should_call_*_boundary` when compiling for a face space and
    # its `idx < left_interior_idx` comparison never mixes a `PlusHalf` with an integer
    # center index -- which would pull in non-GPU-compatible error-formatting code.
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
    calc_level_val(bc::StencilBroadcasted{<:Any, <:AdvectionOperator}, hidx, space)

Compute the value of an `AdvectionOperator` at the current thread's face level. Each
thread computes the velocity (converted to its contravariant3 component) and the
advected field at its own level, caches them in shared memory, and then gathers the
4 neighboring center values (and, for operators with
`advection_velocity_width(op) == Val(:neighboring)`, the 2 neighboring face velocities)
that `stencil_interior` would have read, so each argument sub-expression is evaluated
once per level instead of once per stencil offset. Extra parameters beyond the velocity
and advected field are evaluated at the thread's face and passed through unmodified,
matching `stencil_interior`.

The operator's output space is a face space, so every x-thread maps to a valid face
and computes a value; only the advected (center-valued) argument has a padding thread,
whose garbage entry is cached but never read because the window indices are clamped to
the center range (or wrapped, on periodic spaces) exactly like `stencil_interior`'s.
"""
Base.@propagate_inbounds function calc_level_val(
    bc::BC,
    hidx,
    space,
) where {
    S,
    Op <: Operators.AdvectionOperator,
    BC <: StencilBroadcasted{S, Op},
}
    op = bc.op
    v = threadIdx().x
    block_col_idx = threadIdx().y
    n_faces = CUDA.blockDim().x
    periodic = Topologies.isperiodic(space)
    width = Operators.advection_velocity_width(op)
    velocity_space = reconstruct_placeholder_space(axes(bc.args[1i32]), space)
    arg_space = reconstruct_placeholder_space(axes(bc.args[2i32]), space)
    velocity_val =
        @inbounds @inline calc_level_val(bc.args[1i32], hidx, velocity_space)
    arg_val = @inbounds @inline calc_level_val(bc.args[2i32], hidx, arg_space)
    params = @inbounds @inline UnrolledUtilities.unrolled_map(
        Base.Fix2(reconstruct_space_and_call_calc_level_val, (hidx, space)),
        Base.tail(Base.tail(bc.args)),
    )
    @inbounds lg = Geometry.LocalGeometry(velocity_space, v - half, hidx)
    v³ = Geometry.contravariant3(velocity_val, lg)
    # sync before writing so that no thread is still reading a previous user of the
    # shared memory region (every handler allocates it at offset 0)
    CUDA.sync_threads()
    shmem = CUDA.CuDynamicSharedArray(
        advection_shmem_entry_type(bc),
        CUDA.blockDim().x * CUDA.blockDim().y,
    )
    col_offset = (block_col_idx - 1i32) * n_faces
    @inbounds shmem[v + col_offset] = advection_shmem_entry(width, v³, arg_val)
    CUDA.sync_threads()
    stencil_vals = @inbounds advection_gather(
        width,
        op,
        space,
        shmem,
        v³,
        v,
        n_faces,
        periodic,
        col_offset,
    )
    return Geometry.Contravariant3Vector(op(stencil_vals..., params...))
end

@inline advection_shmem_entry(::Val{:current}, v³, arg_val) = arg_val
@inline advection_shmem_entry(::Val{:neighboring}, v³, arg_val) = (v³, arg_val)

"""
    advection_center_window(v, n_faces, periodic)

Return the shared-memory indices of the 4 center-level stencil values around the face of
x-thread `v` (the thread that holds center level `v`; the face index is `v - half`). Mirrors
`stencil_interior(::AdvectionOperator, ...)`: out-of-range center indices are
clamped to the domain on non-periodic spaces (padding the ghost cells with the closest
interior value) and wrap around on periodic ones.
"""
@inline function advection_center_window(v, n_faces, periodic)
    if periodic
        (mod1(v - 2i32, n_faces), mod1(v - 1i32, n_faces), v, mod1(v + 1i32, n_faces))
    else
        # center levels span 1:n_faces - 1, and v - 2 and v - 1 are already below the
        # upper limit (v ≤ n_faces), so they only need the lower clamp
        n_centers = n_faces - 1i32
        (
            max(v - 2i32, 1i32),
            max(v - 1i32, 1i32),
            min(v, n_centers),
            min(v + 1i32, n_centers),
        )
    end
end

# Ghost-point extrapolation of the out-of-range stencil values, shared with
# the pointwise `stencil_interior(::AdvectionOperator, ...)`; the x-thread `v`
# holds the face `v - 1` faces in from the left boundary and `n_faces - v`
# faces in from the right one. A no-op on periodic spaces, whose indices wrap
# instead (`periodic` is a compile-time constant, so the branch folds).
@inline advection_ghost_values(op, space, v, n_faces, periodic, a⁻⁻, a⁻, a⁺, a⁺⁺) =
    periodic ? (a⁻⁻, a⁻, a⁺, a⁺⁺) :
    Operators.advection_ghost_values(
        op,
        space,
        v - 1i32,
        n_faces - v,
        a⁻⁻,
        a⁻,
        a⁺,
        a⁺⁺,
    )

Base.@propagate_inbounds function advection_gather(
    ::Val{:current},
    op,
    space,
    shmem,
    v³,
    v,
    n_faces,
    periodic,
    col_offset,
)
    (i⁻⁻, i⁻, i⁺, i⁺⁺) = advection_center_window(v, n_faces, periodic)
    a⁻⁻ = @inbounds shmem[i⁻⁻ + col_offset]
    a⁻ = @inbounds shmem[i⁻ + col_offset]
    a⁺ = @inbounds shmem[i⁺ + col_offset]
    a⁺⁺ = @inbounds shmem[i⁺⁺ + col_offset]
    a⁻⁻, a⁻, a⁺, a⁺⁺ =
        advection_ghost_values(op, space, v, n_faces, periodic, a⁻⁻, a⁻, a⁺, a⁺⁺)
    return (v³, a⁻⁻, a⁻, a⁺, a⁺⁺)
end

Base.@propagate_inbounds function advection_gather(
    ::Val{:neighboring},
    op,
    space,
    shmem,
    v³,
    v,
    n_faces,
    periodic,
    col_offset,
)
    (i⁻⁻, i⁻, i⁺, i⁺⁺) = advection_center_window(v, n_faces, periodic)
    # neighboring face indices, clamped/wrapped like `advection_velocities`
    iv⁻ = periodic ? mod1(v - 1i32, n_faces) : max(v - 1i32, 1i32)
    iv⁺ = periodic ? mod1(v + 1i32, n_faces) : min(v + 1i32, n_faces)
    a⁻⁻ = @inbounds shmem[i⁻⁻ + col_offset][2]
    a⁻ = @inbounds shmem[i⁻ + col_offset][2]
    a⁺ = @inbounds shmem[i⁺ + col_offset][2]
    a⁺⁺ = @inbounds shmem[i⁺⁺ + col_offset][2]
    a⁻⁻, a⁻, a⁺, a⁺⁺ =
        advection_ghost_values(op, space, v, n_faces, periodic, a⁻⁻, a⁻, a⁺, a⁺⁺)
    return @inbounds (
        shmem[iv⁻ + col_offset][1],
        v³,
        shmem[iv⁺ + col_offset][1],
        a⁻⁻,
        a⁻,
        a⁺,
        a⁺⁺,
    )
end

"""
    calc_level_val(bc::StencilBroadcasted, hidx, space)

Fallback method of `calc_level_val` that calls `Operators.getidx`. This is used for
affine boundary conditions and for values that do not fit in shared memory.
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

Fields whose space is missing one of the extruded space's dimensions hold a
single value along the missing dimensions, and are broadcast across them: a
level field has no vertical dimension, and a column field has no horizontal
dimensions. Those dimensions are read at index 1, matching `Operators.vidx` and
`Operators.hindices`.

The space gate below decides which case a field is by its space type: the
finite difference families in `Operators.AllFiniteDifferenceSpace` (extruded,
single-column, and multi-column) are read at the thread's level, and anything
else is treated as a field without a vertical dimension and read at level 1.
A new space family with a vertical dimension must be added to that union (and
audited here) before the `eager_supported` launch gate in
`operators_finite_difference.jl` lets it reach the eager kernel; otherwise it
would be misread as a level field.
"""
Base.@propagate_inbounds function calc_level_val(
    arg::F,
    hidx,
    space,
) where {F <: Field}
    data = field_values(arg)
    if space isa Operators.AllFiniteDifferenceSpace
        has_padding_thread(space) &&
            threadIdx().x == CUDA.blockDim().x &&
            return @inline @inbounds new(eltype(data))
        v = threadIdx().x
    else
        v = 1i32
    end
    # mirrors `Operators.hindices`: a single-column space holds one column at
    # (1, 1, 1); extruded and multi-column layouts are indexed by `hidx`
    (i, j, h) =
        space isa Spaces.FiniteDifferenceSpace ? (1i32, 1i32, 1i32) : hidx
    return @inline @inbounds data[v, i, j, h]
end

"""
    calc_level_val(bc::StencilBroadcasted{<:Any, <: FDOperatorMatrix}, hidx, space)

Return the row of the operator matrix for the current thread's level.
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
    op_matrix = bc.op
    args = bc.args
    val = @inbounds @inline get_op_row(op_matrix, args, hidx, space)
    return val
end

"""
    get_op_row(op_matrix, args, hidx, space)

Return the row of the operator matrix for the current thread's level, taking boundary
conditions into account.

This takes the broadcasted `FDOperatorMatrix` itself rather than rebuilding one from its
wrapped operator: the `FDOperatorMatrix` constructor carries a value-dependent
`has_affine_bc` check with an `@warn`, which cannot be compiled into device code when
the operator still holds a value-fixing boundary condition (as it does through the
public `MatrixFields.operator_matrix` API, which does not strip boundary conditions).
"""

Base.@propagate_inbounds function get_op_row(
    op_matrix::FDOperatorMatrix,
    args,
    hidx,
    space,
)
    op = op_matrix.op
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

Project `mat2_row` onto the axis dual to the entries of `mat1_row`, if a projection is
needed, and return the projected row.
"""
Base.@propagate_inbounds function project_row2_for_mul(mat1_row, mat2_row, hidx, space)
    mat1_et = mat1_row isa BandMatrixRow ? eltype(mat1_row) : typeof(mat1_row)
    project_onto =
        ClimaCore.Geometry._dual_axes_for_projection(mat1_et)
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
projection (see `Geometry._dual_axes_for_projection`).
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
@inline project_or_map(axis, lg, y::AbstractTensor) = project(axis, y, lg)

# Zip each component's axis with its component so the componentwise map can reuse
# `Base.Fix2` instead of a closure over `lg`. This must use `unrolled_map_into_tuple`
# rather than `unrolled_map`: the latter derives its output type from a nested
# `Base.promote_op` query, which is not precise inside the `Utilities.return_type`
# call in `cached_operand_type` and widens the projected type to a non-concrete
# `BandMatrixRow`, whereas `unrolled_map_into_tuple` maps directly into a `Tuple`.
Base.@propagate_inbounds paired_projection(axes_per_component, ys, lg) =
    UnrolledUtilities.unrolled_map_into_tuple(
        Base.Fix2(paired_projection_component, lg),
        zip(axes_per_component, ys),
    )
Base.@propagate_inbounds paired_projection_component((axis, y), lg) =
    project_or_map(axis, lg, y)

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
    for m in methods(paired_projection_component)
        m.recursion_relation = dont_limit
    end
    for m in methods(calc_level_val)
        m.recursion_relation = dont_limit
    end
end
