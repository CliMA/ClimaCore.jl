import UnrolledUtilities:
    unrolled_map, unrolled_foreach, unrolled_reduce, unrolled_any
import ..Utilities.Unrolled: unrolled_map_with_inbounds

abstract type AbstractSpectralStyle <: Fields.AbstractFieldStyle end

"""
    SpectralStyle()

Broadcasting requires use of spectral-element operations.
"""
struct SpectralStyle <: AbstractSpectralStyle end

"""
    SlabBlockSpectralStyle()

Applies spectral-element operations using by making use of intermediate
temporaries for each operator. This is used for CPU kernels.
"""
struct SlabBlockSpectralStyle <: AbstractSpectralStyle end


import ClimaComms
AbstractSpectralStyle(::ClimaComms.AbstractCPUDevice) = SlabBlockSpectralStyle


"""
    SpectralElementOperator{I}

Represents an operation that is applied to each element, where `I` is the tuple of axis indices.

Subtypes `Op` of this should define the following:
- [`operator_return_eltype(::Op, ElTypes...)`](@ref)
- [`allocate_work(::Op, args...)`](@ref)
- [`apply_operator(::Op, work, args...)`](@ref)

Additionally, the result type `OpResult <: OperatorSlabResult` of `apply_operator` should define `get_node(::OpResult, ij, slabidx)`.
"""
abstract type SpectralElementOperator{I} <: AbstractOperator end

"""
    operator_axes(space)

Return a tuple of the axis indices a given field operator works over.
"""
function operator_axes end

operator_axes(space::Spaces.AbstractSpace) = ()
operator_axes(space::Spaces.SpectralElementSpace1D) = (1,)
operator_axes(space::Spaces.SpectralElementSpace2D) = (1, 2)
operator_axes(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    operator_axes(Spaces.horizontal_space(space))


"""
    node_indices(space)

The indices of the nodes in one slab of `space`: `Nq` of them along each axis that
an operator on `space` works over (see [`operator_axes`](@ref)).
"""
function node_indices(
    space::Union{
        Spaces.AbstractSpectralElementSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    return CartesianIndices(ntuple(_ -> Nq, Val(length(operator_axes(space)))))
end

# `operator_axes` is empty here, but a slab still holds the one node of a column
node_indices(space::Spaces.FiniteDifferenceSpace) = CartesianIndices((1,))


"""
    SpectralBroadcasted{Style}(op, args[,axes[, work]])

This is similar to a `Base.Broadcast.Broadcasted` object, except it contains space for an intermediate `work` storage.

This is returned by `Base.Broadcast.broadcasted(op::SpectralElementOperator)`.
"""
struct SpectralBroadcasted{Style, Op, Args, Axes, Work} <:
       OperatorBroadcasted{Style}
    op::Op
    args::Args
    axes::Axes
    work::Work
end
SpectralBroadcasted{Style}(
    op::Op,
    args::Args,
    axes::Axes = nothing,
    work::Work = nothing,
) where {Style, Op, Args, Axes, Work} =
    SpectralBroadcasted{Style, Op, Args, Axes, Work}(op, args, axes, work)

Adapt.adapt_structure(to, sbc::SpectralBroadcasted{Style}) where {Style} =
    SpectralBroadcasted{Style}(
        sbc.op,
        Adapt.adapt(to, sbc.args),
        Adapt.adapt(to, sbc.axes),
        Adapt.adapt(to, sbc.work),
    )

return_space(::SpectralElementOperator, space, args...) = space

"""
    FormType

Supertype of the singleton types [`StrongForm`](@ref) and [`WeakForm`](@ref),
which distinguish the variational form of a spectral element operator.

The strong and weak variants of an operator share the same interior
computation; they differ only in three form-dependent factors:
 - whether the derivative matrix is applied directly or transposed with a sign
   flip (from integration by parts); see `form_deriv_entry`,
 - whether the argument is weighted by the quadrature weights `W` or by the
   Jacobian factor; see `form_weighted_arg` and `form_jacobian`,
 - whether the result is rescaled by `J` or by `WJ`; see
   `form_jacobian_rescale`, and by `W` or not at all; see
   `form_weight_rescale`.

Operators with strong/weak variants carry a `FormType` as their second type
parameter, e.g. `Divergence{I, StrongForm}`, with the weak variant available
under an alias, e.g. `WeakDivergence{I} = Divergence{I, WeakForm}`.
"""
abstract type FormType end

"""
    StrongForm()

The [`FormType`](@ref) of an operator that discretizes a derivative directly at
the quadrature points (e.g. [`Divergence`](@ref), [`Gradient`](@ref),
[`Curl`](@ref)).
"""
struct StrongForm <: FormType end

"""
    WeakForm()

The [`FormType`](@ref) of an operator that discretizes the volume-integral
contribution of the corresponding weak-form expression, obtained after
integration by parts (e.g. [`WeakDivergence`](@ref), [`WeakGradient`](@ref),
[`WeakCurl`](@ref)).
"""
struct WeakForm <: FormType end

"""
    form_deriv_entry(form, D, ii, i)

Entry of the derivative matrix `D` applied by an operator of the given
[`FormType`](@ref) when accumulating the contribution of quadrature point `i`
to quadrature point `ii`: `D[ii, i]` for the strong form, and `-D[i, ii]` (the
transpose, with the sign flip from integration by parts) for the weak form.
"""
@inline form_deriv_entry(::StrongForm, D, ii, i) = D[ii, i]
@inline form_deriv_entry(::WeakForm, D, ii, i) = -D[i, ii]

"""
    form_weighted_arg(form, local_geometry, x)

The argument value `x`, weighted as required by an operator of the given
[`FormType`](@ref) whose weak variant integrates against test functions: `x`
itself for the strong form, and `W * x` (with the quadrature weights
`W = WJ * J⁻¹`) for the weak form. Used by [`Gradient`](@ref) and
[`Curl`](@ref).
"""
@inline form_weighted_arg(::StrongForm, local_geometry, x) = x
@inline form_weighted_arg(::WeakForm, local_geometry, x) =
    (local_geometry.WJ * local_geometry.invJ) * x

"""
    form_jacobian(form, local_geometry)

The Jacobian factor that scales the contravariant components summed by an
operator of the given [`FormType`](@ref): `J` for the strong form, and `WJ`
for the weak form. Used by [`Divergence`](@ref).
"""
@inline form_jacobian(::StrongForm, local_geometry) = local_geometry.J
@inline form_jacobian(::WeakForm, local_geometry) = local_geometry.WJ

"""
    form_jacobian_rescale(form, local_geometry, x)

The result value `x`, divided by the `form_jacobian` of the given
[`FormType`](@ref): `x * J⁻¹` for the strong form (using the precomputed
inverse), and `x / WJ` for the weak form. Used by [`Divergence`](@ref) and
[`Curl`](@ref).
"""
@inline form_jacobian_rescale(::StrongForm, local_geometry, x) =
    x * local_geometry.invJ
@inline form_jacobian_rescale(::WeakForm, local_geometry, x) =
    x / local_geometry.WJ

"""
    form_weight_rescale(form, local_geometry, x)

The result value `x`, divided by the quadrature weights `W = WJ * J⁻¹` if the
given [`FormType`](@ref) requires it: `x` itself for the strong form, and
`x / W` for the weak form. Used by [`Gradient`](@ref), whose weak variant
weights its argument by `W` without a Jacobian factor to divide out.

The CPU `apply_operator` methods for [`Gradient`](@ref) inline this rescale
behind an `F === WeakForm` branch instead of calling it, so that the strong form
skips the loop over quadrature points entirely; on GPUs each thread rescales
only its own point, so there is no loop to skip.
"""
@inline form_weight_rescale(::StrongForm, local_geometry, x) = x
@inline form_weight_rescale(::WeakForm, local_geometry, x) =
    x / (local_geometry.WJ * local_geometry.invJ)

"""
    rebuild_operator(op, space)

Reconstruct a `SpectralElementOperator` with its `operator_axes` reset to those
of `space`, preserving all other type parameters (in particular the
[`FormType`](@ref) of operators with strong/weak variants).
"""
rebuild_operator(op::SpectralElementOperator, space) =
    unionall_type(typeof(op)){()}(space)

function Base.Broadcast.broadcasted(op::SpectralElementOperator, args...)
    args′ = map(Base.Broadcast.broadcastable, args)
    style = Base.Broadcast.result_style(
        SpectralStyle(),
        Base.Broadcast.combine_styles(args′...),
    )
    Base.Broadcast.broadcasted(style, op, args′...)
end

function Base.Broadcast.broadcasted(
    ::SpectralStyle,
    op::SpectralElementOperator,
    args...,
)
    args′ =
        unrolled_map(args) do arg
            is_auto_broadcastable(eltype(arg)) ?
            Base.Broadcast.broadcasted(add_auto_broadcasters, arg) : arg
        end
    return SpectralBroadcasted{SpectralStyle}(op, args′)
end

Base.eltype(sbc::SpectralBroadcasted) =
    operator_return_eltype(sbc.op, map(eltype, sbc.args)...)

function Base.Broadcast.instantiate(sbc::SpectralBroadcasted)
    op = sbc.op
    # recursively instantiate the arguments to allocate intermediate work arrays
    args = instantiate_args(sbc.args)
    # axes: same logic as Broadcasted
    if sbc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Base.axes(sbc)
    else
        axes = sbc.axes
        if axes !== Base.axes(sbc)
            Base.Broadcast.check_broadcast_axes(axes, args...)
        end
    end
    # For FiniteDifferenceSpace, return zeros 
    if axes isa Spaces.FiniteDifferenceSpace
        RT = operator_return_eltype(op, map(eltype, args)...)
        return Broadcast.broadcasted(Returns(zero(RT)), Fields.coordinate_field(axes))
    end
    # If we've already instantiated, then we need to reset the operator axes.
    op = rebuild_operator(op, axes)
    Style = AbstractSpectralStyle(ClimaComms.device(axes))
    return SpectralBroadcasted{Style}(op, args, axes)
end

function Base.Broadcast.instantiate(
    bc::Base.Broadcast.Broadcasted{<:AbstractSpectralStyle},
)
    # recursively instantiate the arguments to allocate intermediate work arrays
    args = instantiate_args(bc.args)
    # axes: same logic as Broadcasted
    if bc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Base.Broadcast.combine_axes(args...)
    else
        axes = bc.axes
        Base.Broadcast.check_broadcast_axes(axes, args...)
    end
    # For FiniteDifferenceSpace with operators, return zeros for horizontal operators
    if axes isa Spaces.FiniteDifferenceSpace && bc.f isa SpectralElementOperator
        op = rebuild_operator(bc.f, axes)
        RT = operator_return_eltype(op, map(eltype, args)...)
        return Broadcast.broadcasted(Returns(zero(RT)), Fields.coordinate_field(axes))
    end

    if bc.f isa SpectralElementOperator
        op = rebuild_operator(bc.f, axes)
        Style = AbstractSpectralStyle(ClimaComms.device(axes))
        return Base.Broadcast.Broadcasted{Style}(op, args, axes)
    else
        # For non-operators, use the default broadcast style to avoid needing
        # operator_return_eltype for regular functions
        return Base.Broadcast.Broadcasted(bc.f, args, axes)
    end
end

# Functions for SlabBlockSpectralStyle
function Base.copyto!(
    out::Field,
    sbc::Union{
        SpectralBroadcasted{SlabBlockSpectralStyle},
        Broadcasted{SlabBlockSpectralStyle},
    };
    mask = DataLayouts.NoMask(),
)
    Fields.byslab(axes(out)) do slabidx
        Base.@_inline_meta
        @inbounds copyto_slab!(out, sbc, slabidx)
    end
    call_post_op_callback() && post_op_callback(out, out, sbc)
    return out
end


"""
    copyto_slab!(out, bc, slabidx)

Copy the slab indexed by `slabidx` from `bc` to `out`.
"""
Base.@propagate_inbounds function copyto_slab!(out, bc, slabidx)
    space = axes(out)
    rbc = resolve_operator(bc, slabidx)
    @inbounds for ij in node_indices(axes(out))
        set_node!(space, out, ij, slabidx, get_node(space, rbc, ij, slabidx))
    end
    return nothing
end

"""
    resolve_operator(bc, slabidx)

Recursively evaluate any operators in `bc` at `slabidx`, replacing any
`SpectralBroadcasted` objects.

- if `bc` is a regular `Broadcasted` object, return a new `Broadcasted` with `resolve_operator` called on each `arg`
- if `bc` is a regular `SpectralBroadcasted` object:
 - call `resolve_operator` called on each `arg`
 - call `apply_operator`, returning the resulting "pseudo Field":  a `Field` with a
 [`SlabData`](@ref) data object.
- if `bc` is a `Field`, return that
"""
Base.@propagate_inbounds function resolve_operator(
    bc::SpectralBroadcasted{SlabBlockSpectralStyle},
    slabidx,
)
    args = _resolve_operator(slabidx, bc.args)
    apply_operator(bc.op, bc.axes, slabidx, args...)
end
Base.@propagate_inbounds function resolve_operator(
    bc::Base.Broadcast.Broadcasted{SlabBlockSpectralStyle},
    slabidx,
)
    args = _resolve_operator(slabidx, bc.args)
    Base.Broadcast.Broadcasted{SlabBlockSpectralStyle}(bc.f, args, bc.axes)
end
@inline resolve_operator(x, slabidx) = x

Base.@propagate_inbounds _resolve_operator(slabidx, args) =
    unrolled_map_with_inbounds(args) do arg
        Base.@_propagate_inbounds_meta
        resolve_operator(arg, slabidx)
    end

function strip_space(bc::SpectralBroadcasted{Style}, parent_space) where {Style}
    current_space = axes(bc)
    new_space = placeholder_space(current_space, parent_space)
    return SpectralBroadcasted{Style}(
        bc.op,
        strip_space_args(bc.args, current_space),
        new_space,
    )
end

"""
    reconstruct_placeholder_broadcasted(space, obj)

Recurively reconstructs objects that have been stripped via `strip_space`.
"""
@inline reconstruct_placeholder_broadcasted(parent_space, obj) = obj
@inline function reconstruct_placeholder_broadcasted(
    parent_space::Spaces.AbstractSpace,
    field::Fields.Field,
)
    space = reconstruct_placeholder_space(axes(field), parent_space)
    return Fields.Field(Fields.field_values(field), space)
end
@inline function reconstruct_placeholder_broadcasted(
    parent_space::Spaces.AbstractSpace,
    bc::Broadcasted{Style},
) where {Style}
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    args = _reconstruct_placeholder_broadcasted(space, bc.args)
    return Broadcasted{Style}(bc.f, args, space)
end

@inline function reconstruct_placeholder_broadcasted(
    parent_space::Spaces.AbstractSpace,
    sbc::SpectralBroadcasted{Style},
) where {Style}
    space = reconstruct_placeholder_space(axes(sbc), parent_space)
    args = _reconstruct_placeholder_broadcasted(space, sbc.args)
    return SpectralBroadcasted{Style}(sbc.op, args, space, sbc.work)
end

@inline _reconstruct_placeholder_broadcasted(parent_space, args::Tuple) =
    unrolled_map(arg -> reconstruct_placeholder_broadcasted(parent_space, arg), args)

"""
    is_valid_index(space, ij, slabidx)::Bool

Returns `true` if the node indices `ij` and slab indices `slabidx` are valid for
`space`.
"""
@inline function is_valid_index(space, ij, slabidx)
    # if we want to support interpolate/restrict, we would need to check i <= Nq && j <= Nq
    is_valid_index(space, slabidx)
end
# assumes h is always in a valid range
@inline function is_valid_index(
    space::Spaces.AbstractSpectralElementSpace,
    slabidx,
)
    return true
end
@inline function is_valid_index(
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    slabidx,
)
    Nv = Spaces.nlevels(space)
    return slabidx.v <= Nv
end
@inline function is_valid_index(
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    slabidx,
)
    Nv = Spaces.nlevels(space)
    return slabidx.v + half <= Nv
end

Base.@propagate_inbounds _get_node(space, ij, slabidx, args) =
    unrolled_map_with_inbounds(args) do arg
        Base.@_propagate_inbounds_meta
        get_node(space, arg, ij, slabidx)
    end

Base.@propagate_inbounds function get_node(space, scalar, ij, slabidx)
    scalar[]
end
Base.@propagate_inbounds function get_node(
    space,
    scalar::Tuple{<:Any},
    ij,
    slabidx,
)
    scalar[1]
end
"""
    data_index(ij, v, h)

Index of node `ij` at level `v` of element `h` in a four-index data layout. A node
index has one component per axis that an operator works over (see
[`operator_axes`](@ref)), so the axes it leaves out are indexed at 1 -- the only
node those layouts store along them.
"""
@inline data_index(ij::CartesianIndex{N}, v, h) where {N} =
    CartesianIndex(v, ntuple(d -> d <= N ? ij[d] : 1, Val(2))..., h)

"""
    slab_level_index(space, slabidx)

The level at which the slab `slabidx` reads `space`: `slabidx.v` for center
spaces, staggered by [`half`](@ref) for face spaces, and level 1 for the
horizontal spaces whose slab index carries no level at all (`slabidx.v ===
nothing`).

The staggering of `space` has to be known, so an unrecognised space is an error
rather than a silent read of level `slabidx.v`. Both the one- and the
two-dimensional node accessors used to inline this logic, but only the
one-dimensional ones handled a bare `FiniteDifferenceSpace`; that is deliberate
here, since a horizontal operator applied to an extracted column space reaches
these accessors with a one-dimensional node index over such a space.
"""
@inline function slab_level_index(space, slabidx)
    if space isa Spaces.FaceExtrudedFiniteDifferenceSpace ||
       space isa Spaces.FaceFiniteDifferenceSpace
        _v = slabidx.v + half
    elseif space isa Spaces.CenterExtrudedFiniteDifferenceSpace ||
           space isa Spaces.CenterFiniteDifferenceSpace ||
           space isa Spaces.AbstractSpectralElementSpace
        _v = slabidx.v
    else
        error("invalid space")
    end
    return isnothing(_v) ? 1 : _v
end

Base.@propagate_inbounds function get_node(
    parent_space,
    field::Fields.Field,
    ij::CartesianIndex,
    slabidx,
)
    space = reconstruct_placeholder_space(axes(field), parent_space)
    fv = Fields.field_values(field)
    return fv[data_index(ij, slab_level_index(space, slabidx), slabidx.h)]
end



Base.@propagate_inbounds function get_node(
    parent_space,
    bc::Base.Broadcast.Broadcasted,
    ij,
    slabidx,
)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    args = _get_node(space, ij, slabidx, bc.args)
    bc.f(args...)
end
"""
    SlabData{T}

A [`DataLayouts.DataLayout`](@ref) that stores a single slab of values of type
`T` (a `VIJHWithF` layout with `Nv = Nh = 1`).
"""
const SlabData{T} = DataLayouts.VIJHWithF{T, 1, <:Any, <:Any, 1}

"""
    slab_data(T, FT, Ni, [Nj])

Mutable temporary storage for one slab of values of type `T`, backed by an
`MArray` with eltype `FT` (a `VIJFH` layout with `Nv = Nh = 1`).
"""
@inline function slab_data(::Type{T}, ::Type{FT}, Ni, Nj = 1) where {T, FT}
    Nf = DataLayouts.num_basetypes(FT, T)
    array = MArray{Tuple{1, Ni, Nj, Nf, 1}, FT, 5, Ni * Nj * Nf}(undef)
    return DataLayouts.VIJFH{T, 1, Ni, Nj, 1}(array)
end

# Immutable copy of a slab temporary, used to construct pseudo-Fields.
@inline immutable_slab_data(data::SlabData) =
    DataLayouts.rebuild(data, SArray(parent(data)))

# Index for one node in a slab of data, with v = h = 1.
@inline slab_node_index(ij::CartesianIndex) = data_index(ij, 1, 1)

"""
    slab_node_index(data, ij)

Index for node `ij` in a single slab of `data`. The four-index `(1, i, j, 1)`
form above serves the [`SlabData`](@ref) temporaries, the `MArray` temporaries of
[`tensor_product!`](@ref) and every `VIJHWithF` layout; the layouts that drop a
dimension are indexed with one index per dimension they keep.
"""
@inline slab_node_index(_, ij) = slab_node_index(ij)
@inline slab_node_index(::DataLayouts.VIH1, ij::CartesianIndex{1}) =
    CartesianIndex(1, ij[1])
@inline slab_node_index(::DataLayouts.IH1JH2, ij::CartesianIndex{2}) = ij

Base.@propagate_inbounds function get_node(space, data::SlabData, ij, slabidx)
    data[slab_node_index(ij)]
end
Base.@propagate_inbounds function get_node(
    space,
    field::Fields.Field{<:SlabData},
    ij::CartesianIndex,
    slabidx,
)
    Fields.field_values(field)[slab_node_index(ij)]
end
Base.@propagate_inbounds function get_node(
    space,
    data::StaticArrays.SArray,
    ij,
    slabidx,
)
    data[ij]
end

dont_limit = (args...) -> true
for m in methods(get_node)
    m.recursion_relation = dont_limit
end

Base.@propagate_inbounds function get_local_geometry(
    space::Union{
        Spaces.AbstractSpectralElementSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    ij::CartesianIndex,
    slabidx,
)
    lgd = Spaces.local_geometry_data(space)
    return lgd[data_index(ij, slab_level_index(space, slabidx), slabidx.h)]
end

Base.@propagate_inbounds function set_node!(
    space,
    field::Fields.Field,
    ij::CartesianIndex,
    slabidx,
    val,
)
    fv = Fields.field_values(field)
    fv[data_index(ij, slab_level_index(space, slabidx), slabidx.h)] = val
end

Base.Broadcast.BroadcastStyle(
    ::Type{<:SpectralBroadcasted{Style}},
) where {Style} = Style()

Base.Broadcast.BroadcastStyle(
    style::AbstractSpectralStyle,
    ::Fields.AbstractFieldStyle,
) = style


##### Dimension-generic building blocks
#
# The operators below are tensor-product operators: each applies the same
# one-dimensional stencil along every axis it works over, accumulating the
# per-axis contributions. Together with the `FormType` helpers above -- which
# absorb the differences between an operator's strong and weak variants -- these
# let one implementation cover any axis tuple `I` (`(1,)` on a
# `SpectralElementSpace1D`, `(1, 2)` on a `SpectralElementSpace2D`), keeping each
# axis index in the type domain so that the loop over axes unrolls and the node
# indices stay statically sized.
#
# Loops over axes go through `foreach_axis` and `sum_axes`, which own the
# inlining discipline the per-axis closures depend on. Their bodies still need
# their own `@inbounds`: a closure body does not inherit it from the enclosing
# `@inbounds` block, and only a method (not an anonymous function) can opt into
# propagating it.

"""
    axis_vals(op)

Tuple of `Val{d}` for each axis index `d` that `op` works over, e.g.
`(Val(1), Val(2))`. Iterating over this with `unrolled_foreach`/`unrolled_map`
unrolls the loop over axes and keeps `d` available as a type parameter.
"""
@inline axis_vals(::SpectralElementOperator{I}) where {I} = unrolled_map(Val, I)

"""
    axis_index(::Val{d})

The axis index `d` itself: `ij[axis_index(vd)]` is the node index along axis `d`.
"""
@inline axis_index(::Val{d}) where {d} = d

"""
    foreach_axis(f, op)
    sum_axes(f, op)

Call `f(Val(d))` for each axis index `d` that `op` works over, unrolled;
`sum_axes` sums the results, keeping one accumulator per axis and combining them
once at the end so that the per-axis accumulation loops stay independent
dependency chains.

Two annotations belong in the body of `f`, and neither can be supplied from here,
because only a method -- not an anonymous function -- can carry them:

  - `@inline`, as the first statement. Without it the closure is not inlined, so
    the mutable slab temporary it writes to escapes and is heap-allocated once per
    slab; that measured 1.1-3.8x slower across the operator kernels. (Forcing the
    inline from inside these helpers does not work: wrapping `f` in another
    closure to annotate its call site reintroduces the allocation.)
  - `@inbounds`, since a closure body does not inherit it from an enclosing
    `@inbounds` block.
"""
@inline foreach_axis(f::F, op::SpectralElementOperator) where {F} =
    unrolled_foreach(f, axis_vals(op))

@inline sum_axes(f::F, op::SpectralElementOperator) where {F} =
    unrolled_reduce(+, unrolled_map(f, axis_vals(op)))

"""
    slab_dims(op, Nq)
    slab_dims(axis_vals, Nq)

The shape of one slab of nodes for `op`: `Nq` repeated once per axis it works
over. The second form takes the axes as a tuple of `Val`s, for the slabs that
belong to no operator (see [`tensor_product!`](@ref)).
"""
@inline slab_dims(op::SpectralElementOperator, Nq) = slab_dims(axis_vals(op), Nq)
@inline slab_dims(::NTuple{N, Val}, Nq) where {N} = ntuple(_ -> Nq, Val(N))

"""
    replace_index(index, ::Val{d}, k)

The Cartesian `index` -- or the tuple of extents `index` -- with its `d`th
component replaced by `k`. Used to walk along one axis of a slab while holding
the other axes fixed, and to shrink the extent of an axis that has been
contracted.
"""
@inline replace_index(index::CartesianIndex, vd::Val, k) =
    CartesianIndex(replace_index(Tuple(index), vd, k))
@inline replace_index(index::NTuple{N, Any}, ::Val{d}, k) where {N, d} =
    ntuple(n -> n == d ? k : index[n], Val(N))

"""
    contravariant(::Val{d}, u, local_geometry)
    covariant(::Val{d}, u, local_geometry)

The `d`th contravariant (``u^d``) or covariant (``u_d``) component of `u`.
"""
@inline contravariant(::Val{1}, u, lg) = Geometry.contravariant1(u, lg)
@inline contravariant(::Val{2}, u, lg) = Geometry.contravariant2(u, lg)
@inline contravariant(::Val{3}, u, lg) = Geometry.contravariant3(u, lg)
@inline covariant(::Val{1}, u, lg) = Geometry.covariant1(u, lg)
@inline covariant(::Val{2}, u, lg) = Geometry.covariant2(u, lg)
@inline covariant(::Val{3}, u, lg) = Geometry.covariant3(u, lg)

# The differential operators above accumulate an independent contribution per
# axis, so one pass over a slab covers every axis. The tensor-product operators
# below -- interpolation, restriction and `tensor_product!` -- instead contract
# their axes *sequentially*, applying the same matrix along one axis at a time,
# so they fold over the axes rather than looping over them: each contraction
# reads what the previous one wrote.

"""
    SlabReader(data)

Reads node `ij` of one slab of `data`, in the form [`contract_axis!`](@ref)
expects of its source. A callable struct rather than a closure so that the read
is a method, and can therefore propagate `@inbounds` from its caller.
"""
struct SlabReader{D}
    data::D
end
Base.@propagate_inbounds (reader::SlabReader)(ij) =
    reader.data[slab_node_index(reader.data, ij)]

"""
    contract_axis!(dst, M, src, post, ::Val{d}, dims_out)

Contract axis `d` with the matrix `M`, writing

    dst[ij] = post(ij, ∑ₖ M[ij[d], k] * src(replace_index(ij, Val(d), k)))

for every node `ij` in `CartesianIndices(dims_out)`, where the sum runs over the
`size(M, 2)` nodes along axis `d` of the source. `src` reads one node of the
source by Cartesian index (see [`SlabReader`](@ref) and [`TensorArg`](@ref)), so
the source may be a slab of data or the argument of an operator, read through
`get_node` without materialising a slab of it first. `post` rescales the
contracted value; it is applied here rather than in a second pass over `dst` so
that the arithmetic keeps the shape it had in the per-dimension methods this
replaces. Whether a division sits inside or outside the accumulation loop can
change which of the `muladd`s LLVM contracts into an `fma`, so moving it is not
free of floating-point consequences even though it computes the same expression.
"""
Base.@propagate_inbounds function contract_axis!(
    dst,
    M,
    src::S,
    post::P,
    vd,
    dims_out,
) where {S, P}
    Nq_in = size(M, 2)
    for ij in CartesianIndices(dims_out)
        i = ij[axis_index(vd)]
        r = M[i, 1] * src(replace_index(ij, vd, 1))
        for k in 2:Nq_in
            r = muladd(M[i, k], src(replace_index(ij, vd, k)), r)
        end
        dst[slab_node_index(dst, ij)] = post(ij, r)
    end
    return dst
end

# the `post` of a contraction whose result needs no rescaling
@inline no_rescale(ij, r) = r

"""
    contract_axes!(dsts, axis_vals, M, src, post, dims_in)

Contract every axis in `axis_vals` with the matrix `M`, one axis at a time,
reading the first contraction's source from `src` and each later one from what the
previous contraction wrote. `dims_in` is the shape of the source slab; each
contraction shrinks (or grows) the extent of its own axis from `size(M, 2)` to
`size(M, 1)`.

`dsts` holds one destination per axis: the last contraction writes into
`last(dsts)`, and the intermediate results go into the temporaries before it (see
[`contract_temps`](@ref)), so that a single-axis contraction writes straight into
the output with no temporary at all. `post` rescales the final result only, so
only the last contraction applies it.
"""
Base.@propagate_inbounds contract_axes!(
    dsts::Tuple{Any},
    vds::Tuple{Val},
    M,
    src,
    post,
    dims_in,
) = contract_axis!(
    dsts[1],
    M,
    src,
    post,
    vds[1],
    replace_index(dims_in, vds[1], size(M, 1)),
)
Base.@propagate_inbounds function contract_axes!(
    dsts::Tuple,
    vds::Tuple{Val, Vararg{Val}},
    M,
    src,
    post,
    dims_in,
)
    dims_out = replace_index(dims_in, vds[1], size(M, 1))
    dst = contract_axis!(dsts[1], M, src, no_rescale, vds[1], dims_out)
    return contract_axes!(
        Base.tail(dsts),
        Base.tail(vds),
        M,
        SlabReader(dst),
        post,
        dims_out,
    )
end

"""
    contract_temps(f, axis_vals)

The temporaries that [`contract_axes!`](@ref) writes its intermediate results
into: `f()`, once per axis except the last, since the last contraction writes
straight into the output. Each temporary must be large enough for any
intermediate extent, because an axis holds `size(M, 2)` nodes until it is
contracted and `size(M, 1)` afterwards.
"""
@inline contract_temps(f::F, ::NTuple{N, Val}) where {F, N} =
    ntuple(_ -> f(), Val(N - 1))



"""
    div = Divergence()
    div.(u)

Computes the per-element spectral (strong) divergence of a vector field ``u``.

The divergence of a vector field ``u`` is defined as
```math
\\nabla \\cdot u = \\sum_i \\frac{1}{J} \\frac{\\partial (J u^i)}{\\partial \\xi^i}
```
where ``J`` is the Jacobian determinant, ``u^i`` is the ``i``th contravariant
component of ``u``.

This is discretized by
```math
\\sum_i I \\left\\{\\frac{1}{J} \\frac{\\partial (I\\{J u^i\\})}{\\partial \\xi^i} \\right\\}
```
where ``I\\{x\\}`` is the interpolation operator that projects to the
unique polynomial interpolating ``x`` at the quadrature points. In matrix
form, this can be written as
```math
J^{-1} \\sum_i D_i J u^i
```
where ``D_i`` is the derivative matrix along the ``i``th dimension

## References
- [Taylor2010](@cite), equation 15
"""
struct Divergence{I, F <: FormType} <: SpectralElementOperator{I} end
Divergence() = Divergence{(), StrongForm}()
Divergence{I}() where {I} = Divergence{I, StrongForm}()
rebuild_operator(::Divergence{I, F}, space) where {I, F} =
    Divergence{operator_axes(space), F}()

operator_return_eltype(op::Divergence{I}, ::Type{S}) where {I, S} =
    Geometry.divergence_result_type(S)

# The strong divergence is J⁻¹ ∑ᵢ Dᵢ (J uⁱ), while the weak divergence is
# -(WJ)⁻¹ ∑ᵢ Dᵢᵀ (WJ uⁱ); see form_deriv_entry, form_jacobian, and
# form_jacobian_rescale for the form-dependent factors.

Base.@propagate_inbounds function apply_operator(
    op::Divergence{I, F},
    space,
    slabidx,
    arg,
) where {I, F}
    form = F()
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = slab_data(RT, FT, slab_dims(op, Nq)...)
    fill!(parent(out), zero(FT))
    @inbounds for ij in node_indices(space)
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(space, arg, ij, slabidx)
        foreach_axis(op) do vd
            @inline
            @inbounds begin
                i = ij[axis_index(vd)]
                Jvᵈ =
                    form_jacobian(form, local_geometry) *
                    contravariant(vd, v, local_geometry)
                for k in 1:Nq
                    out[slab_node_index(replace_index(ij, vd, k))] +=
                        form_deriv_entry(form, D, k, i) * Jvᵈ
                end
            end
        end
    end
    @inbounds for ij in node_indices(space)
        local_geometry = get_local_geometry(space, ij, slabidx)
        node = slab_node_index(ij)
        out[node] = form_jacobian_rescale(form, local_geometry, out[node])
    end
    return Field(immutable_slab_data(out), space)
end

"""
    split_div = SplitDivergence()
    split_div.(ρu, ψ)

Computes the divergence of the product `ρu * ψ` using a **split-form (entropy-stable)** discretization.

This operator is designed for the advection of scalar quantities in conservation laws (e.g., 
thermodynamic variables or tracers). By evaluating the divergence using a specific averaging of the 
conservative and advective forms, this formulation cancels aliasing errors that arise from the product 
of two spectrally variable fields, thereby inhibiting the growth of quadratic instabilities (such as 
cold temperature spikes) without requiring hyperviscosity.

# Arguments
- `ρu`: The transport vector field, typically the **mass flux**. It must be a vector quantity (e.g., `Geometry.Contravariant12Vector`).
- `ψ`: The **specific** scalar quantity to be advected (e.g., specific total energy ``e_{tot}`` or specific humidity ``q_{tot}``).

# Mathematical Formulation

## Continuous
The split form of the divergence operator is defined as the arithmetic mean of
the conservative and advective forms:
```math
\\nabla \\cdot (\\rho \\mathbf{u} \\psi)|_\\textrm{split} =
    \\frac{1}{2} \\nabla \\cdot (\\rho \\mathbf{u} \\psi) +
    \\frac{1}{2} \\left(
        \\psi \\nabla \\cdot (\\rho \\mathbf{u}) +
        \\rho \\mathbf{u} \\cdot \\nabla \\psi
    \\right)
```

## Discrete
The discretized split operator is equivalent to using the strong formulation of
the gradient operator and the weak formulation of the divergence operator:
```math
\\textrm{split_div}(\\rho \\mathbf{u}, \\psi) =
    \\frac{1}{2} \\textrm{wdiv}(\\rho \\mathbf{u} \\psi) +
    \\frac{1}{2} \\left(
        \\psi \\textrm{wdiv}(\\rho \\mathbf{u}) +
        \\rho \\mathbf{u} \\cdot \\textrm{grad}(\\psi)
    \\right)
```
Swapping the weak and strong formulations in the last two terms also results in
the same operator. The discrete form of the divergence theorem, which stems from
the generalized summation-by-parts (SBP) property, guarantees that the integral
of the first term vanishes,
```math
\\int_\\Omega \\textrm{wdiv}(\\rho \\mathbf{u} \\psi) dV = 0
```
while the integrals of the other two terms cancel out,
```math
\\int_\\Omega \\psi \\textrm{wdiv}(\\rho \\mathbf{u}) dV =
    -\\int_\\Omega \\rho \\mathbf{u} \\cdot \\textrm{grad}(\\psi) dV
```
So, this discretization ensures that the split operator conserves the integral
of ``\\rho \\mathbf{u} \\psi``.

## Two-Point
A more compact representation of the discretized operator can be obtained with
the symmetric two-point flux, whose values in one dimension are
```math
(F^1)_{ij} =
    \\frac{1}{2} (\\rho_i J_i (u^1)_i + \\rho_j J_j (u^1)_j) (\\psi_i + \\psi_j)
```
With ``D`` denoting the spectral derivative matrix, the split operator in one
dimension can be expressed as
```math
\\textrm{split_div}(\\rho \\mathbf{u}, \\psi)_i =
    \\frac{1}{J_i} \\sum_{j \\neq i} D_{ij} (F^1)_{ij}
```
In two dimensions, ``F^1`` and the analogous quantity ``F^2`` provide a similar
expression for the split divergence, with the one-dimensional operator applied
sequentially along each dimension.

# Properties
1.  **Conservation:** The split operator conserves ``\\rho \\mathbf{u} \\psi``
2.  **Consistency:** If ``\\psi = 1``, the split operator degenerates to the
    weak formulation of ``\\nabla \\cdot \\rho \\mathbf{u}`` (mass continuity)
3.  **Complexity:** The split operator has the same ``O(N^2)`` complexity per
    element as the strong and weak operators, but needs twice as many operations

# References
- Fisher, T. C., & Carpenter, M. H. (2013). High-order entropy stable finite difference schemes for nonlinear conservation laws: Finite domains. Journal of Computational Physics, 252, 518-557. [https://doi.org/10.1016/j.jcp.2013.06.014](https://doi.org/10.1016/j.jcp.2013.06.014)
- Gassner, G. J. (2013). A skew-symmetric discontinuous Galerkin spectral element discretization and its relation to SBP-SAT finite difference methods. SIAM Journal on Scientific Computing, 35, A1233-A1253. [https://doi.org/10.1137/120890144](https://doi.org/10.1137/120890144)
"""
struct SplitDivergence{I} <: SpectralElementOperator{I} end
SplitDivergence() = SplitDivergence{()}()
SplitDivergence{()}(space) = SplitDivergence{operator_axes(space)}()

operator_return_eltype(
    ::SplitDivergence{I},
    ::Type{S1},
    ::Type{S2},
) where {I, S1, S2} =
    Geometry.mul_return_type(Geometry.divergence_result_type(S1), S2)

Base.@propagate_inbounds function apply_operator(
    op::SplitDivergence{I},
    space,
    slabidx,
    arg1,
    arg2,
) where {I}
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    JT = operator_return_eltype(op, eltype(arg1), FT)
    RT = operator_return_eltype(op, eltype(arg1), eltype(arg2))
    dims = slab_dims(op, Nq)

    # `Ju[d]` is the mass flux along axis `d`; `psi` is the advected scalar
    Ju = unrolled_map(_ -> slab_data(JT, FT, dims...), axis_vals(op))
    psi = slab_data(eltype(arg2), eltype(arg2), dims...)
    @inbounds for ij in node_indices(space)
        node = slab_node_index(ij)
        local_geometry = get_local_geometry(space, ij, slabidx)
        u = get_node(space, arg1, ij, slabidx)
        foreach_axis(op) do vd
            @inline
            @inbounds Ju[axis_index(vd)][node] =
                local_geometry.J * contravariant(vd, u, local_geometry)
        end
        psi[node] = get_node(space, arg2, ij, slabidx)
    end

    out = slab_data(RT, FT, dims...)
    fill!(parent(out), zero(FT))
    @inbounds for ij in node_indices(space)
        node = slab_node_index(ij)
        foreach_axis(op) do vd
            @inline
            @inbounds begin
                i = ij[axis_index(vd)]
                Juᵈ = Ju[axis_index(vd)]
                for k in 1:(i - 1) # loop over half the indices, since F[i,k] = F[k,i]
                    node_k = slab_node_index(replace_index(ij, vd, k))
                    F = ((Juᵈ[node] + Juᵈ[node_k]) * (psi[node] + psi[node_k])) / 2
                    out[node] += D[i, k] * F
                    out[node_k] += D[k, i] * F
                end
            end
        end
    end
    @inbounds for ij in node_indices(space)
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_node_index(ij)] *= local_geometry.invJ
    end

    return Field(immutable_slab_data(out), space)
end

"""
    grad = Gradient()
    grad.(f)

Compute the (strong) gradient of `f` on each element, returning a
`CovariantVector`-field.

The ``i``th covariant component of the gradient is the partial derivative with
respect to the reference element:
```math
(\\nabla f)_i = \\frac{\\partial f}{\\partial \\xi^i}
```

Discretely, this can be written in matrix form as
```math
D_i f
```
where ``D_i`` is the derivative matrix along the ``i``th dimension.

## References
- [Taylor2010](@cite), equation 16
"""
struct Gradient{I, F <: FormType} <: SpectralElementOperator{I} end
Gradient() = Gradient{(), StrongForm}()
Gradient{I}() where {I} = Gradient{I, StrongForm}()
rebuild_operator(::Gradient{I, F}, space) where {I, F} =
    Gradient{operator_axes(space), F}()

operator_return_eltype(::Gradient{I}, ::Type{S}) where {I, S} =
    Geometry.gradient_result_type(Val(I), S)

# The strong gradient is Dᵢ f, while the weak gradient is -W⁻¹ Dᵢᵀ (W f); see
# form_deriv_entry and form_weighted_arg for the form-dependent factors. Only
# the weak form needs the final W⁻¹ rescale, which stays inlined in each
# apply_operator so that the strong form can skip that loop entirely.

Base.@propagate_inbounds function apply_operator(
    op::Gradient{I, F},
    space,
    slabidx,
    arg,
) where {I, F}
    form = F()
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = slab_data(RT, FT, slab_dims(op, Nq)...)
    fill!(parent(out), zero(FT))
    @inbounds for ij in node_indices(space)
        local_geometry = get_local_geometry(space, ij, slabidx)
        x = form_weighted_arg(form, local_geometry, get_node(space, arg, ij, slabidx))
        foreach_axis(op) do vd
            @inline
            @inbounds begin
                i = ij[axis_index(vd)]
                for k in 1:Nq
                    ∂f∂ξᵈ =
                        Geometry.covariant_basis_vector(
                            Val(I),
                            vd,
                            form_deriv_entry(form, D, k, i),
                        ) ⊗ x
                    out[slab_node_index(replace_index(ij, vd, k))] += ∂f∂ξᵈ
                end
            end
        end
    end
    # the weak form weights its argument by W without a Jacobian factor to divide
    # out, so it alone needs this pass; see form_weight_rescale
    if F === WeakForm
        @inbounds for ij in node_indices(space)
            local_geometry = get_local_geometry(space, ij, slabidx)
            W = local_geometry.WJ * local_geometry.invJ
            out[slab_node_index(ij)] /= W
        end
    end
    return Field(immutable_slab_data(out), space)
end

abstract type CurlSpectralElementOperator{I} <: SpectralElementOperator{I} end

"""
    curl_uses_component(op, ::Val{k})

Whether a curl over the operator's axes uses the `k`th covariant component of its
argument. Since ``ε^{i d k}`` vanishes unless `k ≠ d`, axis `d` never uses the
`d`th component: a curl over `(1,)` needs only ``u_2`` and ``u_3``, while a curl
over `(1, 2)` needs all three.
"""
@inline curl_uses_component(
    ::CurlSpectralElementOperator{I},
    ::Val{k},
) where {I, k} = unrolled_any(!=(k), I)

"""
    curl_covariant_components(op, form, u, local_geometry)

The covariant components of `u` as a 3-tuple, weighted as the `form` requires
(see [`form_weighted_arg`](@ref)), with `nothing` in place of the components that
a curl over the operator's axes does not use (see [`curl_uses_component`](@ref)).
"""
@inline curl_covariant_components(
    op::CurlSpectralElementOperator,
    form,
    u,
    lg,
) =
    unrolled_map((Val(1), Val(2), Val(3))) do vk
        curl_uses_component(op, vk) ?
        form_weighted_arg(form, lg, covariant(vk, u, lg)) : nothing
    end

"""
    curl_term(::Val{d}, c, u)

The contribution of axis `d` to a curl, ``ε^{i d k} c u_k``, where `u` holds the
covariant components of the argument (as returned by
[`curl_covariant_components`](@ref)) and `c` scales the derivative along axis
`d`. This is `c * (eᵈ × u)`, so only the components `k ≠ d` are read and the
others may be `nothing`. The weak form needs no separate expression: its sign
flip comes from `c` (see [`form_deriv_entry`](@ref)).
"""
@inline curl_term(::Val{1}, c, u) =
    Geometry.Contravariant123Vector(zero(c), -(c * u[3]), c * u[2])
@inline curl_term(::Val{2}, c, u) =
    Geometry.Contravariant123Vector(c * u[3], zero(c), -(c * u[1]))

"""
    curl = Curl()
    curl.(u)

Computes the per-element spectral (strong) curl of a covariant vector field ``u``.

Note: The vector field ``u`` needs to be excliclty converted to a `CovaraintVector`,
as then the `Curl` is independent of the local metric tensor.

The curl of a vector field ``u`` is a vector field with contravariant components
```math
(\\nabla \\times u)^i = \\frac{1}{J} \\sum_{jk} \\epsilon^{ijk} \\frac{\\partial u_k}{\\partial \\xi^j}
```
where ``J`` is the Jacobian determinant, ``u_k`` is the ``k``th covariant
component of ``u``, and ``\\epsilon^{ijk}`` are the [Levi-Civita
symbols](https://en.wikipedia.org/wiki/Levi-Civita_symbol#Three_dimensions_2).
In other words
```math
\\begin{bmatrix}
  (\\nabla \\times u)^1 \\\\
  (\\nabla \\times u)^2 \\\\
  (\\nabla \\times u)^3
\\end{bmatrix}
=
\\frac{1}{J} \\begin{bmatrix}
  \\frac{\\partial u_3}{\\partial \\xi^2} - \\frac{\\partial u_2}{\\partial \\xi^3} \\\\
  \\frac{\\partial u_1}{\\partial \\xi^3} - \\frac{\\partial u_3}{\\partial \\xi^1} \\\\
  \\frac{\\partial u_2}{\\partial \\xi^1} - \\frac{\\partial u_1}{\\partial \\xi^2}
\\end{bmatrix}
```

In matrix form, this becomes
```math
\\epsilon^{ijk} J^{-1} D_j u_k
```
Note that unused dimensions will be dropped: e.g. the 2D curl of a
`Covariant12Vector`-field will return a `Contravariant3Vector`.

## References
- [Taylor2010](@cite), equation 17
"""
struct Curl{I, F <: FormType} <: CurlSpectralElementOperator{I} end
Curl() = Curl{(), StrongForm}()
Curl{I}() where {I} = Curl{I, StrongForm}()
rebuild_operator(::Curl{I, F}, space) where {I, F} =
    Curl{operator_axes(space), F}()

operator_return_eltype(::Curl{I}, ::Type{S}) where {I, S} =
    Geometry.curl_result_type(Val(I), S)

# The strong curl is εⁱʲᵏ J⁻¹ Dⱼ uₖ, while the weak curl is
# -εⁱʲᵏ (WJ)⁻¹ Dⱼᵀ (W uₖ); see form_deriv_entry, form_weighted_arg, and
# form_jacobian_rescale for the form-dependent factors.

Base.@propagate_inbounds function apply_operator(
    op::Curl{I, F},
    space,
    slabidx,
    arg,
) where {I, F}
    form = F()
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = slab_data(RT, FT, slab_dims(op, Nq)...)
    fill!(parent(out), zero(FT))
    @inbounds for ij in node_indices(space)
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(space, arg, ij, slabidx)
        u = curl_covariant_components(op, form, v, local_geometry)
        foreach_axis(op) do vd
            @inline
            @inbounds begin
                i = ij[axis_index(vd)]
                for k in 1:Nq
                    out[slab_node_index(replace_index(ij, vd, k))] +=
                        curl_term(vd, form_deriv_entry(form, D, k, i), u)
                end
            end
        end
    end
    @inbounds for ij in node_indices(space)
        local_geometry = get_local_geometry(space, ij, slabidx)
        node = slab_node_index(ij)
        out[node] = form_jacobian_rescale(form, local_geometry, out[node])
    end
    return Field(immutable_slab_data(out), space)
end

# interplation / restriction
abstract type TensorOperator{I} <: SpectralElementOperator{I} end

return_space(op::TensorOperator, inspace) = op.space
operator_return_eltype(::TensorOperator, ::Type{S}) where {S} = S

"""
    i = Interpolate(space)
    i.(f)

Interpolates `f` to the `space`. If `space` has equal or higher polynomial
degree as the space of `f`, this is exact, otherwise it will be lossy.

In matrix form, it is the linear operator
```math
I = \\bigotimes_i I_i
```
where ``I_i`` is the barycentric interpolation matrix in the ``i``th dimension.

See also [`Restrict`](@ref).
"""
struct Interpolate{I, S} <: TensorOperator{I}
    space::S
end
Interpolate(space) = Interpolate{operator_axes(space), typeof(space)}(space)
Interpolate{()}(space) = Interpolate{operator_axes(space), typeof(space)}(space)

"""
    r = Restrict(space)
    r.(f)

Computes the projection of a field `f` on ``\\mathcal{V}_0`` to a lower degree
polynomial space `space` (``\\mathcal{V}_0^*``). `space` must be on the same
topology as the space of `f`, but have a lower polynomial degree.

It is defined as the field ``\\theta \\in \\mathcal{V}_0^*`` such that for all ``\\phi \\in \\mathcal{V}_0^*``
```math
\\int_\\Omega \\phi \\theta \\,d\\Omega = \\int_\\Omega \\phi f \\,d\\Omega
```
In matrix form, this is
```math
\\phi^\\top W^* J^* \\theta = (I \\phi)^\\top WJ f
```
where ``W^*`` and ``J^*`` are the quadrature weights and Jacobian determinant of
``\\mathcal{V}_0^*``, and ``I`` is the interpolation operator (see [`Interpolate`](@ref))
from ``\\mathcal{V}_0^*`` to ``\\mathcal{V}_0``. This reduces to
```math
\\theta = (W^* J^*)^{-1} I^\\top WJ f
```
"""
struct Restrict{I, S} <: TensorOperator{I}
    space::S
end
Restrict(space) = Restrict{operator_axes(space), typeof(space)}(space)
Restrict{()}(space) = Restrict{operator_axes(space), typeof(space)}(space)

# Interpolation and restriction contract the same interpolation matrix along every
# axis, so one implementation covers both, for any axis tuple `I`. They differ only
# in three factors, in the same way that the strong and weak forms of the
# differential operators differ in the `form_*` factors above: the matrix
# contracted along each axis, whether the argument is weighted by the Jacobian
# factor `WJ` of the input space, and whether the result is divided by the `WJ` of
# the output space.

"""
    tensor_matrix(op, FT, QS_out, QS_in)

The matrix that `op` contracts along each of its axes: the barycentric
interpolation matrix from `QS_in` to `QS_out` for [`Interpolate`](@ref), and the
transpose of the interpolation matrix from `QS_out` to `QS_in` for
[`Restrict`](@ref).
"""
@inline tensor_matrix(::Interpolate, ::Type{FT}, QS_out, QS_in) where {FT} =
    Quadratures.interpolation_matrix(FT, QS_out, QS_in)
@inline tensor_matrix(::Restrict, ::Type{FT}, QS_out, QS_in) where {FT} =
    Quadratures.interpolation_matrix(FT, QS_in, QS_out)' # transpose

"""
    tensor_weighted_arg(op, space_in, arg, ij, slabidx)

The value of `arg` at node `ij`, weighted as `op` requires: unweighted for
[`Interpolate`](@ref), and multiplied by the Jacobian factor `WJ` of the input
space for [`Restrict`](@ref), which integrates its argument against the test
functions of the output space.
"""
Base.@propagate_inbounds tensor_weighted_arg(
    ::Interpolate,
    space_in,
    arg,
    ij,
    slabidx,
) = get_node(space_in, arg, ij, slabidx)
Base.@propagate_inbounds tensor_weighted_arg(
    ::Restrict,
    space_in,
    arg,
    ij,
    slabidx,
) =
    get_local_geometry(space_in, ij, slabidx).WJ *
    get_node(space_in, arg, ij, slabidx)

"""
    tensor_rescale(op, space_out, ij, slabidx, x)

The result value `x` at node `ij`, with the weighting that `op` applied to its
argument divided back out: `x` itself for [`Interpolate`](@ref), and
`x / WJ` for [`Restrict`](@ref), whose ``(W^* J^*)^{-1}`` factor uses the
Jacobian factor of the output space.
"""
Base.@propagate_inbounds tensor_rescale(::Interpolate, space_out, ij, slabidx, x) = x
Base.@propagate_inbounds tensor_rescale(::Restrict, space_out, ij, slabidx, x) =
    x / get_local_geometry(space_out, ij, slabidx).WJ

"""
    TensorArg(op, space_in, arg, slabidx)

Reads node `ij` of the argument of a [`TensorOperator`](@ref), weighted as the
operator requires (see [`tensor_weighted_arg`](@ref)), in the form
[`contract_axis!`](@ref) expects of its source. Reading the argument this way is
what lets the first contraction consume it directly, without materialising a slab
of it first.
"""
struct TensorArg{O, S, A, SI}
    op::O
    space_in::S
    arg::A
    slabidx::SI
end
Base.@propagate_inbounds (a::TensorArg)(ij) =
    tensor_weighted_arg(a.op, a.space_in, a.arg, ij, a.slabidx)

"""
    TensorRescale(op, space_out, slabidx)

Divides the weighting that a [`TensorOperator`](@ref) applied to its argument back
out of the value at node `ij` (see [`tensor_rescale`](@ref)), in the form
[`contract_axis!`](@ref) expects of its `post`.
"""
struct TensorRescale{O, S, SI}
    op::O
    space_out::S
    slabidx::SI
end
Base.@propagate_inbounds (p::TensorRescale)(ij, x) =
    tensor_rescale(p.op, p.space_out, ij, p.slabidx, x)

Base.@propagate_inbounds function apply_operator(
    op::TensorOperator{I},
    space_out,
    slabidx,
    arg,
) where {I}
    FT = Spaces.undertype(space_out)
    space_in = axes(arg)
    QS_in = Spaces.quadrature_style(space_in)
    QS_out = Spaces.quadrature_style(space_out)
    Nq_in = Quadratures.degrees_of_freedom(QS_in)
    Nq_out = Quadratures.degrees_of_freedom(QS_out)
    M = tensor_matrix(op, FT, QS_out, QS_in)
    RT = eltype(arg)
    vds = axis_vals(op)
    out = slab_data(RT, FT, slab_dims(vds, Nq_out)...)
    # temporary storage, sized for the largest intermediate extent
    Nq_max = max(Nq_in, Nq_out)
    temps = contract_temps(vds) do
        slab_data(RT, FT, slab_dims(vds, Nq_max)...)
    end
    @inbounds contract_axes!(
        (temps..., out),
        vds,
        M,
        TensorArg(op, space_in, arg, slabidx),
        TensorRescale(op, space_out, slabidx),
        slab_dims(vds, Nq_in),
    )
    return Field(immutable_slab_data(out), space_out)
end


"""
    slab_axis_vals(data)

The axes of one slab of `data`, as a tuple of `Val`s: just `(Val(1),)` for a
layout with a single node along the second horizontal direction, and
`(Val(1), Val(2))` otherwise.
"""
@inline slab_axis_vals(::DataLayouts.VIJHWithF{<:Any, <:Any, <:Any, 1}) = (Val(1),)
@inline slab_axis_vals(::DataLayouts.VIJHWithF) = (Val(1), Val(2))

"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element, contracting `M`
along every axis of a slab of `in` (see [`contract_axes!`](@ref)). Unlike
[`Interpolate`](@ref) this works on data directly rather than on fields, so it can
write into the plotting layouts that drop a dimension
([`DataLayouts.VIH1`](@ref) and [`DataLayouts.IH1JH2`](@ref)).
"""
function tensor_product! end

function tensor_product!(
    out::Union{
        DataLayouts.VIJHWithF{S, Nv, Ni_out},
        DataLayouts.VIH1{S, Nv, Ni_out},
        DataLayouts.IH1JH2{S, Ni_out},
    },
    indata::DataLayouts.VIJHWithF{S, Nv, Ni_in, Nj_in},
    M::SMatrix{Ni_out, Ni_in},
) where {S, Nv, Ni_in, Nj_in, Ni_out}
    Nh_in = DataLayouts.nelems(indata)
    Nh_out = DataLayouts.nelems(out)
    # TODO: assumes the same number of levels (horizontal only)
    @assert Nh_in == Nh_out
    # the same M is contracted along every axis, so a slab with a second node axis
    # has to be square, on the way in and on the way out
    @assert Nj_in == 1 ||
            (Nj_in == Ni_in && DataLayouts.shape_params(out).Nj == Ni_out)
    # IH1JH2 keeps a single horizontal plane, so it can only take a single level
    @assert Nv == 1 || !(out isa DataLayouts.IH1JH2)
    vds = slab_axis_vals(indata)
    dims_in = slab_dims(vds, Ni_in)
    # temporary storage, sized for the largest intermediate extent
    Nq_max = max(Ni_in, Ni_out)
    temps = contract_temps(vds) do
        MArray{Tuple{1, Nq_max, Nq_max, 1}, S, 4, Nq_max * Nq_max}(undef)
    end
    @inbounds for h in 1:Nh_out, v in 1:Nv
        in_slab = slab(indata, v, h)
        out_slab = slab(out, v, h)
        contract_axes!(
            (temps..., out_slab),
            vds,
            M,
            SlabReader(in_slab),
            no_rescale,
            dims_in,
        )
    end
    return out
end

function tensor_product!(
    inout::DataLayouts.VIJHWithF{S, 1, Nij, Nij},
    M::SMatrix{Nij, Nij},
) where {S, Nij}
    inout_bc = Base.broadcastable(inout)
    tensor_product!(inout_bc, inout_bc, M)
end

"""
    matrix_interpolate(field, quadrature)

Computes the tensor product given a uniform quadrature `out = (M ⊗ M) * in` on each element.
Returns a 2D Matrix for plotting / visualizing 2D Fields.
"""
function matrix_interpolate end

function matrix_interpolate(
    field::Fields.SpectralElementField2D,
    Q_interp::Quadratures.Uniform{Nu},
) where {Nu}
    S = eltype(field)
    space = axes(field)
    topology = Spaces.topology(space)
    quadrature_style = Spaces.quadrature_style(space)
    mesh = topology.mesh
    n1, n2 = size(Meshes.elements(mesh))
    interp_data =
        DataLayouts.IH1JH2{S, Nu, Nu, nothing}(Matrix{S}(undef, (Nu * n1, Nu * n2)))
    M = Quadratures.interpolation_matrix(Float64, Q_interp, quadrature_style)
    Operators.tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end

function matrix_interpolate(
    field::Fields.ExtrudedFiniteDifferenceField,
    Q_interp::Union{Quadratures.Uniform{Nu}, Quadratures.ClosedUniform{Nu}},
) where {Nu}
    S = eltype(field)
    space = axes(field)
    quadrature_style = Spaces.quadrature_style(space)
    nl = Spaces.nlevels(space)
    n1 = Topologies.nlocalelems(Spaces.topology(space))
    interp_data =
        DataLayouts.VIH1{S, nl, Nu, nothing}(Matrix{S}(undef, (nl, Nu * n1)))
    M = Quadratures.interpolation_matrix(Float64, Q_interp, quadrature_style)
    Operators.tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end

"""
    matrix_interpolate(field, Nu::Integer)

Computes the tensor product given a uniform quadrature degree of Nu on each element.
Returns a 2D Matrix for plotting / visualizing 2D Fields.
"""
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())

function apply_operator(
    op::SpectralElementOperator{()},
    space,
    _,
    arg,
)
    RT = operator_return_eltype(op, eltype(arg))
    return map(Returns(zero(RT)), space)
end
