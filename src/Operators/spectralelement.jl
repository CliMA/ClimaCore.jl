"""
    FormType

Supertype of the singleton types [`StrongForm`](@ref) and [`WeakForm`](@ref),
which distinguish the variational form of a spectral element operator. The two
variants share the same interior computation; they differ only in the applied
derivative matrix (`D` vs. its integration-by-parts counterpart `-Dᵀ`) and in
the weights that multiply the argument and divide the result. Weak variants are
defined as aliases; e.g., `WeakDivergence()` is a `Divergence{WeakForm}`.
"""
abstract type FormType end

"""
    StrongForm()

The [`FormType`](@ref) of an operator that discretizes a derivative directly at
the quadrature points (e.g. [`Divergence`](@ref), [`Gradient`](@ref), and
[`Curl`](@ref)).
"""
struct StrongForm <: FormType end

"""
    WeakForm()

The [`FormType`](@ref) of an operator that discretizes the volume-integral
contribution of the corresponding weak-form expression, obtained after
integration by parts (e.g. [`WeakDivergence`](@ref), [`WeakGradient`](@ref), and
[`WeakCurl`](@ref)).
"""
struct WeakForm <: FormType end

"""
    materialize_buffer(arg)

Like `Base.materialize(arg)`, but storing the result of a lazy `Broadcasted`
expression in a buffer from [`buffer_similar`](@ref) that every thread in the
argument's scope can read; register-resident `Field`s (see
[`register_similar`](@ref)) are also copied into such a buffer. On GPUs two
buffers of equal byte size can share memory, so callers must keep their
lifetimes disjoint; see the buffer reuse invariant in [`apply_operator`](@ref).
"""
@inline materialize_buffer(arg) = arg
@inline materialize_buffer(bc::Base.Broadcast.Broadcasted) = constant_field(
    copyto!(
        buffer_similar(bc, drop_auto_broadcasters(Utilities.safe_eltype(bc))),
        bc;
        mask = Spaces.get_mask(axes(bc)),
    ),
)
@inline materialize_buffer(arg::Fields.Field) =
    DataLayouts.stored_in_registers(Fields.field_values(arg)) ?
    materialize_buffer(Base.broadcasted(identity, arg)) : arg

# Like materialize_buffer, but leaving arg lazy when it is a shallow Broadcasted
# over materialized values and a single thread owns the data (cross-thread
# scopes must publish through shared memory). Per benchmark_ops.jl, fusing
# Gradient's basis products wins; Divergence/Curl's deeper ones are 2x slower.
@inline fused_buffer(arg) = arg
@inline fused_buffer(bc::Base.Broadcast.Broadcasted) =
    has_private_buffers(bc) &&
    unrolled_all(arg -> !(arg isa Base.Broadcast.Broadcasted), bc.args) ? bc :
    materialize_buffer(bc)

"""
    has_private_buffers(arg)

Whether a buffer allocated for `arg` is private to the allocating thread, which
holds exactly when `arg`'s [`DataLayouts.DataScope`](@ref) is
[`DataLayouts.ThisThread`](@ref) (as on CPUs); otherwise buffers live in shared
memory and must obey the buffer reuse invariant in [`apply_operator`](@ref).
"""
@inline has_private_buffers(arg) =
    DataLayouts.DataScope(arg) == DataLayouts.ThisThread()

"""
    register_similar(arg, T)

Like `Base.similar(arg, T)`, but with the new `Field`'s data in each thread's
registers; used for every [`apply_operator`](@ref) destination, which only its
own thread reads and writes. This lets two applications in one fused expression
be live at once (see the buffer reuse invariant in [`apply_operator`](@ref)).
"""
register_similar(arg, ::Type{T}) where {T} = Fields.Field(
    DataLayouts.register_similar(Fields.field_values(Base.broadcastable(arg)), T),
    axes(arg),
)

"""
    buffer_similar(arg, T)

Like `Base.similar(arg, T)`, but always allocated through the argument's
[`DataLayouts.DataScope`](@ref) (shared memory on GPUs), never in per-thread
registers; used for every buffer whose values cross a thread boundary.
"""
buffer_similar(arg, ::Type{T}) where {T} = Fields.Field(
    DataLayouts.buffer_similar(Fields.field_values(arg), T),
    axes(arg),
)

# Lazy expression applying the component extractor f to each value of arg, with
# lg narrowed to just the metric f reads (at most one 3x3 tensor per point).
@inline function components_broadcasted(f::F, basis, arg, lg) where {F}
    eltype′ = drop_auto_broadcasters(Utilities.safe_eltype(arg))
    metric = Geometry.metric_for_components_type(basis, eltype′, lg)
    return isnothing(metric) ? Base.broadcasted(Base.Fix2(f, nothing), arg) :
           Base.broadcasted(f, arg, metric)
end

# Multiply arg by, or divide dest by, the Jacobian factor for the given form:
# J for StrongForm, WJ for WeakForm. The weighted values are read exactly once,
# at their own point, so they never cross a thread boundary and are only
# materialized when has_private_buffers holds.
jacobian_weight(::StrongForm, arg) = Fields.local_geometry_field(arg).J
jacobian_weight(::WeakForm, arg) = Fields.local_geometry_field(arg).WJ
@inline materialize_jacobian_weighted(form::FormType, arg) =
    maybe_private_buffer(Base.broadcasted(*, arg, jacobian_weight(form, arg)))
@inline jacobian_unweighted(form, dest) =
    Base.broadcasted(/, dest, jacobian_weight(form, dest))

# materialize_buffer for values never read across a thread boundary, applied
# only when has_private_buffers holds; a shared buffer would only replace a
# register with a shared memory round trip.
@inline maybe_private_buffer(arg) =
    has_private_buffers(arg) ? materialize_buffer(arg) : arg

# Lazily multiply arg by, or divide dest by, the quadrature weights W = WJ / J
# for WeakForm operators; a no-op for StrongForm. In both forms arg is
# materialized first via maybe_private_buffer, which must be skipped when
# buffers are shared: the buffer stays live for the whole application, and its
# byte size can equal the per-dimension intermediate's, so on GPUs the first
# dimension's intermediate would overwrite arg.
@inline materialize_quadrature_weighted(::StrongForm, arg) =
    maybe_private_buffer(arg)
@inline function materialize_quadrature_weighted(::WeakForm, arg)
    (; WJ, J) = Fields.local_geometry_field(arg)
    return Base.broadcasted(
        *, maybe_private_buffer(arg), Base.broadcasted(/, WJ, J),
    )
end

@inline quadrature_unweighted(::StrongForm, dest) = dest
@inline function quadrature_unweighted(::WeakForm, dest)
    (; WJ, J) = Fields.local_geometry_field(dest)
    return Base.broadcasted(*, dest, Base.broadcasted(/, J, WJ))
end

# The derivative matrix for the given form: D for StrongForm, and its discrete
# integration-by-parts counterpart -Dᵀ for WeakForm.
deriv_matrix(::StrongForm, dest) = Quadratures.differentiation_matrix(
    Spaces.undertype(axes(dest)),
    Spaces.quadrature_style(axes(dest)),
)
deriv_matrix(::WeakForm, dest) = -deriv_matrix(StrongForm(), dest)'

# The matrix that remaps arg into dest: I for Interpolate (StrongForm), and
# (I⁻¹)ᵀ for Restrict (WeakForm); the two satisfy a discrete "interpolation by
# parts" identity.
interp_matrix(::StrongForm, dest, arg) = Quadratures.interpolation_matrix(
    Spaces.undertype(axes(dest)),
    Spaces.quadrature_style(axes(dest)),
    Spaces.quadrature_style(axes(arg)),
)
interp_matrix(::WeakForm, dest, arg) = interp_matrix(StrongForm(), arg, dest)'

# When dest's data is thread-local (single-thread DataScope, or a
# RegisterArray), copy it into an immutable StaticArrays.SArray, which is
# always stack-allocated (an MArray heap-allocates unless every read and write
# is inlined). GPUs only have stack memory, so this is what allows GPU
# compilation without full inlining; it also cuts CPU compile time.
@inline function constant_field(dest)
    data = Fields.field_values(dest)
    is_thread_local =
        has_private_buffers(data) ||
        DataLayouts.parent_type(data) <: DataLayouts.RegisterArray
    return is_thread_local ?
           Fields.Field(DataLayouts.rebuild(data, StaticArrays.SArray), axes(dest)) :
           dest
end

# Tuple of the horizontal dimensions covered by a Field: (1, 2), (1,), (2,),
# or () (e.g., for a point or column Field).
function horizontal_dims(arg)
    (; Ni, Nj) = DataLayouts.vijh_params(arg)
    Nq = DataLayouts.nquadpoints(arg)
    return Nq == 1 ? () : Ni == Nj ? (1, 2) : (Ni == Nq ? 1 : 2,)
end

# Broadcasting requires use of spectral-element operations.
struct SpectralStyle <: Fields.AbstractFieldStyle end

Broadcast.BroadcastStyle(::SpectralStyle, ::Fields.FieldStyle) = SpectralStyle()

# A SpectralStyle broadcast expression with an operator of type F.
const SpectralBroadcasted{F} = Broadcast.Broadcasted{SpectralStyle, <:Any, F}

"""
    SpectralElementOperator

Operator applied to the quadrature points in each spectral element of a `Field`.
Each subtype must define [`return_eltype`](@ref) and [`apply_operator`](@ref).
"""
abstract type SpectralElementOperator <: AbstractOperator end

slab(op::SpectralElementOperator, _...) = op
level(op::SpectralElementOperator, _...) = op

function Broadcast.broadcasted(op::SpectralElementOperator, args...)
    args′ = unrolled_tuple_map(Broadcast.broadcastable, args)
    style = Broadcast.result_style(SpectralStyle(), Broadcast.combine_styles(args′...))
    return Broadcast.broadcasted(style, op, args′...)
end

# As in Fields.sliced_broadcasted, but keeping the SpectralStyle so slab-level
# operator nodes are still recognized by apply_operators.
@inline Fields.sliced_broadcasted(op::SpectralElementOperator, args, axes) =
    Broadcast.Broadcasted(
        Broadcast.result_style(SpectralStyle(), Broadcast.combine_styles(args...)),
        op,
        args,
        axes,
    )

# Apply size/scope primitives to an operator-free pointwise equivalent of bc.
for f in (:size, :length, :ndims)
    @eval Base.$f(bc::SpectralBroadcasted) = $f(drop_operators(bc))
end
for f in (:DataScope, :shape_params, :inferred_size, :nelems)
    @eval DataLayouts.$f(bc::SpectralBroadcasted) = DataLayouts.$f(drop_operators(bc))
end

drop_operators(arg) = arg
drop_operators(bc::SpectralBroadcasted) =
    Fields.sliced_broadcasted(bc.f, unrolled_tuple_map(drop_operators, bc.args), nothing)
drop_operators(bc::SpectralBroadcasted{<:SpectralElementOperator}) =
    Fields.sliced_broadcasted(
        Returns(new(eltype(bc))), unrolled_tuple_map(drop_operators, bc.args), nothing,
    )

# Allocate materialized results from the broadcast's own data; the space-based
# LazyField fallback would allocate through coordinate data whose kernel-wide
# scope has no allocation method inside a fused slice loop.
Base.similar(bc::SpectralBroadcasted, ::Type{T}) where {T} =
    similar(drop_operators(bc), T)

# Copy one slab of an operator-free expression into one slab of the destination.
# Broadcast expressions skip .= to avoid inferring the per-slab point loop into
# both of its materialize! layers; other arguments are rare enough to keep .=.
@inline copyto_slab!(dest, bc::Fields.LazyField) = copyto!(
    Fields.field_values(dest),
    Base.Broadcast.instantiate(Fields.field_values(bc));
    mask = Spaces.get_mask(axes(dest)),
)
@inline copyto_slab!(dest, arg) = (dest .= arg)

# Evaluate copyto! slab by slab, replacing operator broadcasts with pointwise ones.
function Base.copyto!(
    dest::Fields.Field, bc::SpectralBroadcasted; mask = DataLayouts.NoMask(),
)
    bc_no_space = strip_space(bc, axes(dest)) # Drop copies of space before sending to GPU.
    # A mask cannot skip slabs: a slab is live whenever any of its columns is
    # active, and a spectral operator reads every point of its slab.
    DataLayouts.foreach_slab(dest, bc_no_space) do dest_slab, bc_slab_no_space
        bc_slab = unstrip_space(bc_slab_no_space, axes(dest_slab))
        copyto_slab!(dest_slab, apply_operators(bc_slab))
    end
    call_post_op_callback() && post_op_callback(dest, dest, bc; mask)
    return dest
end

"""
    apply_operator(op, args...)

Eagerly evaluate the [`SpectralElementOperator`](@ref) `op` over one slab of
slab `Field`s and/or lazy pointwise broadcasts over slabs.

# Buffer reuse invariant

On GPUs, buffers of equal byte size can be assigned to the same shared memory
(see `DataLayouts.scoped_static_array`), so each `apply_operator` method keeps
at most one buffer of each size live at a time, separating every reuse from the
previous use with a `DataLayouts.synchronize`. The destination stays live for
the whole application and may alias a *different* live application's destination
(as in `@. wdiv(grad(a)) + wdiv(grad(b))`), so it lives in per-thread registers
([`register_similar`](@ref)), as do temporaries materialized inside a fused
slice loop; [`materialize_buffer`](@ref) republishes register-resident values
whenever an operator must read them across threads. Two configurations rely on
their allocation sites compiling into one unit (equal-size globals merge only
across separately compiled functions): interpolation to an equal or lower
degree gives a buffered argument and `sequential_muladd_slab!`'s partial result
the same byte size, and `SplitDivergence`'s buffered second argument stays live
across its equal-size per-dimension intermediates. Both are pinned by GPU value
tests in `test/Operators/spectralelement/gpu_rectilinear.jl`.
"""
function apply_operator end

# Replace every spectral operator broadcast in bc with the result of the
# corresponding apply_operator call, which is evaluated eagerly.
apply_operators(arg) = arg
apply_operators(bc::SpectralBroadcasted) =
    Broadcast.broadcasted(bc.f, unrolled_tuple_map(apply_operators, bc.args)...)
apply_operators(bc::SpectralBroadcasted{<:SpectralElementOperator}) =
    scoped_apply_operator(
        DataLayouts.DataScope(bc),
        Val(inlined_buffer_bytes(bc) <= MAX_INLINED_BUFFER_BYTES),
        bc,
    )

# Upper bound on the per-thread shared memory that inlining ONE broadcast
# expression's operator applications requests: each application is charged
# 2 buffers of its largest eltype sizeof. Assumes each slice's scope has at
# least as many threads as points; halving that doubles every buffer (see
# DataLayouts.slice_subscope).
inlined_buffer_bytes(arg) = 0
inlined_buffer_bytes(bc::Union{Broadcast.Broadcasted, SpectralBroadcasted}) =
    +(0, unrolled_tuple_map(inlined_buffer_bytes, bc.args)...)
inlined_buffer_bytes(bc::SpectralBroadcasted{<:SpectralElementOperator}) =
    2 * max(unrolled_tuple_map(sizeof ∘ eltype, (bc, bc.args...))...) +
    +(0, unrolled_tuple_map(inlined_buffer_bytes, bc.args)...)

# Inline each operator unless its expression asks a block for more than CUDA's
# 48 KB of static shared memory (a compilation error): 48 * 1024 bytes over
# MAX_SUBBLOCK_LAUNCH_THREADS = 256 threads. The bound's ~2x slack fits two
# at-budget expressions per kernel; RESIDUAL RISK: three or more still overflow,
# signaled only by the ptxas error.
const MAX_INLINED_BUFFER_BYTES = 48 * 1024 ÷ 256
@inline scoped_apply_operator(scope, ::Val{true}, bc) =
    apply_operator(bc.f, unrolled_tuple_map(apply_operators, bc.args)...)
@noinline scoped_apply_operator(scope, ::Val{false}, bc) =
    apply_operator(bc.f, unrolled_tuple_map(apply_operators, bc.args)...)

# The muladd_slab! variants loop over DataLayouts instead of Fields so slicing
# does not construct a new space type per distinct argument type. Instantiate so
# every node carries its own axes; without them, Base recombines the arguments'
# axes once per node per quadrature point.
slab_data(arg) = Broadcast.instantiate(Fields.field_values(Base.broadcastable(arg)))

# Zero out the destination before calling muladd_slab! or one of its variants.
@inline muladd_slab_init!(dest) =
    fill!(slab_data(dest), zero(eltype(Base.broadcastable(dest))))

# Read one point of a muladd_slab! argument. A fused lazy expression (see
# fused_buffer) is evaluated at the point rather than sliced: slicing would
# rebuild its axes-free Broadcasted tree at every point.
@inline arg_point_value(arg_data, i, j) = @inbounds column(arg_data, i, j, 1)[]
@inline arg_point_value(bc::DataLayouts.LazyDataLayout, i, j) =
    @inbounds Base.Broadcast._broadcast_getindex(bc, CartesianIndex(1, i, j, 1))

# The function passed to unrolled_sum in muladd_slab!.
sum_value(matrix, arg_value::F, dim::Union{Val{:i}, Val{:j}}, i, j) where {F} =
    dim isa Val{:i} ?
    (i′ -> (@inbounds matrix[i, i′] * arg_value(i′, j))) :
    (j′ -> (@inbounds matrix[j, j′] * arg_value(i, j′)))

# Indices summed over by the muladd_slab! variants, as a Tuple (a compile-time
# constant) rather than a UnitRange, so unrolled_sum stays on UnrolledUtilities'
# Tuple fast path instead of its generic output-type-promotion chain
# (re-inferred per matrix/argument type combination).
@inline summed_indices(matrix) = ntuple(identity, Val(size(matrix, 2)))

# Set dest_slice .+= matrix * arg_slice for each 1D i or j slice of the inputs.
# All threads that materialize arg must be synchronized before this is called.
# A single thread owning the whole slab (as on CPUs) visits a slice as a unit;
# otherwise each thread may only write the destination points in its own
# registers (see register_similar), so it visits them one at a time.
@inline muladd_slab!(dest, matrix, arg, dim) =
    has_private_buffers(dest) ? sliced_muladd_slab!(dest, matrix, arg, dim) :
    pointwise_muladd_slab!(dest, matrix, arg, dim)

@inline function pointwise_muladd_slab!(dest, matrix, arg, dim, clip = Val(false))
    arg_data = slab_data(arg)
    DataLayouts.foreach_column(
        slab_data(dest); enumerate = Val(true),
    ) do dest_index, dest_point
        (i, j, _) = Tuple(dest_index)
        arg_value(i′, j′) = arg_point_value(arg_data, i′, j′)
        in_bounds = if clip isa Val{true}
            (; Ni, Nj) = DataLayouts.vijh_params(arg_data)
            Nq = size(matrix, 1)
            dim isa Val{:i} ? (i <= Nq && j <= Nj) : (i <= Ni && j <= Nq)
        else
            true
        end
        in_bounds && @inbounds dest_point[] += unrolled_sum(
            sum_value(matrix, arg_value, dim, i, j),
            summed_indices(matrix),
        )
        nothing
    end
end

# Destination indices come from each_slice_index so their bounds checks are
# still elided under --check-bounds: an index of unproven provenance lets
# dest's MArray escape into the check's error path, moving it to the heap.
@inline function sliced_muladd_slab!(dest, matrix, arg, dim)
    (arg_data, dest_data) = (slab_data(arg), slab_data(dest))
    n′s = summed_indices(matrix)
    indices = DataLayouts.each_slice_index(column, dest_data)
    ordered(k, n) = dim isa Val{:i} ? (k, n) : (n, k)
    @inbounds for n in axes(indices, dim isa Val{:i} ? 2 : 1)
        values = unrolled_tuple_map(n′ -> arg_point_value(arg_data, ordered(n′, n)...), n′s)
        for k in axes(indices, dim isa Val{:i} ? 1 : 2)
            column(dest_data, Tuple(indices[ordered(k, n)..., 1])...)[] +=
                unrolled_sum(n′ -> matrix[k, n′] * values[n′], n′s)
        end
    end
end

# Like muladd_slab!, but skipping output points beyond the size of arg along
# the non-sliced dimension or of matrix along the sliced one; either can occur
# in sequential_muladd_slab!, where partial_result has the larger size.
@inline clipped_muladd_slab!(dest, matrix, arg, dim) =
    pointwise_muladd_slab!(dest, matrix, arg, dim, Val(true))

# Set dest_slice[n] .+= (matrix * Fₙ)[n] for each 1D i or j slice of the inputs,
# where Fₙ = (arg1[n] + arg1) * (arg2[n] + arg2) / 2 is a slice of the symmetric
# two-point flux tensor F[n, m] = (arg1[n] + arg1[m]) * (arg2[n] + arg2[m]) / 2.
@inline function split_muladd_slab!(dest, matrix, arg1, arg2, dim)
    arg1_data = slab_data(arg1)
    arg2_data = slab_data(arg2)
    DataLayouts.foreach_column(
        slab_data(dest); enumerate = Val(true),
    ) do dest_index, dest_point
        (i, j, _) = Tuple(dest_index)
        arg1_value(i′, j′) = arg_point_value(arg1_data, i′, j′)
        arg2_value(i′, j′) = arg_point_value(arg2_data, i′, j′)
        flux_value(i′, j′) =
            @inbounds (arg1_value(i, j) + arg1_value(i′, j′)) *
                      (arg2_value(i, j) + arg2_value(i′, j′)) / 2
        @inbounds dest_point[] += unrolled_sum(
            sum_value(matrix, flux_value, dim, i, j),
            summed_indices(matrix),
        )
    end
end

# Apply clipped_muladd_slab! along all available horizontal dimensions (or
# plain muladd_slab! when only one is available). The two synchronizations make
# every write to partial_result visible before it is read, and every read
# complete before its buffer can be reused (see the buffer reuse invariant).
@inline sequential_muladd_slab!(dest, matrix, arg) =
    if horizontal_dims(arg) == (1,)
        muladd_slab!(dest, matrix, arg, Val(:i))
    elseif horizontal_dims(arg) == (2,)
        muladd_slab!(dest, matrix, arg, Val(:j))
    elseif horizontal_dims(arg) == (1, 2)
        # partial_result crosses threads (the :j pass reads the :i pass's
        # points), so it must be a shared buffer, never registers.
        partial_result =
            DataLayouts.nquadpoints(dest) > DataLayouts.nquadpoints(arg) ?
            buffer_similar(dest, eltype(dest)) : buffer_similar(arg, eltype(arg))
        muladd_slab_init!(partial_result)
        clipped_muladd_slab!(partial_result, matrix, arg, Val(:i))
        DataLayouts.synchronize(DataLayouts.DataScope(partial_result))
        clipped_muladd_slab!(dest, matrix, partial_result, Val(:j))
        DataLayouts.synchronize(DataLayouts.DataScope(partial_result))
    end

# Zero out dest, then muladd! each dimension h's contribution, materialized by
# intermediate(Val(h)). The two intermediates have the same byte size and would
# alias, so each is materialized, used, and retired before the next; the
# synchronizations (the final one protects later applications) enforce this.
# See the buffer reuse invariant in apply_operator.
@inline function muladd_slab_dims!(
    muladd!::M, intermediate::I, scope, dims, dest, matrix, trailing_args...,
) where {M, I}
    muladd_slab_init!(dest)
    if unrolled_in(1, dims)
        arg = intermediate(Val(1))
        DataLayouts.synchronize(scope)
        muladd!(dest, matrix, arg, trailing_args..., Val(:i))
        DataLayouts.synchronize(scope)
    end
    if unrolled_in(2, dims)
        arg = intermediate(Val(2))
        DataLayouts.synchronize(scope)
        muladd!(dest, matrix, arg, trailing_args..., Val(:j))
        DataLayouts.synchronize(scope)
    end
    DataLayouts.synchronize(scope)
    return dest
end

# Contravariant component along horizontal dimension h of a Jacobian-weighted
# argument, and the covariant basis vector eʰ that Gradient multiplies by.
@inline materialize_contravariant(::Val{h}, arg, lg) where {h} = materialize_buffer(
    components_broadcasted(
        h == 1 ? Geometry.contravariant1 : Geometry.contravariant2,
        Geometry.Contravariant(),
        arg,
        lg,
    ),
)
@inline covariant_basis_vector(::Val{h}) where {h} =
    h == 1 ? Geometry.Covariant1Vector(true) : Geometry.Covariant2Vector(true)

# Levi-Civita contraction εʰⁿᵐ argₙ eₘ of a quadrature-weighted argument, whose
# two nonzero terms reinterpret the covariant components f₁ and f₂ of arg as the
# contravariant vectors V₁ and V₂.
@inline function materialize_levi_civita(::Val{h}, arg, lg) where {h}
    (V₁, f₁, V₂, f₂) =
        h == 1 ?
        (
            Geometry.Contravariant3Vector, Geometry.covariant2,
            Geometry.Contravariant2Vector, Geometry.covariant3,
        ) :
        (
            Geometry.Contravariant1Vector, Geometry.covariant3,
            Geometry.Contravariant3Vector, Geometry.covariant1,
        )
    term(V, f) =
        Base.broadcasted(V, components_broadcasted(f, Geometry.Covariant(), arg, lg))
    return materialize_buffer(Base.broadcasted(-, term(V₁, f₁), term(V₂, f₂)))
end

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
struct Divergence{F <: FormType} <: SpectralElementOperator end
Divergence() = Divergence{StrongForm}()

return_space(::Divergence, space) = space
return_eltype(::Divergence, arg) = Geometry.divergence_result_type(eltype(arg))

# Strong form is J⁻¹ ∑ₕ Dₕ J argʰ, weak form is -(WJ)⁻¹ ∑ₕ Dₕᵀ WJ argʰ.
function apply_operator(op::Divergence{F}, arg) where {F}
    dims = horizontal_dims(arg)
    lg = Fields.local_geometry_field(arg)
    arg′ = materialize_jacobian_weighted(F(), arg)
    dest = register_similar(arg, return_eltype(op, arg))
    matrix = isempty(dims) ? nothing : deriv_matrix(F(), dest)
    @inline contravariant(dim) = materialize_contravariant(dim, arg′, lg)
    muladd_slab_dims!(
        muladd_slab!, contravariant, DataLayouts.DataScope(arg), dims, dest, matrix,
    )
    return jacobian_unweighted(F(), constant_field(dest))
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

 1. **Conservation:** The split operator conserves ``\\rho \\mathbf{u} \\psi``
 2. **Consistency:** If ``\\psi = 1``, the split operator degenerates to the
    weak formulation of ``\\nabla \\cdot \\rho \\mathbf{u}`` (mass continuity)
 3. **Complexity:** The split operator has the same ``O(N^2)`` complexity per
    element as the strong and weak operators, but needs twice as many operations

# References

  - Fisher, T. C., & Carpenter, M. H. (2013). High-order entropy stable finite difference schemes for nonlinear conservation laws: Finite domains. Journal of Computational Physics, 252, 518-557. [https://doi.org/10.1016/j.jcp.2013.06.014](https://doi.org/10.1016/j.jcp.2013.06.014)
  - Gassner, G. J. (2013). A skew-symmetric discontinuous Galerkin spectral element discretization and its relation to SBP-SAT finite difference methods. SIAM Journal on Scientific Computing, 35, A1233-A1253. [https://doi.org/10.1137/120890144](https://doi.org/10.1137/120890144)
"""
struct SplitDivergence <: SpectralElementOperator end

return_space(::SplitDivergence, space, _) = space
return_eltype(::SplitDivergence, arg1, arg2) =
    Geometry.mul_return_type(Geometry.divergence_result_type(eltype(arg1)), eltype(arg2))

# Split form at index n is J⁻¹ ∑ₕ [Dₕ - Diag(Dₕ)] Fʰ[n, :], where F[n, :] is a
# slice of the tensor F[n, m] = (arg1[n] + arg1[m]) * (arg2[n] + arg2[m]) / 2.
function apply_operator(op::SplitDivergence, arg1, arg2)
    dims = horizontal_dims(arg1)
    lg = Fields.local_geometry_field(arg1)
    arg1′ = materialize_jacobian_weighted(StrongForm(), arg1)
    arg2′ = materialize_buffer(arg2)
    dest = register_similar(arg1, return_eltype(op, arg1, arg2))
    matrix = if isempty(dims)
        nothing
    else
        full_matrix = deriv_matrix(StrongForm(), dest)
        full_matrix - LinearAlgebra.Diagonal(full_matrix)
    end
    @inline contravariant(dim) = materialize_contravariant(dim, arg1′, lg)
    muladd_slab_dims!(
        split_muladd_slab!, contravariant, DataLayouts.DataScope(arg1), dims, dest,
        matrix, arg2′,
    )
    return jacobian_unweighted(StrongForm(), constant_field(dest))
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
struct Gradient{F <: FormType} <: SpectralElementOperator end
Gradient() = Gradient{StrongForm}()

return_space(::Gradient, space) = space
return_eltype(::Gradient, arg) =
    Geometry.gradient_result_type(Val(horizontal_dims(arg)), eltype(arg))

# Strong form is ∑ₕ Dₕ eʰ ⊗ f, weak form is -W⁻¹ ∑ₕ Dₕᵀ eʰ ⊗ W f.
function apply_operator(op::Gradient{F}, arg) where {F}
    dims = horizontal_dims(arg)
    arg′ = materialize_quadrature_weighted(F(), arg)
    dest = register_similar(arg, return_eltype(op, arg))
    matrix = isempty(dims) ? nothing : deriv_matrix(F(), dest)
    @inline eʰ_arg′(dim) =
        fused_buffer(Base.broadcasted(⊗, (covariant_basis_vector(dim),), arg′))
    muladd_slab_dims!(
        muladd_slab!, eʰ_arg′, DataLayouts.DataScope(arg), dims, dest, matrix,
    )
    return quadrature_unweighted(F(), constant_field(dest))
end

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
struct Curl{F <: FormType} <: SpectralElementOperator end
Curl() = Curl{StrongForm}()

return_space(::Curl, space) = space
return_eltype(::Curl, arg) =
    Geometry.curl_result_type(Val(horizontal_dims(arg)), eltype(arg))

# Strong form is J⁻¹ ∑ₕₙₘ εʰⁿᵐ Dₕ argₙ eₘ, weak form is -(WJ)⁻¹ ∑ₕₙₘ εʰⁿᵐ Dₕᵀ W argₙ eₘ.
function apply_operator(op::Curl{F}, arg) where {F}
    dims = horizontal_dims(arg)
    lg = Fields.local_geometry_field(arg)
    arg′ = materialize_quadrature_weighted(F(), arg)
    dest = register_similar(arg, return_eltype(op, arg))
    matrix = isempty(dims) ? nothing : deriv_matrix(F(), dest)
    @inline εʰⁿᵐ_arg′ₙ_eₘ(dim) = materialize_levi_civita(dim, arg′, lg)
    muladd_slab_dims!(
        muladd_slab!, εʰⁿᵐ_arg′ₙ_eₘ, DataLayouts.DataScope(arg), dims, dest, matrix,
    )
    return jacobian_unweighted(F(), constant_field(dest))
end

abstract type TensorOperator <: SpectralElementOperator end

# A TensorOperator's result is shaped by its target space, not its argument's.
# The constant is broadcasted over a Field (not its layout) so the node keeps
# the FieldStyle that the size and scope queries expect.
drop_operators(bc::SpectralBroadcasted{<:TensorOperator}) = Fields.sliced_broadcasted(
    Returns(new(eltype(bc))), (Fields.local_geometry_field(bc.f.space),), nothing,
)

return_space(op::TensorOperator, _) = op.space
return_eltype(::TensorOperator, arg) = eltype(arg)

Base.@propagate_inbounds slab(op::TensorOperator, inds...) =
    unionall_type(typeof(op))(slab(op.space, inds...))
Base.@propagate_inbounds level(op::TensorOperator, inds...) =
    unionall_type(typeof(op))(level(op.space, inds...))

Adapt.adapt_structure(to, op::TensorOperator) =
    unionall_type(typeof(op))(Adapt.adapt(to, op.space))

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
struct Interpolate{S} <: TensorOperator
    space::S
end

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
struct Restrict{S} <: TensorOperator
    space::S
end

# Interpolate leaves its argument unweighted, while Restrict weights it by WJ
# and divides the result by WJ.
tensor_form(::Interpolate) = StrongForm()
tensor_form(::Restrict) = WeakForm()
tensor_weighted(::Interpolate, arg) = materialize_buffer(arg)
function tensor_weighted(::Restrict, arg)
    # Restrict reads the lazily weighted argument at other threads' points, so
    # register-resident values cannot reach it: a copy republished through
    # materialize_buffer would have the same byte size as the arg-sized partial
    # result of sequential_muladd_slab! and alias it (see the buffer reuse
    # invariant). Fail loudly instead of silently reading each thread's own points.
    DataLayouts.stored_in_registers(Fields.field_values(arg)) && throw(
        ArgumentError(
            "Restrict cannot read register-resident values on GPUs; \
             materialize its argument outside the fused loop",
        ),
    )
    return materialize_jacobian_weighted(WeakForm(), arg)
end
tensor_unweighted(::Interpolate, dest) = dest
tensor_unweighted(::Restrict, dest) = jacobian_unweighted(WeakForm(), dest)

function apply_operator(op::TensorOperator, arg)
    arg′ = tensor_weighted(op, arg)
    DataLayouts.synchronize(DataLayouts.DataScope(arg))
    # The destination's shape comes from the output space's local geometry type
    # (not a value, which would make the buffer's size dynamic and its
    # allocation dynamically dispatched) and its scope from the argument.
    dest_space = slab(op.space, 1, 1)
    dest_shape = DataLayouts.reassign(
        Spaces.local_geometry_data(dest_space),
        DataLayouts.DataScope(slab_data(arg′)),
    )
    dest =
        Fields.Field(DataLayouts.register_similar(dest_shape, eltype(arg)), dest_space)
    matrix = interp_matrix(tensor_form(op), dest, arg)
    muladd_slab_init!(dest)
    # arg′ and sequential_muladd_slab!'s own partial_result buffer are both
    # retired by the synchronization at the end of that function.
    sequential_muladd_slab!(dest, matrix, arg′)
    # Synchronize before returning; see the comment in the Divergence method.
    DataLayouts.synchronize(DataLayouts.DataScope(arg))
    return tensor_unweighted(op, constant_field(dest))
end


"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element.
"""
function tensor_product! end

function tensor_product!(
    out::Union{
        DataLayouts.VIJHWithF{S, Nv, Ni_out, 1},
        DataLayouts.VIH1{S, Nv, Ni_out},
    },
    indata::DataLayouts.VIJHWithF{S, Nv, Ni_in, 1},
    M::SMatrix{Ni_out, Ni_in},
) where {S, Nv, Ni_out, Ni_in}
    Nh_in = DataLayouts.nelems(indata)
    Nh_out = DataLayouts.nelems(out)
    # TODO: assumes the same number of levels (horizontal only)
    @assert Nh_in == Nh_out
    @inbounds for h in 1:Nh_out, v in 1:Nv
        in_slab = slab(indata, v, h)
        out_slab = slab(out, v, h)
        for i in 1:Ni_out
            r = M[i, 1] * in_slab[1]
            for ii in 2:Ni_in
                r = muladd(M[i, ii], in_slab[ii], r)
            end
            out_slab[i] = r
        end
    end
    return out
end

function tensor_product!(
    out::DataLayouts.VIJHWithF{S, Nv, Nij_out, Nij_out},
    indata::DataLayouts.VIJHWithF{S, Nv, Nij_in, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nv, Nij_out, Nij_in}
    Nh = size(indata, 4)
    @assert Nh == size(out, 4)

    # temporary storage
    temp = MArray{Tuple{1, Nij_out, Nij_in, 1}, S, 4, Nij_out * Nij_in}(undef)

    @inbounds for h in 1:Nh, v in 1:Nv
        in_slab = slab(indata, v, h)
        out_slab = slab(out, v, h)
        for j in 1:Nij_in, i in 1:Nij_out
            temp[1, i, j, 1] = rmatmul1(M, in_slab, i, j)
        end
        for j in 1:Nij_out, i in 1:Nij_out
            out_slab[1, i, j, 1] = rmatmul2(M, temp, i, j)
        end
    end
    return out
end

function tensor_product!(
    out::DataLayouts.IH1JH2{S, Nij_out, Nij_out},
    indata::DataLayouts.VIJHWithF{S, 1, Nij_in, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}
    Nh = DataLayouts.nelems(indata)
    @assert Nh == DataLayouts.nelems(out)

    # temporary storage
    temp = MArray{Tuple{1, Nij_out, Nij_in, 1}, S, 4, Nij_out * Nij_in}(undef)

    @inbounds for h in 1:Nh
        in_slab = slab(indata, 1, h)
        out_slab = slab(out, 1, h)
        for j in 1:Nij_in, i in 1:Nij_out
            temp[1, i, j, 1] = rmatmul1(M, in_slab, i, j)
        end
        for j in 1:Nij_out, i in 1:Nij_out
            out_slab[i, j] = rmatmul2(M, temp, i, j)
        end
    end
    return out
end

function tensor_product!(
    out_slab::DataLayouts.VIJHWithF{S, 1, Nij_out, Nij_out, 1},
    in_slab::DataLayouts.VIJHWithF{S, 1, Nij_in, Nij_in, 1},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}
    # temporary storage
    temp = MArray{Tuple{1, Nij_out, Nij_in, 1}, S, 4, Nij_out * Nij_in}(undef)
    @inbounds for j in 1:Nij_in, i in 1:Nij_out
        temp[1, i, j, 1] = rmatmul1(M, in_slab, i, j)
    end
    @inbounds for j in 1:Nij_out, i in 1:Nij_out
        out_slab[1, i, j, 1] = rmatmul2(M, temp, i, j)
    end
    return out_slab
end

function tensor_product!(
    inout::DataLayouts.VIJHWithF{S, Nv, Nij, Nij},
    M::SMatrix{Nij, Nij},
) where {S, Nv, Nij}
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
    tensor_product!(interp_data, Fields.field_values(field), M)
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
    tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end

"""
    matrix_interpolate(field, Nu::Integer)

Computes the tensor product given a uniform quadrature degree of Nu on each element.
Returns a 2D Matrix for plotting / visualizing 2D Fields.
"""
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())

"""
    rmatmul1(W, S, i, j)

Recursive matrix product along the 1st dimension of `S`. Equivalent to:

    mapreduce(*, +, W[i,:], S[:,j])
"""
function rmatmul1(W, S, i, j)
    Nq = size(W, 2)
    @inbounds r = W[i, 1] * S[1, 1, j, 1]
    @inbounds for ii in 2:Nq
        r = muladd(W[i, ii], S[1, ii, j, 1], r)
    end
    return r
end

"""
    rmatmul2(W, S, i, j)

Recursive matrix product along the 2nd dimension `S`. Equivalent to:

    mapreduce(*, +, W[j,:], S[i, :])
"""
function rmatmul2(W, S, i, j)
    Nq = size(W, 2)
    @inbounds r = W[j, 1] * S[1, i, 1, 1]
    @inbounds for jj in 2:Nq
        r = muladd(W[j, jj], S[1, i, jj, 1], r)
    end
    return r
end

# Disabling constant propagation here cuts about a quarter of the inference
# allocation of a spectral expression (keeping the one-process test suite inside
# a 16 GB runner). The @inline annotations on the buffer helpers, weighting
# functions, and per-dimension closures are load-bearing for CPU runtime:
# without them, buffer Fields cross through memory and combine_axes runs per
# slab, costing 30-120% of a fused expression's runtime. Also inlining
# apply_operator(s) buys little (~10%) and roughly doubles LLVM time and memory.
@drop_constprop apply_operator, muladd_slab!, muladd_slab_dims!,
materialize_buffer, maybe_private_buffer, materialize_jacobian_weighted,
materialize_quadrature_weighted, jacobian_unweighted,
quadrature_unweighted, fused_buffer, register_similar, buffer_similar
