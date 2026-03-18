import UnrolledUtilities: unrolled_map

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

Return a tuple of the axis indicies a given field operator works over.
"""
function operator_axes end

operator_axes(space::Spaces.AbstractSpace) = ()
operator_axes(space::Spaces.SpectralElementSpace1D) = (1,)
operator_axes(space::Spaces.SpectralElementSpace2D) = (1, 2)
operator_axes(space::Spaces.SpectralElementSpaceSlab1D) = (1,)
operator_axes(space::Spaces.SpectralElementSpaceSlab2D) = (1, 2)
operator_axes(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    operator_axes(Spaces.horizontal_space(space))


function node_indices(space::Spaces.SpectralElementSpace1D)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    CartesianIndices((Nq,))
end
function node_indices(space::Spaces.SpectralElementSpace2D)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    CartesianIndices((Nq, Nq))
end
node_indices(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    node_indices(Spaces.horizontal_space(space))

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


function Base.Broadcast.broadcasted(op::SpectralElementOperator, args...)
    args′ = map(Base.Broadcast.broadcastable, args)
    style = Base.Broadcast.result_style(
        SpectralStyle(),
        Base.Broadcast.combine_styles(args′...),
    )
    Base.Broadcast.broadcasted(style, op, args′...)
end

Base.Broadcast.broadcasted(
    ::SpectralStyle,
    op::SpectralElementOperator,
    args...,
) = SpectralBroadcasted{SpectralStyle}(op, args)

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
    # If we've already instantiated, then we need to strip the type parameters,
    # for example, `Divergence{()}(axes)`.
    op = unionall_type(typeof(op)){()}(axes)
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
        op = unionall_type(typeof(bc.f)){()}(axes)
        RT = operator_return_eltype(op, map(eltype, args)...)
        return Broadcast.broadcasted(Returns(zero(RT)), Fields.coordinate_field(axes))
    end

    if bc.f isa SpectralElementOperator
        op = unionall_type(typeof(bc.f)){()}(axes)
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
    },
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
 - call `apply_operator`, returning the resulting "pseudo Field":  a `Field` with an
 `IF`/`IJF` data object.
- if `bc` is a `Field`, return that
"""
Base.@propagate_inbounds function resolve_operator(
    bc::SpectralBroadcasted{SlabBlockSpectralStyle},
    slabidx,
)
    args = _resolve_operator_args(slabidx, bc.args)
    apply_operator(bc.op, bc.axes, slabidx, args...)
end
Base.@propagate_inbounds function resolve_operator(
    bc::Base.Broadcast.Broadcasted{SlabBlockSpectralStyle},
    slabidx,
)
    args = _resolve_operator_args(slabidx, bc.args)
    Base.Broadcast.Broadcasted{SlabBlockSpectralStyle}(bc.f, args, bc.axes)
end
@inline resolve_operator(x, slabidx) = x

"""
    _resolve_operator_args(slabidx, args)

Calls `resolve_operator(arg, slabidx)` for each `arg` in `args`
"""
Base.@propagate_inbounds _resolve_operator_args(slabidx, args) =
    unrolled_map(arg -> resolve_operator(arg, slabidx), args)

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

Base.@propagate_inbounds _get_node(space, ij, slabidx, args::Tuple) =
    unrolled_map(arg -> get_node(space, arg, ij, slabidx), args)

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
Base.@propagate_inbounds function get_node(
    parent_space,
    field::Fields.Field,
    ij::CartesianIndex{1},
    slabidx,
)
    space = reconstruct_placeholder_space(axes(field), parent_space)
    i, = Tuple(ij)
    if space isa Spaces.FaceExtrudedFiniteDifferenceSpace ||
       space isa Spaces.FaceFiniteDifferenceSpace
        _v = slabidx.v + half
    elseif space isa Spaces.CenterExtrudedFiniteDifferenceSpace ||
           space isa Spaces.AbstractSpectralElementSpace ||
           space isa Spaces.CenterFiniteDifferenceSpace
        _v = slabidx.v
    else
        error("invalid space")
    end
    h = slabidx.h
    fv = Fields.field_values(field)
    v = isnothing(_v) ? 1 : _v
    return fv[CartesianIndex(i, 1, 1, v, h)]
end
Base.@propagate_inbounds function get_node(
    parent_space,
    field::Fields.Field,
    ij::CartesianIndex{2},
    slabidx,
)
    space = reconstruct_placeholder_space(axes(field), parent_space)
    i, j = Tuple(ij)
    if space isa Spaces.FaceExtrudedFiniteDifferenceSpace
        _v = slabidx.v + half
    elseif space isa Spaces.CenterExtrudedFiniteDifferenceSpace ||
           space isa Spaces.AbstractSpectralElementSpace
        _v = slabidx.v
    else
        error("invalid space")
    end
    h = slabidx.h
    fv = Fields.field_values(field)
    v = isnothing(_v) ? 1 : _v
    return fv[CartesianIndex(i, j, 1, v, h)]
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
Base.@propagate_inbounds function get_node(
    space,
    data::Union{DataLayouts.IJF, DataLayouts.IF},
    ij,
    slabidx,
)
    data[ij]
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
    ij::CartesianIndex{1},
    slabidx,
)
    i, = Tuple(ij)
    h = slabidx.h
    if space isa Spaces.FaceExtrudedFiniteDifferenceSpace
        _v = slabidx.v + half
    else
        _v = slabidx.v
    end
    lgd = Spaces.local_geometry_data(space)
    v = isnothing(_v) ? 1 : _v
    return lgd[CartesianIndex(i, 1, 1, v, h)]
end
Base.@propagate_inbounds function get_local_geometry(
    space::Union{
        Spaces.AbstractSpectralElementSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    ij::CartesianIndex{2},
    slabidx,
)
    i, j = Tuple(ij)
    h = slabidx.h
    if space isa Spaces.FaceExtrudedFiniteDifferenceSpace
        _v = slabidx.v + half
    else
        _v = slabidx.v
    end
    v = isnothing(_v) ? 1 : _v
    lgd = Spaces.local_geometry_data(space)
    return lgd[CartesianIndex(i, j, 1, v, h)]
end

Base.@propagate_inbounds function set_node!(
    space,
    field::Fields.Field,
    ij::CartesianIndex{1},
    slabidx,
    val,
)
    i, = Tuple(ij)
    if space isa Spaces.FaceExtrudedFiniteDifferenceSpace ||
       space isa Spaces.FaceFiniteDifferenceSpace
        _v = slabidx.v + half
    else
        _v = slabidx.v
    end
    h = slabidx.h
    fv = Fields.field_values(field)
    v = isnothing(_v) ? 1 : _v
    fv[CartesianIndex(i, 1, 1, v, h)] = val
end
Base.@propagate_inbounds function set_node!(
    space,
    field::Fields.Field,
    ij::CartesianIndex{2},
    slabidx,
    val,
)
    i, j = Tuple(ij)
    if space isa Spaces.FaceExtrudedFiniteDifferenceSpace
        _v = slabidx.v + half
    else
        _v = slabidx.v
    end
    h = slabidx.h
    fv = Fields.field_values(field)
    v = isnothing(_v) ? 1 : _v
    fv[CartesianIndex(i, j, 1, v, h)] = val
end

Base.Broadcast.BroadcastStyle(
    ::Type{<:SpectralBroadcasted{Style}},
) where {Style} = Style()

Base.Broadcast.BroadcastStyle(
    style::AbstractSpectralStyle,
    ::Fields.AbstractFieldStyle,
) = style






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
struct Divergence{I} <: SpectralElementOperator{I} end
Divergence() = Divergence{()}()
Divergence{()}(space) = Divergence{operator_axes(space)}()

operator_return_eltype(op::Divergence{I}, ::Type{S}) where {I, S} =
    Geometry.divergence_result_type(S)

function apply_operator(op::Divergence{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = IF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(space, arg, ij, slabidx)
        Jv¹ = local_geometry.J * Geometry.contravariant1(v, local_geometry)
        for ii in 1:Nq
            out[slab_index(ii)] += D[ii, i] * Jv¹
        end
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i)] *= local_geometry.invJ
    end
    return Field(SArray(out), space)
end

Base.@propagate_inbounds function apply_operator(
    op::Divergence{(1, 2)},
    space,
    slabidx,
    arg,
)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IJF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(space, arg, ij, slabidx)
        Jv¹ = local_geometry.J * Geometry.contravariant1(v, local_geometry)
        for ii in 1:Nq
            out[slab_index(ii, j)] += D[ii, i] * Jv¹
        end
        Jv² = local_geometry.J * Geometry.contravariant2(v, local_geometry)
        for jj in 1:Nq
            out[slab_index(i, jj)] += D[jj, j] * Jv²
        end
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i, j)] *= local_geometry.invJ
    end
    return Field(SArray(out), space)
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

function apply_operator(op::SplitDivergence{(1,)}, space, slabidx, arg1, arg2)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    JT = operator_return_eltype(op, eltype(arg1), FT)
    RT = operator_return_eltype(op, eltype(arg1), eltype(arg2))

    Ju1 = IF{JT, Nq}(MArray, FT)
    psi = IF{eltype(arg2), Nq}(MArray, eltype(arg2))
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        u = get_node(space, arg1, ij, slabidx)
        Ju1[slab_index(i)] =
            local_geometry.J * Geometry.contravariant1(u, local_geometry)
        psi[slab_index(i)] = get_node(space, arg2, ij, slabidx)
    end

    out = IF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))
    @inbounds for i in 1:Nq
        for j in 1:(i - 1) # loop over half the indices, since F1[i,j] = F1[j,i]
            F1 =
                (
                    (Ju1[slab_index(i)] + Ju1[slab_index(j)]) *
                    (psi[slab_index(i)] + psi[slab_index(j)])
                ) / 2
            out[slab_index(i)] += D[i, j] * F1
            out[slab_index(j)] += D[j, i] * F1
        end
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i)] *= local_geometry.invJ
    end

    return Field(SArray(out), space)
end

function apply_operator(op::SplitDivergence{(1, 2)}, space, slabidx, arg1, arg2)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    JT = operator_return_eltype(op, eltype(arg1), FT)
    RT = operator_return_eltype(op, eltype(arg1), eltype(arg2))

    Ju1 = DataLayouts.IJF{JT, Nq}(MArray, FT)
    Ju2 = DataLayouts.IJF{JT, Nq}(MArray, FT)
    psi = DataLayouts.IJF{eltype(arg2), Nq}(MArray, eltype(arg2))
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        u = get_node(space, arg1, ij, slabidx)
        Ju1[slab_index(i, j)] =
            local_geometry.J * Geometry.contravariant1(u, local_geometry)
        Ju2[slab_index(i, j)] =
            local_geometry.J * Geometry.contravariant2(u, local_geometry)
        psi[slab_index(i, j)] = get_node(space, arg2, ij, slabidx)
    end

    out = DataLayouts.IJF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))
    @inbounds for j in 1:Nq, i in 1:Nq
        for k in 1:(i - 1) # loop over half the indices, since F1[i,k] = F1[k,i]
            F1 =
                (
                    (Ju1[slab_index(i, j)] + Ju1[slab_index(k, j)]) *
                    (psi[slab_index(i, j)] + psi[slab_index(k, j)])
                ) / 2
            out[slab_index(i, j)] += D[i, k] * F1
            out[slab_index(k, j)] += D[k, i] * F1
        end
        for k in 1:(j - 1) # loop over half the indices, since F2[j,k] = F2[k,j]
            F2 =
                (
                    (Ju2[slab_index(i, j)] + Ju2[slab_index(i, k)]) *
                    (psi[slab_index(i, j)] + psi[slab_index(i, k)])
                ) / 2
            out[slab_index(i, j)] += D[j, k] * F2
            out[slab_index(i, k)] += D[k, j] * F2
        end
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i, j)] *= local_geometry.invJ
    end

    return Field(SArray(out), space)
end

"""
    wdiv = WeakDivergence()
    wdiv.(u)

Computes the "weak divergence" of a vector field `u`.

This is defined as the scalar field ``\\theta \\in \\mathcal{V}_0`` such that
for all ``\\phi\\in \\mathcal{V}_0``
```math
\\int_\\Omega \\phi \\theta \\, d \\Omega
=
- \\int_\\Omega (\\nabla \\phi) \\cdot u \\,d \\Omega
```
where ``\\mathcal{V}_0`` is the space of ``u``.

This arises as the contribution of the volume integral after applying
integration by parts to the weak form expression of the divergence
```math
\\int_\\Omega \\phi (\\nabla \\cdot u) \\, d \\Omega
=
- \\int_\\Omega (\\nabla \\phi) \\cdot u \\,d \\Omega
+ \\oint_{\\partial \\Omega} \\phi (u \\cdot n) \\,d \\sigma
```

It can be written in matrix form as
```math
ϕ^\\top WJ θ = - \\sum_i (D_i ϕ)^\\top WJ u^i
```
which reduces to
```math
θ = -(WJ)^{-1} \\sum_i D_i^\\top WJ u^i
```
where
 - ``J`` is the diagonal Jacobian matrix
 - ``W`` is the diagonal matrix of quadrature weights
 - ``D_i`` is the derivative matrix along the ``i``th dimension
"""
struct WeakDivergence{I} <: SpectralElementOperator{I} end
WeakDivergence() = WeakDivergence{()}()
WeakDivergence{()}(space) = WeakDivergence{operator_axes(space)}()

operator_return_eltype(::WeakDivergence{I}, ::Type{S}) where {I, S} =
    Geometry.divergence_result_type(S)

function apply_operator(op::WeakDivergence{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(space, arg, ij, slabidx)
        WJv¹ = local_geometry.WJ * Geometry.contravariant1(v, local_geometry)
        for ii in 1:Nq
            out[slab_index(ii)] += D[i, ii] * WJv¹
        end
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i)] /= -local_geometry.WJ
    end
    return Field(SArray(out), space)
end

function apply_operator(op::WeakDivergence{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IJF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))

    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        v = get_node(space, arg, ij, slabidx)
        WJv¹ = local_geometry.WJ * Geometry.contravariant1(v, local_geometry)
        for ii in 1:Nq
            out[slab_index(ii, j)] += D[i, ii] * WJv¹
        end
        WJv² = local_geometry.WJ * Geometry.contravariant2(v, local_geometry)
        for jj in 1:Nq
            out[slab_index(i, jj)] += D[j, jj] * WJv²
        end
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i, j)] /= -local_geometry.WJ
    end
    return Field(SArray(out), space)
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
struct Gradient{I} <: SpectralElementOperator{I} end
Gradient() = Gradient{()}()
Gradient{()}(space) = Gradient{operator_axes(space)}()

operator_return_eltype(::Gradient{I}, ::Type{S}) where {I, S} =
    Geometry.gradient_result_type(Val(I), S)

function apply_operator(op::Gradient{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        x = get_node(space, arg, ij, slabidx)
        for ii in 1:Nq
            ∂f∂ξ = Geometry.Covariant1Vector(D[ii, i]) ⊗ x
            out[slab_index(ii)] += ∂f∂ξ
        end
    end
    return Field(SArray(out), space)
end

Base.@propagate_inbounds function apply_operator(
    op::Gradient{(1, 2)},
    space,
    slabidx,
    arg,
)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IJF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))

    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        x = get_node(space, arg, ij, slabidx)
        for ii in 1:Nq
            ∂f∂ξ₁ = Geometry.Covariant12Vector(D[ii, i], zero(eltype(D))) ⊗ x
            out[slab_index(ii, j)] += ∂f∂ξ₁
        end
        for jj in 1:Nq
            ∂f∂ξ₂ = Geometry.Covariant12Vector(zero(eltype(D)), D[jj, j]) ⊗ x
            out[slab_index(i, jj)] += ∂f∂ξ₂
        end
    end
    return Field(SArray(out), space)
end

"""
    wgrad = WeakGradient()
    wgrad.(f)

Compute the "weak gradient" of `f` on each element.

This is defined as the the vector field ``\\theta \\in \\mathcal{V}_0`` such
that for all ``\\phi \\in \\mathcal{V}_0``
```math
\\int_\\Omega \\phi \\cdot \\theta \\, d \\Omega
=
- \\int_\\Omega (\\nabla \\cdot \\phi) f \\, d\\Omega
```
where ``\\mathcal{V}_0`` is the space of ``f``.

This arises from the contribution of the volume integral after by applying
integration by parts to the weak form expression of the gradient
```math
\\int_\\Omega \\phi \\cdot (\\nabla f) \\, d \\Omega
=
- \\int_\\Omega f (\\nabla \\cdot \\phi) \\, d\\Omega
+ \\oint_{\\partial \\Omega} f (\\phi \\cdot n) \\, d \\sigma
```

In matrix form, this becomes
```math
{\\phi^i}^\\top W J \\theta_i = - ( J^{-1} D_i J \\phi^i )^\\top W J f
```
which reduces to
```math
\\theta_i = -W^{-1} D_i^\\top W f
```
where ``D_i`` is the derivative matrix along the ``i``th dimension.
"""
struct WeakGradient{I} <: SpectralElementOperator{I} end
WeakGradient() = WeakGradient{()}()
WeakGradient{()}(space) = WeakGradient{operator_axes(space)}()

operator_return_eltype(::WeakGradient{I}, ::Type{S}) where {I, S} =
    Geometry.gradient_result_type(Val(I), S)

function apply_operator(op::WeakGradient{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        W = local_geometry.WJ * local_geometry.invJ
        Wx = W * get_node(space, arg, ij, slabidx)
        for ii in 1:Nq
            Dᵀ₁Wf = Geometry.Covariant1Vector(D[i, ii]) ⊗ Wx
            out[slab_index(ii)] -= Dᵀ₁Wf
        end
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        W = local_geometry.WJ * local_geometry.invJ
        out[slab_index(i)] /= W
    end
    return Field(SArray(out), space)
end

function apply_operator(op::WeakGradient{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IJF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))

    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        W = local_geometry.WJ * local_geometry.invJ
        Wx = W * get_node(space, arg, ij, slabidx)
        for ii in 1:Nq
            Dᵀ₁Wf = Geometry.Covariant12Vector(D[i, ii], zero(eltype(D))) ⊗ Wx
            out[slab_index(ii, j)] -= Dᵀ₁Wf
        end
        for jj in 1:Nq
            Dᵀ₂Wf = Geometry.Covariant12Vector(zero(eltype(D)), D[j, jj]) ⊗ Wx
            out[slab_index(i, jj)] -= Dᵀ₂Wf
        end
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        W = local_geometry.WJ * local_geometry.invJ
        out[slab_index(i, j)] /= W
    end
    return Field(SArray(out), space)
end

abstract type CurlSpectralElementOperator{I} <: SpectralElementOperator{I} end

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
struct Curl{I} <: CurlSpectralElementOperator{I} end
Curl() = Curl{()}()
Curl{()}(space) = Curl{operator_axes(space)}()

operator_return_eltype(::Curl{I}, ::Type{S}) where {I, S} =
    Geometry.curl_result_type(Val(I), S)

function apply_operator(op::Curl{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))
    if RT <: Geometry.Contravariant2Vector
        @inbounds for i in 1:Nq
            ij = CartesianIndex((i,))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            v₃ = Geometry.covariant3(v, local_geometry)
            for ii in 1:Nq
                D₁v₃ = D[ii, i] * v₃
                out[slab_index(ii)] += Geometry.Contravariant2Vector(-D₁v₃)
            end
        end
    elseif RT <: Geometry.Contravariant3Vector
        @inbounds for i in 1:Nq
            ij = CartesianIndex((i,))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            v₂ = Geometry.covariant2(v, local_geometry)
            for ii in 1:Nq
                D₁v₂ = D[ii, i] * v₂
                out[slab_index(ii)] += Geometry.Contravariant3Vector(D₁v₂)
            end
        end
    elseif RT <: Geometry.Contravariant23Vector
        @inbounds for i in 1:Nq
            ij = CartesianIndex((i,))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            v₂ = Geometry.covariant2(v, local_geometry)
            v₃ = Geometry.covariant3(v, local_geometry)
            for ii in 1:Nq
                D₁v₃ = D[ii, i] * v₃
                D₁v₂ = D[ii, i] * v₂
                out[slab_index(ii)] +=
                    Geometry.Contravariant23Vector(-D₁v₃, D₁v₂)
            end
        end
    else
        error("invalid return type: $RT")
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i)] *= local_geometry.invJ
    end
    return Field(SArray(out), space)
end

function apply_operator(op::Curl{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IJF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))

    # input data is a Covariant12Vector field
    if RT <: Geometry.Contravariant3Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i, j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            v₁ = Geometry.covariant1(v, local_geometry)
            for jj in 1:Nq
                D₂v₁ = D[jj, j] * v₁
                out[slab_index(i, jj)] += Geometry.Contravariant3Vector(-D₂v₁)
            end
            v₂ = Geometry.covariant2(v, local_geometry)
            for ii in 1:Nq
                D₁v₂ = D[ii, i] * v₂
                out[slab_index(ii, j)] += Geometry.Contravariant3Vector(D₁v₂)
            end
        end
        # input data is a Covariant3Vector field
    elseif RT <: Geometry.Contravariant12Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i, j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            v₃ = Geometry.covariant3(v, local_geometry)
            for ii in 1:Nq
                D₁v₃ = D[ii, i] * v₃
                out[slab_index(ii, j)] +=
                    Geometry.Contravariant12Vector(zero(D₁v₃), -D₁v₃)
            end
            for jj in 1:Nq
                D₂v₃ = D[jj, j] * v₃
                out[slab_index(i, jj)] +=
                    Geometry.Contravariant12Vector(D₂v₃, zero(D₂v₃))
            end
        end
    elseif RT <: Geometry.Contravariant123Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i, j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            v₁ = Geometry.covariant1(v, local_geometry)
            v₂ = Geometry.covariant2(v, local_geometry)
            v₃ = Geometry.covariant3(v, local_geometry)
            for ii in 1:Nq
                D₁v₃ = D[ii, i] * v₃
                D₁v₂ = D[ii, i] * v₂
                out[slab_index(ii, j)] +=
                    Geometry.Contravariant123Vector(zero(D₁v₃), -D₁v₃, D₁v₂)
            end
            for jj in 1:Nq
                D₂v₃ = D[jj, j] * v₃
                D₂v₁ = D[jj, j] * v₁
                out[slab_index(i, jj)] +=
                    Geometry.Contravariant123Vector(D₂v₃, zero(D₂v₃), -D₂v₁)
            end
        end
    else
        error("invalid return type")
    end
    @inbounds for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i, j)] *= local_geometry.invJ
    end
    return Field(SArray(out), space)
end

"""
    wcurl = WeakCurl()
    wcurl.(u)

Computes the "weak curl" on each element of a covariant vector field `u`.

Note: The vector field ``u`` needs to be excliclty converted to a `CovaraintVector`,
as then the `WeakCurl` is independent of the local metric tensor.

This is defined as the vector field ``\\theta \\in \\mathcal{V}_0`` such that
for all ``\\phi \\in \\mathcal{V}_0``
```math
\\int_\\Omega \\phi \\cdot \\theta \\, d \\Omega
=
\\int_\\Omega (\\nabla \\times \\phi) \\cdot u \\,d \\Omega
```
where ``\\mathcal{V}_0`` is the space of ``f``.

This arises from the contribution of the volume integral after by applying
integration by parts to the weak form expression of the curl
```math
\\int_\\Omega \\phi \\cdot (\\nabla \\times u) \\,d\\Omega
=
\\int_\\Omega (\\nabla \\times \\phi) \\cdot u \\,d \\Omega
- \\oint_{\\partial \\Omega} (\\phi \\times u) \\cdot n \\,d\\sigma
```

In matrix form, this becomes
```math
{\\phi_i}^\\top W J \\theta^i = (J^{-1} \\epsilon^{kji} D_j \\phi_i)^\\top W J u_k
```
which, by using the anti-symmetry of the Levi-Civita symbol, reduces to
```math
\\theta^i = - \\epsilon^{ijk} (WJ)^{-1} D_j^\\top W u_k
```
"""
struct WeakCurl{I} <: CurlSpectralElementOperator{I} end
WeakCurl() = WeakCurl{()}()
WeakCurl{()}(space) = WeakCurl{operator_axes(space)}()

operator_return_eltype(::WeakCurl{I}, ::Type{S}) where {I, S} =
    Geometry.curl_result_type(Val(I), S)

function apply_operator(op::WeakCurl{(1,)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))
    # input data is a Covariant3Vector field
    if RT <: Geometry.Contravariant2Vector
        @inbounds for i in 1:Nq
            ij = CartesianIndex((i,))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            W = local_geometry.WJ * local_geometry.invJ
            Wv₃ = W * Geometry.covariant3(v, local_geometry)
            for ii in 1:Nq
                Dᵀ₁Wv₃ = D[i, ii] * Wv₃
                out[slab_index(ii)] += Geometry.Contravariant2Vector(Dᵀ₁Wv₃)
            end
        end
    elseif RT <: Geometry.Contravariant3Vector
        @inbounds for i in 1:Nq
            ij = CartesianIndex((i,))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            W = local_geometry.WJ * local_geometry.invJ
            Wv₂ = W * Geometry.covariant2(v, local_geometry)
            for ii in 1:Nq
                Dᵀ₁Wv₂ = D[i, ii] * Wv₂
                out[slab_index(ii)] += Geometry.Contravariant3Vector(-Dᵀ₁Wv₂)
            end
        end
    elseif RT <: Geometry.Contravariant23Vector
        @inbounds for i in 1:Nq
            ij = CartesianIndex((i,))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            W = local_geometry.WJ * local_geometry.invJ
            Wv₃ = W * Geometry.covariant3(v, local_geometry)
            Wv₂ = W * Geometry.covariant2(v, local_geometry)
            for ii in 1:Nq
                Dᵀ₁Wv₃ = D[i, ii] * Wv₃
                Dᵀ₁Wv₂ = D[i, ii] * Wv₂
                out[slab_index(ii)] +=
                    Geometry.Contravariant23Vector(Dᵀ₁Wv₃, -Dᵀ₁Wv₂)
            end
        end
    else
        error("invalid return type: $RT")
    end
    @inbounds for i in 1:Nq
        ij = CartesianIndex((i,))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i)] /= local_geometry.WJ
    end
    return Field(SArray(out), space)
end

function apply_operator(op::WeakCurl{(1, 2)}, space, slabidx, arg)
    FT = Spaces.undertype(space)
    QS = Spaces.quadrature_style(space)
    Nq = Quadratures.degrees_of_freedom(QS)
    D = Quadratures.differentiation_matrix(FT, QS)
    # allocate temp output
    RT = operator_return_eltype(op, eltype(arg))
    out = DataLayouts.IJF{RT, Nq}(MArray, FT)
    fill!(parent(out), zero(FT))

    # input data is a Covariant12Vector field
    if RT <: Geometry.Contravariant3Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i, j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            W = local_geometry.WJ * local_geometry.invJ
            Wv₁ = W * Geometry.covariant1(v, local_geometry)
            for jj in 1:Nq
                Dᵀ₂Wv₁ = D[j, jj] * Wv₁
                out[slab_index(i, jj)] +=
                    Geometry.Contravariant3Vector(Dᵀ₂Wv₁)
            end
            Wv₂ = W * Geometry.covariant2(v, local_geometry)
            for ii in 1:Nq
                Dᵀ₁Wv₂ = D[i, ii] * Wv₂
                out[slab_index(ii, j)] +=
                    Geometry.Contravariant3Vector(-Dᵀ₁Wv₂)
            end
        end
        # input data is a Covariant3Vector field
    elseif RT <: Geometry.Contravariant12Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i, j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            W = local_geometry.WJ * local_geometry.invJ
            Wv₃ = W * Geometry.covariant3(v, local_geometry)
            for ii in 1:Nq
                Dᵀ₁Wv₃ = D[i, ii] * Wv₃
                out[slab_index(ii, j)] +=
                    Geometry.Contravariant12Vector(zero(Dᵀ₁Wv₃), Dᵀ₁Wv₃)
            end
            for jj in 1:Nq
                Dᵀ₂Wv₃ = D[j, jj] * Wv₃
                out[slab_index(i, jj)] +=
                    Geometry.Contravariant12Vector(-Dᵀ₂Wv₃, zero(Dᵀ₂Wv₃))
            end
        end
    elseif RT <: Geometry.Contravariant123Vector
        @inbounds for j in 1:Nq, i in 1:Nq
            ij = CartesianIndex((i, j))
            local_geometry = get_local_geometry(space, ij, slabidx)
            v = get_node(space, arg, ij, slabidx)
            W = local_geometry.WJ * local_geometry.invJ
            Wv₁ = W * Geometry.covariant1(v, local_geometry)
            Wv₂ = W * Geometry.covariant2(v, local_geometry)
            Wv₃ = W * Geometry.covariant3(v, local_geometry)
            for ii in 1:Nq
                Dᵀ₁Wv₃ = D[i, ii] * Wv₃
                Dᵀ₁Wv₂ = D[i, ii] * Wv₂
                out[slab_index(ii, j)] += Geometry.Contravariant123Vector(
                    zero(Dᵀ₁Wv₃),
                    Dᵀ₁Wv₃,
                    -Dᵀ₁Wv₂,
                )
            end
            for jj in 1:Nq
                Dᵀ₂Wv₃ = D[j, jj] * Wv₃
                Dᵀ₂Wv₁ = D[j, jj] * Wv₁
                out[slab_index(i, jj)] +=
                    Geometry.Contravariant123Vector(
                        -Dᵀ₂Wv₃,
                        zero(Dᵀ₂Wv₃),
                        Dᵀ₂Wv₁,
                    )
            end
        end
    else
        error("invalid return type")
    end
    for j in 1:Nq, i in 1:Nq
        ij = CartesianIndex((i, j))
        local_geometry = get_local_geometry(space, ij, slabidx)
        out[slab_index(i, j)] /= local_geometry.WJ
    end
    return Field(SArray(out), space)
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

function apply_operator(op::Interpolate{(1,)}, space_out, slabidx, arg)
    FT = Spaces.undertype(space_out)
    space_in = axes(arg)
    QS_in = Spaces.quadrature_style(space_in)
    QS_out = Spaces.quadrature_style(space_out)
    Nq_in = Quadratures.degrees_of_freedom(QS_in)
    Nq_out = Quadratures.degrees_of_freedom(QS_out)
    Imat = Quadratures.interpolation_matrix(FT, QS_out, QS_in)
    RT = eltype(arg)
    out = DataLayouts.IF{RT, Nq_out}(MArray, FT)
    @inbounds for i in 1:Nq_out
        # manually inlined rmatmul with slab_getnode
        ij = CartesianIndex((1,))
        r = Imat[i, 1] * get_node(space_in, arg, ij, slabidx)
        for ii in 2:Nq_in
            ij = CartesianIndex((ii,))
            r = muladd(
                Imat[i, ii],
                get_node(space_in, arg, ij, slabidx),
                r,
            )
        end
        out[slab_index(i)] = r
    end
    return Field(SArray(out), space_out)
end

function apply_operator(op::Interpolate{(1, 2)}, space_out, slabidx, arg)
    FT = Spaces.undertype(space_out)
    space_in = axes(arg)
    QS_in = Spaces.quadrature_style(space_in)
    QS_out = Spaces.quadrature_style(space_out)
    Nq_in = Quadratures.degrees_of_freedom(QS_in)
    Nq_out = Quadratures.degrees_of_freedom(QS_out)
    Imat = Quadratures.interpolation_matrix(FT, QS_out, QS_in)
    RT = eltype(arg)
    # temporary storage
    temp = DataLayouts.IJF{RT, max(Nq_in, Nq_out)}(MArray, FT)
    out = DataLayouts.IJF{RT, Nq_out}(MArray, FT)
    @inbounds for j in 1:Nq_in, i in 1:Nq_out
        # manually inlined rmatmul1 with slab get_node
        # we do this to remove one allocated intermediate array
        ij = CartesianIndex((1, j))
        r = Imat[i, 1] * get_node(space_in, arg, ij, slabidx)
        for ii in 2:Nq_in
            ij = CartesianIndex((ii, j))
            r = muladd(
                Imat[i, ii],
                get_node(space_in, arg, ij, slabidx),
                r,
            )
        end
        temp[slab_index(i, j)] = r
    end
    @inbounds for j in 1:Nq_out, i in 1:Nq_out
        out[slab_index(i, j)] = rmatmul2(Imat, temp, i, j)
    end
    return Field(SArray(out), space_out)
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
struct Restrict{I, S} <: TensorOperator{I}
    space::S
end
Restrict(space) = Restrict{operator_axes(space), typeof(space)}(space)
Restrict{()}(space) = Restrict{operator_axes(space), typeof(space)}(space)

function apply_operator(op::Restrict{(1,)}, space_out, slabidx, arg)
    FT = Spaces.undertype(space_out)
    space_in = axes(arg)
    QS_in = Spaces.quadrature_style(space_in)
    QS_out = Spaces.quadrature_style(space_out)
    Nq_in = Quadratures.degrees_of_freedom(QS_in)
    Nq_out = Quadratures.degrees_of_freedom(QS_out)
    ImatT = Quadratures.interpolation_matrix(FT, QS_in, QS_out)' # transpose
    RT = eltype(arg)
    out = IF{RT, Nq_out}(MArray, FT)
    @inbounds for i in 1:Nq_out
        # manually inlined rmatmul with slab get_node
        ij = CartesianIndex((1,))
        WJ = get_local_geometry(space_in, ij, slabidx).WJ
        r = ImatT[i, 1] * (WJ * get_node(space_in, arg, ij, slabidx))
        for ii in 2:Nq_in
            ij = CartesianIndex((ii,))
            WJ = get_local_geometry(space_in, ij, slabidx).WJ
            r = muladd(
                ImatT[i, ii],
                WJ * get_node(space_in, arg, ij, slabidx),
                r,
            )
        end
        ij_out = CartesianIndex((i,))
        WJ_out = get_local_geometry(space_out, ij_out, slabidx).WJ
        out[slab_index(i)] = r / WJ_out
    end
    return Field(SArray(out), space_out)
end

function apply_operator(op::Restrict{(1, 2)}, space_out, slabidx, arg)
    FT = Spaces.undertype(space_out)
    space_in = axes(arg)
    QS_in = Spaces.quadrature_style(space_in)
    QS_out = Spaces.quadrature_style(space_out)
    Nq_in = Quadratures.degrees_of_freedom(QS_in)
    Nq_out = Quadratures.degrees_of_freedom(QS_out)
    ImatT = Quadratures.interpolation_matrix(FT, QS_in, QS_out)' # transpose
    RT = eltype(arg)
    # temporary storage
    temp = DataLayouts.IJF{RT, max(Nq_in, Nq_out)}(MArray, FT)
    out = DataLayouts.IJF{RT, Nq_out}(MArray, FT)
    @inbounds for j in 1:Nq_in, i in 1:Nq_out
        # manually inlined rmatmul1 with slab get_node
        ij = CartesianIndex((1, j))
        WJ = get_local_geometry(space_in, ij, slabidx).WJ
        r = ImatT[i, 1] * (WJ * get_node(space_in, arg, ij, slabidx))
        for ii in 2:Nq_in
            ij = CartesianIndex((ii, j))
            WJ = get_local_geometry(space_in, ij, slabidx).WJ
            r = muladd(
                ImatT[i, ii],
                WJ * get_node(space_in, arg, ij, slabidx),
                r,
            )
        end
        temp[slab_index(i, j)] = r
    end
    @inbounds for j in 1:Nq_out, i in 1:Nq_out
        ij_out = CartesianIndex((i, j))
        WJ_out = get_local_geometry(space_out, ij_out, slabidx).WJ
        out[slab_index(i, j)] = rmatmul2(ImatT, temp, i, j) / WJ_out
    end
    return Field(SArray(out), space_out)
end


"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element.
"""
function tensor_product! end

function tensor_product!(
    out::DataLayouts.Data1DX{S, Nv, Ni_out},
    indata::DataLayouts.Data1DX{S, Nv, Ni_in},
    M::SMatrix{Ni_out, Ni_in},
) where {S, Nv, Ni_out, Ni_in}
    (_, _, _, _, Nh_in) = size(indata)
    (_, _, _, _, Nh_out) = size(out)
    # TODO: assumes the same number of levels (horizontal only)
    @assert Nh_in == Nh_out
    @inbounds for h in 1:Nh_out, v in 1:Nv
        in_slab = slab(indata, v, h)
        out_slab = slab(out, v, h)
        for i in 1:Ni_out
            r = M[i, 1] * in_slab[slab_index(1)]
            for ii in 2:Ni_in
                r = muladd(M[i, ii], in_slab[slab_index(ii)], r)
            end
            out_slab[slab_index(i)] = r
        end
    end
    return out
end

function tensor_product!(
    out::DataLayouts.Data2D{S, Nij_out},
    indata::DataLayouts.Data2D{S, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}

    Nh = length(indata)
    @assert Nh == length(out)

    # temporary storage
    temp = MArray{Tuple{Nij_out, Nij_in}, S, 2, Nij_out * Nij_in}(undef)

    @inbounds for h in 1:Nh
        in_slab = slab(indata, h)
        out_slab = slab(out, h)
        for j in 1:Nij_in, i in 1:Nij_out
            temp[slab_index(i, j)] = rmatmul1(M, in_slab, i, j)
        end
        for j in 1:Nij_out, i in 1:Nij_out
            out_slab[slab_index(i, j)] = rmatmul2(M, temp, i, j)
        end
    end
    return out
end

function tensor_product!(
    out_slab::DataLayouts.DataSlab2D{S, Nij_out},
    in_slab::DataLayouts.DataSlab2D{S, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}
    # temporary storage
    temp = MArray{Tuple{Nij_out, Nij_in}, S, 2, Nij_out * Nij_in}(undef)
    @inbounds for j in 1:Nij_in, i in 1:Nij_out
        temp[slab_index(i, j)] = rmatmul1(M, in_slab, i, j)
    end
    @inbounds for j in 1:Nij_out, i in 1:Nij_out
        out_slab[slab_index(i, j)] = rmatmul2(M, temp, i, j)
    end
    return out_slab
end

function tensor_product!(
    inout::Data2D{S, Nij},
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
        DataLayouts.IH1JH2{S, Nu}(Matrix{S}(undef, (Nu * n1, Nu * n2)))
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
    interp_data = DataLayouts.IV1JH2{S, nl, Nu}(Matrix{S}(undef, (nl, Nu * n1)))
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

import .DataLayouts: slab_index
import .Spaces: slab_type

"""
    rmatmul1(W, S, i, j)

Recursive matrix product along the 1st dimension of `S`. Equivalent to:

    mapreduce(*, +, W[i,:], S[:,j])

"""
function rmatmul1(W, S, i, j)
    Nq = size(W, 2)
    @inbounds r = W[i, 1] * S[slab_index(1, j)]
    @inbounds for ii in 2:Nq
        r = muladd(W[i, ii], S[slab_index(ii, j)], r)
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
    @inbounds r = W[j, 1] * S[slab_index(i, 1)]
    @inbounds for jj in 2:Nq
        r = muladd(W[j, jj], S[slab_index(i, jj)], r)
    end
    return r
end

function apply_operator(
    op::SpectralElementOperator{()},
    space,
    _,
    arg,
)
    RT = operator_return_eltype(op, eltype(arg))
    return map(Returns(zero(RT)), space)
end
