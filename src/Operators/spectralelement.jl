





abstract type AbstractSpectralStyle <: Fields.AbstractFieldStyle end

# the .f field is an operator
struct SpectralStyle <: AbstractSpectralStyle end

# the .f field is a function, but one of the args (or their args) is a StencilStyle
struct CompositeSpectralStyle <: AbstractSpectralStyle end

# f.(args...)
# if f isa SpectralElementOperator, broadcasted returns a SpectralOperatorBroadcasted object
# otherwise, if arg is a SpectralOperatorBroadcasted, broadcasted returns a Broadcasted{CompositeSpectralStyle} object

"""
    SpectralElementOperator

Represents an operation that is applied to each element.

Subtypes `Op` of this should define the following:
- [`operator_return_eltype(::Op, ElTypes...)`](@ref)
- [`allocate_work(::Op, args...)`](@ref)
- [`apply_slab(::Op, work, args...)`](@ref)

Additionally, the result type `OpResult <: OperatorSlabResult` of `apply_slab` should define `get_node(::OpResult, i, j)`.
"""
abstract type SpectralElementOperator end

"""
    SpectralBroadcasted{Style}(op, args[,axes[, work]])

This is similar to a `Base.Broadcast.Broadcasted` object, except it contains space for an intermediate `work` storage.

This is returned by `Base.Broadcast.broadcasted(op::SpectralElementOperator)`.
"""
struct SpectralBroadcasted{Style, Op, Args, Axes, Work} <: Base.AbstractBroadcasted
    op::Op
    args::Args
    axes::Axes
    work::Work
end
SpectralBroadcasted{Style}(op::Op, args::Args, axes::Axes=nothing, work::Work=nothing) where {Style, Op, Args, Axes, Work} =
    SpectralBroadcasted{Style, Op, Args, Axes, Work}(op, args, axes, work)

Base.axes(sbc::SpectralBroadcasted) =
    isnothing(sbc.axes) ? Base.Broadcast.combine_axes(sbc.args...) : sbc.axes

Base.Broadcast.broadcasted(op::SpectralElementOperator, args...) =
    Base.Broadcast.broadcasted(SpectralStyle(), op, args...)

Base.Broadcast.broadcasted(::SpectralStyle, op::SpectralElementOperator, args...) =
    SpectralBroadcasted{SpectralStyle}(op, args)

Base.eltype(sbc::SpectralBroadcasted) =
    operator_return_eltype(sbc.op, map(eltype, sbc.args)...)

function Base.Broadcast.instantiate(sbc::SpectralBroadcasted{Style}) where {Style}
    op = sbc.op
    # recursively instantiate the arguments to allocate intermediate work arrays
    args = map(Base.Broadcast.instantiate, sbc.args)
    # axes: same logic as Broadcasted
    if sbc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Base.Broadcast.combine_axes(args...)
    else
        axes = sbc.axes
        Base.Broadcast.check_broadcast_axes(axes, args...)
    end
    # allocate intermediate work space
    work = allocate_work(op, args...)
    return SpectralBroadcasted{Style}(op, args, axes, work)
end

function Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{Style}) where {Style<:AbstractSpectralStyle}
    # recursively instantiate the arguments to allocate intermediate work arrays
    args = map(Base.Broadcast.instantiate, bc.args)
    # axes: same logic as Broadcasted
    if bc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Base.Broadcast.combine_axes(args...)
    else
        axes = bc.axes
        Base.Broadcast.check_broadcast_axes(axes, args...)
    end
    return Base.Broadcast.Broadcasted{Style}(bc.f, args, axes)
end

# @. divergence(A *  gradient(rhou / rho))
                      # slab_size(space)

function Base.similar(
    sbc::SpectralBroadcasted,
    ::Type{Eltype},
) where {Eltype}
    space = axes(sbc)
    return Field(similar(Spaces.coordinates(space), Eltype), space)
end

function Base.similar(
    bc::Base.Broadcast.Broadcasted{<:AbstractSpectralStyle},
    ::Type{Eltype},
) where {Eltype}
    space = axes(bc)
    return Field(similar(Spaces.coordinates(space), Eltype), space)
end

function Base.copy(sbc::SpectralBroadcasted)
    # figure out return type
    dest = similar(sbc, eltype(sbc))
    # allocate dest
    copyto!(dest, sbc)
end
Base.Broadcast.broadcastable(sbc::SpectralBroadcasted) = sbc

function Base.Broadcast.materialize(sbc::SpectralBroadcasted)
    copy(Base.Broadcast.instantiate(sbc))
end

# TODO: Monday slab_flux is a broadcasted expression, we need to define the eqv of a
# recursive getindex call on a broadcasted element expression for the next step
# like apply_slab!(out_slab, in_expr)
function Base.copyto!(
    field_out::Field,
    sbc::SpectralBroadcasted
)
    data_out = Fields.field_values(field_out)
    Nh = length(data_out)
    for h in 1:Nh
        slab_out = slab(field_out, h)
        slab_args = map(arg -> _apply_slab(slab(arg, h)), sbc.args)
        # TODO have a slab field type with local geometry
        #apply_slab!(slab_out, sbc.op, sbc.work, slab_args...)
        copy_slab!(slab_out, apply_slab(sbc.op, sbc.work, slab_args...))
    end
    return field_out
end

function Base.Broadcast.materialize!(dest, sbc::SpectralBroadcasted)
    copyto!(dest, Base.Broadcast.instantiate(sbc))
end

function slab(sbc::SpectralBroadcasted{Style}, h) where {Style<:SpectralStyle}
    _args = map(a -> slab(a, h), sbc.args)
    _axes = slab(axes(sbc), h)
    SpectralBroadcasted{Style}(sbc.op, _args, _axes, sbc.work)
end

function slab(bc::Base.Broadcast.Broadcasted{Style}, h) where {Style<:AbstractSpectralStyle}
    _args = map(a -> slab(a, h), bc.args)
    _axes = slab(axes(bc), h)
    Base.Broadcast.Broadcasted{Style}(bc.f, _args, _axes)
end
abstract type OperatorSlabResult{S,Nq} <: DataLayouts.DataSlab2D{S,Nq}
end

Base.getproperty(slab_res::OperatorSlabResult, name::Symbol) = getfield(slab_res, name)

function Base.copyto!(
    field_out::Field,
    bc::Base.Broadcast.Broadcasted{Style}
) where {Style <: AbstractSpectralStyle}
    data_out = Fields.field_values(field_out)
    Nh = length(data_out)
    for h in 1:Nh
        slab_out = slab(field_out, h)
        slab_args = map(arg -> _apply_slab(slab(arg, h)), bc.args)
        # TODO have a slab field type with local geometry
        #apply_slab!(slab_out, sbc.op, sbc.work, slab_args...)
        copy_slab!(slab_out, Base.Broadcast.Broadcasted{Style}(bc.f, slab_args, axes(slab_out)))
    end
    return field_out
end

function copy_slab!(slab_out, res)
    space = axes(slab_out)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)
    for i in 1:Nq, j in 1:Nq
        # slab_out[i, j] = res[i, j]
        set_node!(slab_out, i, j, get_node(res, i, j))
    end
    return slab_out
end


@inline get_node(field::Fields.SlabField, i, j) = getindex(Fields.field_values(field), i,j)
@inline get_node(bc::Base.Broadcast.Broadcasted, i, j) = bc.f(map(arg -> get_node(arg, i, j), bc.args)...)
@inline get_node(scalar, i, j) = scalar

set_node!(field::Fields.SlabField, i, j, val) = setindex!(Fields.field_values(field), val, i,j)





#res = Broadcasted{CompositeSpectralStyle}(-, Field(StrongDivergenceResult{S, Nq}(Jv¹, Jv²), slab_space))

#=
function Base.copyto!(slab_out::Fields.SlabField, res::Base.Broadcast.Broadcasted{<:AbstractSpectralStyle})
    space = axes(slab_out)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)
    for i in 1:Nq, j in 1:Nq
        slab_out[i, j] = res[i, j]
    end
    return slab_out
end

=#

_apply_slab(x) = x
_apply_slab(sbc::SpectralBroadcasted) = apply_slab(sbc.op, sbc.work, map(_apply_slab, sbc.args)...)
_apply_slab(bc::Base.Broadcast.Broadcasted{CompositeSpectralStyle}) =
    Base.Broadcast.Broadcasted{CompositeSpectralStyle}(bc.f, map(_apply_slab, bc.args), bc.axes)


function Base.Broadcast.BroadcastStyle(::Type{SB}) where {SB<:SpectralBroadcasted}
    CompositeSpectralStyle()
end

function Base.Broadcast.BroadcastStyle(
    ::CompositeSpectralStyle,
    ::Fields.AbstractFieldStyle,
)
    CompositeSpectralStyle()
end


"""
    StrongDivergence()

Computes the "strong" divergence of a vector field `v`.

We compute the divergence as

    [∂(Jv¹)/∂ξ¹ + ∂(Jv²)/∂ξ²] / J

where `J` is the Jacobian determinant, `vⁱ` is the `i`th contravariant component of `v`.

This is discretized at the quadrature points as

    I{[∂(I{Jv¹})/∂ξ¹ + ∂(I{Jv²})/∂ξ²] / J}

where `I{x}` is the interpolation operator applied to a field `x`.

## References
 - Taylor and Fournier (2010), equation 15
"""
struct StrongDivergence <: SpectralElementOperator
end

"""
    WeakDivergence()

Computes the divergence of a vector field `v` weakly.

This spolves the variational problem of finding `θ` such that

    ⟨ϕ, J θ⟩ = - ⟨∂ϕ/∂ξ¹, J v¹⟩ + ⟨∂ϕ/∂ξ², J v²⟩

for all `ϕ` (which arises by integration by parts).

Discretely it is equivalent to

    - J / (D₁' * W * J * v¹ + D₂' * W * J * v²)

where

 - `J` is the diagonal Jacobian matrix
 - `W` is the diagonal matrix of quadrature weights
 - `D₁` and `D₂` are the discrete derivative matrices along the first and second dimensions.
"""
struct WeakDivergence <: SpectralElementOperator
end

operator_return_eltype(op::Union{WeakDivergence, StrongDivergence}, S) =
    RecursiveApply.rmaptype(Geometry.divergence_result_type, S)

function allocate_work(op::Union{WeakDivergence, StrongDivergence}, arg)
    space = axes(arg)

    FT = Spaces.undertype(space)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    ST = operator_return_eltype(op, eltype(arg))

    a¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
    a² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
    return (a¹, a²)
end

struct StrongDivergenceResult{S,Nq,JM} <: OperatorSlabResult{S,Nq}
    Jv¹::JM
    Jv²::JM
end
StrongDivergenceResult{S,Nq}(Jv¹::JM, Jv²::JM) where {S,Nq,JM} =
    StrongDivergenceResult{S,Nq,JM}(Jv¹, Jv²)

function apply_slab(op::StrongDivergence, (Jv¹, Jv²), slab_flux)
    slab_space = axes(slab_flux)
    Nq = Quadratures.degrees_of_freedom(slab_space.quadrature_style)
    local_geometry_slab = slab_space.local_geometry
    for j in 1:Nq, i in 1:Nq
        local_geometry = local_geometry_slab[i, j]
        # compute flux in contravariant coordinates (v¹,v²)
        # alternatively we could do this conversion _after_ taking the derivatives
        # may have an effect on the accuracy
        # materialize if lazy
        F = get_node(slab_flux, i, j)  # materialize the flux
        Jv¹[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.J *
                Geometry.contravariant1(x, local_geometry),
            F,
        )
        Jv²[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.J *
                Geometry.contravariant2(x, local_geometry),
            F,
        )
    end
    S = eltype(Jv¹)
    return Field(StrongDivergenceResult{S, Nq}(Jv¹, Jv²), slab_space)
end

function get_node(field::Fields.SlabField{<:StrongDivergenceResult}, i ,j)
    slab_space = axes(field)
    FT = Spaces.undertype(slab_space)
    D = Quadratures.differentiation_matrix(FT, slab_space.quadrature_style)
    res = Fields.field_values(field)

    # compute spectral deriv along first dimension
    ∂₁Jv¹ = RecursiveApply.rmatmul1(D, res.Jv¹, i, j) # ∂(Jv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
    # compute spectral deriv along second dimension
    ∂₂Jv² = RecursiveApply.rmatmul2(D, res.Jv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
    return inv(slab_space.local_geometry[i, j].J) ⊠ (∂₁Jv¹ ⊞ ∂₂Jv²)
end



struct WeakDivergenceResult{S,Nq,JM} <: OperatorSlabResult{S,Nq}
    WJv¹::JM
    WJv²::JM
end
WeakDivergenceResult{S,Nq}(Jv¹::JM, Jv²::JM) where {S,Nq,JM} =
    WeakDivergenceResult{S,Nq,JM}(Jv¹, Jv²)

function apply_slab(op::WeakDivergence, (WJv¹, WJv²), slab_flux)
    slab_space = axes(slab_flux)
    Nq = Quadratures.degrees_of_freedom(slab_space.quadrature_style)
    local_geometry_slab = slab_space.local_geometry
    for j in 1:Nq, i in 1:Nq
        local_geometry = local_geometry_slab[i, j]
        # compute flux in contravariant coordinates (v¹,v²)
        # alternatively we could do this conversion _after_ taking the derivatives
        # may have an effect on the accuracy
        # materialize if lazy
        F = get_node(slab_flux, i, j)  # materialize the flux
        WJv¹[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.WJ *
                Geometry.contravariant1(x, local_geometry),
            F,
        )
        WJv²[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.WJ *
                Geometry.contravariant2(x, local_geometry),
            F,
        )
    end
    S = eltype(WJv¹)
    return Field(WeakDivergenceResult{S, Nq}(WJv¹, WJv²), slab_space)
end

function get_node(field::Fields.SlabField{<:WeakDivergenceResult}, i ,j)
    slab_space = axes(field)
    FT = Spaces.undertype(slab_space)
    D = Quadratures.differentiation_matrix(FT, slab_space.quadrature_style)
    res = Fields.field_values(field)

    # compute spectral deriv along first dimension
    Dᵀ₁WJv¹ = RecursiveApply.rmatmul1(D', res.WJv¹, i, j) # D'WJv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
    # compute spectral deriv along second dimension
    Dᵀ₂WJv² = RecursiveApply.rmatmul2(D', res.WJv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
    return (-inv(slab_space.local_geometry[i, j].WJ)) ⊠ (Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv²)
end


# Strong gradient
"""
    Gradient()

Compute the (strong) gradient on each element via the chain rule:

    ∂f/∂xⁱ = ∂f/∂ξʲ * ∂ξʲ/∂xⁱ
"""
struct Gradient <: SpectralElementOperator end

operator_return_eltype(op::Gradient, S) =
    RecursiveApply.rmaptype(T -> Cartesian12Vector{T}, S)

struct GradientResult{S,Nq,JM}  <: OperatorSlabResult{S,Nq}
    M::JM
end
GradientResult{S,Nq}(M::JM) where {S,Nq,JM} = GradientResult{S,Nq,JM}(M)

function allocate_work(op::Gradient, arg)
    space = axes(arg)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)
    S = eltype(arg)
    # TODO: switch memory order?
    return MArray{Tuple{Nq, Nq}, S, 2, Nq * Nq}(undef)
end

function apply_slab(op::Gradient, M, slab_field)
    slab_space = axes(slab_field)
    Nq = Quadratures.degrees_of_freedom(slab_space.quadrature_style)
    for i in 1:Nq, j in 1:Nq
        M[i,j] = get_node(slab_field, i, j)
    end
    S = operator_return_eltype(op::Gradient, eltype(M))
    return Field(GradientResult{S,Nq}(M), slab_space)
end

function get_node(field::Fields.SlabField{<:GradientResult}, i, j)
    slab_space = axes(field)
    FT = Spaces.undertype(slab_space)
    D = Quadratures.differentiation_matrix(FT, slab_space.quadrature_style)
    res = Fields.field_values(field)

    ∂f∂ξ₁ = RecursiveApply.rmatmul1(D, res.M, i, j)
    ∂f∂ξ₂ = RecursiveApply.rmatmul2(D, res.M, i, j)
    ∂f∂ξ = RecursiveApply.rmap(Covariant12Vector, ∂f∂ξ₁, ∂f∂ξ₂)
    # TODO: return a CovariantVector by default;
    #       use subsequent broadcasting to convert to desired basis
    return RecursiveApply.rmap(
        x -> Cartesian12Vector(x, slab_space.local_geometry[i,j]),
        ∂f∂ξ,
    )
end


struct WeakCurl <: SpectralElementOperator end
struct StrongCurl <: SpectralElementOperator end


abstract type TensorOperator <: SpectralElementOperator
struct Interpolate{S} <: TensorOperator
    space::S
end
struct Restrict{S} <: TensorOperator
    space::S
end


operator_return_eltype(op::TensorOperator, S) = S

struct TensorResult{S,Nq,M,TM}  <: OperatorSlabResult{S,Nq}
    mat::M
    temp2::TM
end
TensorResult{S,Nq}(mat::M,temp2::TM) where {S,Nq,M,TM} = TensorResult{S,Nq,M,TM}(mat, temp2)


function allocate_work(op::Interpolate, arg)
    space_in = axes(arg)
    Nq_in = Quadratures.degrees_of_freedom(space_in.quadrature_style)
    space_out = op.space
    Nq_out = Quadratures.degrees_of_freedom(space_out.quadrature_style)
    mat = Quadratures.interpolation_matrix(
        Float64,
        space_in.quadrature_style,
        space_out.quadrature_style,
    )

    S = eltype(arg)
    # TODO: switch memory order?
    temp1 = MArray{Tuple{Nq_in, Nq_in}, S, 2, Nq * Nq}(undef)
    temp2 = MArray{Tuple{Nq_out, Nq_in}, S, 2, Nq * Nq}(undef)
    return (mat, temp1, temp2)
end

function allocate_work(op::Restrict, arg)
    (mat, temp1, temp2) = allocate_work(Interpolate(op.space), arg)
    return (mat', temp1, temp2)
end

function apply_slab(op::TensorOperator, (mat, temp1, temp2), slab_field)
    space_in = axes(slab_field)
    Nq_in = Quadratures.degrees_of_freedom(space_in.quadrature_style)
    space_out = op.space
    Nq_out = Quadratures.degrees_of_freedom(space_out.quadrature_style)
    for i in 1:Nq_in, j in 1:Nq_in
        temp1[i,j] = get_node(slab_field, i, j)
    end
    for j in 1:Nq_in, i in 1:Nq_out
        temp2[i, j] = RecursiveApply.rmatmul1(mat, temp1, i, j)
    end
    S = eltype(arg)
    return Field(InterpolateResult{S,Nq_out}(mat, temp2), slab_space)
end

function get_node(field::Fields.SlabField{<:TensorResult}, i, j)
    res = Fields.field_values(field)
    return RecursiveApply.rmatmul2(res.mat, res.temp2, i, j)
end













# TODO:
#  - convenience operations for fields
#  - determine output element type
#  - let inputs be Broadcasted objects
#    - make sure that
"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element.
"""
function tensor_product!(
    out::Data2D{S, Nij_out},
    in::Data2D{S, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}

    Nh = length(in)
    @assert Nh == length(out)

    # temporary storage
    temp = MArray{Tuple{Nij_out, Nij_in}, S, 2, Nij_out * Nij_in}(undef)

    for h in 1:Nh
        in_slab = slab(in, h)
        out_slab = slab(out, h)
        for j in 1:Nij_in, i in 1:Nij_out
            temp[i, j] = RecursiveApply.rmatmul1(M, in_slab, i, j)
        end
        for j in 1:Nij_out, i in 1:Nij_out
            out_slab[i, j] = RecursiveApply.rmatmul2(M, temp, i, j)
        end
    end
    return out
end

"""
    tensor_product!(out, in, M)
    tensor_product!(inout, M)

Computes the tensor product `out = (M ⊗ M) * in` on each element.
"""
function tensor_product!(
    out_slab::DataLayouts.DataSlab2D{S, Nij_out},
    in_slab::DataLayouts.DataSlab2D{S, Nij_in},
    M::SMatrix{Nij_out, Nij_in},
) where {S, Nij_out, Nij_in}

    # temporary storage
    temp = MArray{Tuple{Nij_out, Nij_in}, S, 2, Nij_out * Nij_in}(undef)

    for j in 1:Nij_in, i in 1:Nij_out
        temp[i, j] = RecursiveApply.rmatmul1(M, in_slab, i, j)
    end
    for j in 1:Nij_out, i in 1:Nij_out
        out_slab[i, j] = RecursiveApply.rmatmul2(M, temp, i, j)
    end
    return out_slab
end

function tensor_product!(
    inout::Data2D{S, Nij},
    M::SMatrix{Nij, Nij},
) where {S, Nij}
    tensor_product!(inout, inout, M)
end


"""
    slab_gradient!(∇data, data, space)

Compute the gradient on each element via the chain rule:

    ∂f/∂xⁱ = ∂f/∂ξʲ * ∂ξʲ/∂xⁱ
"""
function slab_gradient!(∇data, data, space)
    # all derivatives calculated in the reference local geometry FT precision
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    # for each element in the element stack
    Nh = length(data)
    for h in 1:Nh
        ∇data_slab = slab(∇data, h)
        # TODO: can we just call materialize(slab(data,h))
        data_slab = slab(data, h)
        local_geometry_slab = slab(space.local_geometry, h)

        for i in 1:Nq, j in 1:Nq
            # TODO: materialize data_slab once
            # on GPU this would be done to shared memory, then synchronize threads
            local_geometry = local_geometry_slab[i, j]

            # compute covariant derivatives
            ∂f∂ξ₁ = RecursiveApply.rmatmul1(D, data_slab, i, j)
            ∂f∂ξ₂ = RecursiveApply.rmatmul2(D, data_slab, i, j)
            ∂f∂ξ = RecursiveApply.rmap(Covariant12Vector, ∂f∂ξ₁, ∂f∂ξ₂)

            # convert to desired basis
            ∇data_slab[i, j] = RecursiveApply.rmap(
                x -> Cartesian12Vector(x, local_geometry),
                ∂f∂ξ,
            )
        end
    end
    return ∇data
end


"""
    slab_divergence!(divflux, flux, space)

Compute the divergence of `flux`, storing the result in `divflux`.

Given a vector field `v`, we compute the divergence as

    [∂(Jv¹)/∂ξ¹ + ∂(Jv²)/∂ξ²] / J

where `J` is the Jacobian determinant, `vⁱ` is the `i`th contravariant component of `v`.

This is discretized at the quadrature points as

    I{[∂(I{Jv¹})/∂ξ¹ + ∂(I{Jv²})/∂ξ²] / J}

where `I{x}` is the interpolation operator applied to a field `x`.

## References
 - Taylor and Fournier (2010), equation 15
"""
function slab_divergence!(divflux, flux, space)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        divflux_slab = slab(divflux, h)
        flux_slab = slab(flux, h)
        local_geometry_slab = slab(space.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        Jv¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        Jv² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute flux in contravariant coordinates (v¹,v²)
            # alternatively we could do this conversion _after_ taking the derivatives
            # may have an effect on the accuracy
            # materialize if lazy
            F = flux_slab[i, j]  # materialize the flux
            Jv¹[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.J *
                    Geometry.contravariant1(x, local_geometry),
                F,
            )
            Jv²[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.J *
                    Geometry.contravariant2(x, local_geometry),
                F,
            )
        end
        # GPU synchronize
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            ∂₁v₂ = RecursiveOperators.rmatmul1(D, Jv¹, i, j) # ∂(Jv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
            # compute spectral deriv along second dimesion
            ∂₂v₁ = RecursiveOperators.rmatmul2(D, Jv², i, j)
            divflux_slab[i, j] = Contravariant3Vector(inv(local_geometry.J) ⊠ (∂₁v₂ ⊟ ∂₂v₁))
        end
    end
    return divflux
end

function slab_strong_curl!(curlvec, vec, mesh)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Meshes.undertype(mesh)
    D = Quadratures.differentiation_matrix(FT, mesh.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(mesh.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        curlvec = slab(curlvec, h)
        vec_slab = slab(vec, h)
        local_geometry_slab = slab(mesh.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        v₁ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        v₂ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # we convert the vec to covariant coordinates (v₁, v₂),
            # and return an orthogonal vector in contravariant coordinates v³
            v₁[i, j] = RecursiveOperators.rmap(
                x ->
                    Geometry.covariant1(x, local_geometry),
                vec_slab,
            )
            v₂[i, j] = RecursiveOperators.rmap(
                x ->
                    Geometry.covariant2(x, local_geometry),
                vec_slab,
            )
        end
        # GPU synchronize
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            ∂₁v₂ = RecursiveOperators.rmatmul1(D, v₂, i, j)
            # compute spectral deriv along second dimension
            ∂₂v₁ = RecursiveOperators.rmatmul2(D, v₁, i, j)
            curlvec_slab[i, j] = Contravariant3Vector(inv(local_geometry.J) ⊠ (∂₁v₂ ⊟ ∂₂v₁))
        end
    end
    return curlvec
end

function slab_weak_curl!(curlvec, vec, mesh)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Meshes.undertype(mesh)
    D = Quadratures.differentiation_matrix(FT, mesh.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(mesh.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        curlvec = slab(curlvec, h)
        vec_slab = slab(vec, h)
        local_geometry_slab = slab(mesh.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        v₁ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        v₂ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # we convert the vec to covariant coordinates (v₁, v₂),
            # and return an orthogonal vector in contravariant coordinates v³
            v₁[i, j] = RecursiveOperators.rmap(
                x ->
                    Geometry.covariant1(x, local_geometry),
                vec_slab,
            )
            v₂[i, j] = RecursiveOperators.rmap(
                x ->
                    Geometry.covariant2(x, local_geometry),
                vec_slab,
            )
        end
        # GPU synchronize
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            ∂₁v₂ = RecursiveOperators.rmatmul1(D, v₂, i, j)
            # compute spectral deriv along second dimension
            ∂₂v₁ = RecursiveOperators.rmatmul2(D, v₁, i, j)
            curlvec_slab[i, j] = Contravariant3Vector(inv(local_geometry.J) ⊠ (∂₁v₂ ⊟ ∂₂v₁))
        end
    end
    return curlvec
end


"""
    slab_weak_divergence!(divflux, flux, space)

Compute the right-hand side of the divergence of `flux` weakly, storing the result in `divflux`.

This computes the right-hand side of the variational problem of finding θ such that

    - ⟨ϕ, J θ⟩ = ⟨∂ϕ/∂ξ¹, J u¹⟩ + ⟨∂ϕ/∂ξ², J u²⟩

for all `ϕ` (which arises by integration by parts).

Discretely it is equivalent to

    (D₁' * W * J * u¹ + D₂' * W * J * u²)

where
 - `J` is the diagonal Jacobian matrix
 - `W` is the diagonal matrix of quadrature weights
 - `D₁` and `D₂` are the discrete derivative matrices along the first and second dimensions.
"""
function slab_weak_divergence!(divflux, flux, space)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Spaces.undertype(space)
    D = Quadratures.differentiation_matrix(FT, space.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space.quadrature_style)

    # for each element in the element stack
    Nh = length(flux)
    for h in 1:Nh
        divflux_slab = slab(divflux, h)
        flux_slab = slab(flux, h)
        local_geometry_slab = slab(space.local_geometry, h)

        ST = eltype(divflux)
        # Shared on GPU
        WJv¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        WJv² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute flux in contravariant coordinates (v¹,v²)
            # alternatively we could do this conversion _after_ taking the derivatives
            # may have an effect on the accuracy
            # materialize if lazy
            F = flux_slab[i, j]
            WJv¹[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.WJ *
                    Geometry.contravariant1(x, local_geometry),
                F,
            )
            WJv²[i, j] = RecursiveApply.rmap(
                x ->
                    local_geometry.WJ *
                    Geometry.contravariant2(x, local_geometry),
                F,
            )
        end
        # GPU synchronize

        for i in 1:Nq, j in 1:Nq
            local_geometry = local_geometry_slab[i, j]
            # compute spectral deriv along first dimension
            Dᵀ₁WJv¹ = RecursiveApply.rmatmul1(D', WJv¹, i, j) # D'WJv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
            # compute spectral deriv along second dimension
            Dᵀ₂WJv² = RecursiveApply.rmatmul2(D', WJv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
            divflux_slab[i, j] = Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv²
        end
    end
    return divflux
end

function slab_weak_divergence!(
    divflux_slab::DataLayouts.DataSlab2D,
    flux_slab::DataLayouts.DataSlab2D,
    space_slab::Spaces.SpectralElementSpaceSlab,
)
    # all derivatives calculated in the reference local geometry with FT precision
    FT = Spaces.undertype(space_slab)
    D = Quadratures.differentiation_matrix(FT, space_slab.quadrature_style)
    Nq = Quadratures.degrees_of_freedom(space_slab.quadrature_style)

    # for each element in the element stack
    local_geometry_slab = space_slab.local_geometry

    ST = eltype(divflux_slab)
    # Shared on GPU
    WJv¹ = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
    WJv² = MArray{Tuple{Nq, Nq}, ST, 2, Nq * Nq}(undef)
    for i in 1:Nq, j in 1:Nq
        local_geometry = local_geometry_slab[i, j]
        # compute flux in contravariant coordinates (v¹,v²)
        # alternatively we could do this conversion _after_ taking the derivatives
        # may have an effect on the accuracy
        # materialize if lazy
        F = flux_slab[i, j]
        WJv¹[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.WJ * Geometry.contravariant1(x, local_geometry),
            F,
        )
        WJv²[i, j] = RecursiveApply.rmap(
            x ->
                local_geometry.WJ * Geometry.contravariant2(x, local_geometry),
            F,
        )
    end
    # GPU synchronize

    for i in 1:Nq, j in 1:Nq
        # compute spectral deriv along first dimension
        Dᵀ₁WJv¹ = RecursiveApply.rmatmul1(D', WJv¹, i, j) # D'WJv¹)/∂ξ¹ = D[i,:]*Jv¹[:,j]
        # compute spectral deriv along second dimension
        Dᵀ₂WJv² = RecursiveApply.rmatmul2(D', WJv², i, j) # ∂(Jv²)/∂ξ² = D[j,:]*Jv²[i,:]
        divflux_slab[i, j] = Dᵀ₁WJv¹ ⊞ Dᵀ₂WJv²
    end
    return divflux_slab
end





function slab_gradient!(∇field::Field, field::Field)
    @assert axes(∇field) === axes(field)
    Operators.slab_gradient!(
        Fields.field_values(∇field),
        Fields.field_values(field),
        axes(field),
    )
    return ∇field
end
function slab_divergence!(divflux::Field, flux::Field)
    @assert axes(divflux) === axes(flux)
    Operators.slab_divergence!(
        Fields.field_values(divflux),
        Fields.field_values(flux),
        axes(flux),
    )
    return divflux
end
function slab_weak_divergence!(divflux::Field, flux::Field)
    @assert axes(divflux) === axes(flux)
    Operators.slab_weak_divergence!(
        Fields.field_values(divflux),
        Fields.field_values(flux),
        axes(flux),
    )
    return divflux
end

function slab_gradient(field::Field)
    S = eltype(field)
    ∇S = RecursiveApply.rmaptype(T -> Cartesian12Vector{T}, S)
    Operators.slab_gradient!(similar(field, ∇S), field)
end

function slab_divergence(field::Field)
    S = eltype(field)
    divS = RecursiveApply.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_divergence!(similar(field, divS), field)
end
function slab_weak_divergence(field::Field)
    S = eltype(field)
    divS = RecursiveApply.rmaptype(Geometry.divergence_result_type, S)
    Operators.slab_weak_divergence!(similar(field, divS), field)
end



function interpolate(space_to::AbstractSpace, field_from::Field)
    field_to = similar(field_from, (space_to,), eltype(field_from))
    interpolate!(field_to, field_from)
end
function interpolate!(field_to::Field, field_from::Field)
    space_to = axes(field_to)
    space_from = axes(field_from)
    # @assert space_from.topology == space_to.topology

    M = Quadratures.interpolation_matrix(
        Float64,
        space_to.quadrature_style,
        space_from.quadrature_style,
    )
    Operators.tensor_product!(
        Fields.field_values(field_to),
        Fields.field_values(field_from),
        M,
    )
    return field_to
end

function restrict!(field_to::Field, field_from::Field)
    space_to = axes(field_to)
    space_from = axes(field_from)
    # @assert space_from.topology == space_to.topology

    M = Quadratures.interpolation_matrix(
        Float64,
        space_from.quadrature_style,
        space_to.quadrature_style,
    )
    Operators.tensor_product!(
        Fields.field_values(field_to),
        Fields.field_values(field_from),
        M',
    )
    return field_to
end

function matrix_interpolate(
    field::Field,
    Q_interp::Quadratures.Uniform{Nu},
) where {Nu}
    S = eltype(field)
    space = axes(field)
    mesh = space.topology.mesh
    n1 = mesh.n1
    n2 = mesh.n2

    interp_data =
        DataLayouts.IH1JH2{S, Nu}(Matrix{S}(undef, (Nu * n1, Nu * n2)))

    M = Quadratures.interpolation_matrix(
        Float64,
        Q_interp,
        space.quadrature_style,
    )
    Operators.tensor_product!(interp_data, Fields.field_values(field), M)
    return parent(interp_data)
end
matrix_interpolate(field::Field, Nu::Integer) =
    matrix_interpolate(field, Quadratures.Uniform{Nu}())
