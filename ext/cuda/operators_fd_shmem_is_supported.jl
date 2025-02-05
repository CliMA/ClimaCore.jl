import ClimaCore.MatrixFields
import ClimaCore.Operators: any_fd_shmem_supported

@inline _any_fd_shmem_supported_args(falsesofar, args::Tuple, rargs...) =
    falsesofar &&
    _any_fd_shmem_supported(falsesofar, args[1], rargs...) &&
    _any_fd_shmem_supported_args(falsesofar, Base.tail(args), rargs...)

@inline _any_fd_shmem_supported_args(falsesofar, args::Tuple{Any}, rargs...) =
    falsesofar && _any_fd_shmem_supported(falsesofar, args[1], rargs...)
@inline _any_fd_shmem_supported_args(falsesofar, args::Tuple{}, rargs...) =
    falsesofar

@inline function _any_fd_shmem_supported(
    falsesofar,
    bc::Base.Broadcast.Broadcasted,
)
    return falsesofar && _any_fd_shmem_supported_args(falsesofar, bc.args)
end

@inline _any_fd_shmem_supported(falsesofar, _, x::AbstractData) = false
@inline _any_fd_shmem_supported(falsesofar, _, x) = falsesofar

@inline any_fd_shmem_supported(bc) = any_fd_shmem_supported(false, bc)

@inline any_fd_shmem_supported(falsesofar, bc::StencilBroadcasted) =
    falsesofar ||
    Operators.fd_shmem_is_supported(bc) ||
    _any_fd_shmem_supported_args(falsesofar, bc.args)

@inline any_fd_shmem_supported(falsesofar, bc::Operators.Operator2Stencil) =
    falsesofar || Operators.fd_shmem_is_supported(bc)

@inline any_fd_shmem_supported(falsesofar, bc::Operators.ComposeStencils) =
    falsesofar || Operators.fd_shmem_is_supported(bc)

@inline any_fd_shmem_supported(falsesofar, bc::Operators.ApplyStencil) =
    falsesofar || Operators.fd_shmem_is_supported(bc)

@inline any_fd_shmem_supported(falsesofar, bc::Base.Broadcast.Broadcasted) =
    falsesofar || _any_fd_shmem_supported_args(falsesofar, bc.args)

@inline any_fd_shmem_supported(bc::Base.Broadcast.Broadcasted) =
    _any_fd_shmem_supported_args(false, bc.args)


# Fallback is false:
@inline Operators.fd_shmem_is_supported(bc::StencilBroadcasted) =
    Operators.fd_shmem_is_supported(bc.op)

##### MatrixFields
@inline Operators.fd_shmem_is_supported(op::Operators.Operator2Stencil) = false

@inline Operators.fd_shmem_is_supported(
    bc::MatrixFields.LazyOperatorBroadcasted,
) = false

@inline Operators.fd_shmem_is_supported(bc::Operators.ApplyStencil) = false

@inline Operators.fd_shmem_is_supported(bc::Operators.ComposeStencils) = false

@inline Operators.fd_shmem_is_supported(op::MatrixFields.FDOperatorMatrix) =
    false

@inline Operators.fd_shmem_is_supported(
    op::MatrixFields.LazyOneArgFDOperatorMatrix,
) = false
#####

@inline Operators.fd_shmem_is_supported(op::Operators.AbstractOperator) =
    Operators.fd_shmem_is_supported(op, op.bcs)

@inline Operators.fd_shmem_is_supported(
    op::MatrixFields.MultiplyColumnwiseBandMatrixField,
) = false

@inline Operators.fd_shmem_is_supported(
    op::Operators.AbstractOperator,
    bcs::NamedTuple,
) = false

# Add cases here where shmem is supported:
@inline Operators.fd_shmem_is_supported(op::Operators.DivergenceF2C) =
    Operators.fd_shmem_is_supported(op, op.bcs)
@inline Operators.fd_shmem_is_supported(
    op::Operators.DivergenceF2C,
    ::@NamedTuple{},
) = true
@inline Operators.fd_shmem_is_supported(
    op::Operators.DivergenceF2C,
    bcs::NamedTuple,
) =
    all(values(bcs)) do bc
        all(supported_bc -> bc isa supported_bc, (Operators.SetValue,))
    end
