import ClimaCore.MatrixFields
import ClimaCore.Operators: any_fd_shmem_supported

"""
    any_fd_shmem_supported(x)

Returns a Bool indicating if any broadcasted object can support a shmem style
"""
function any_fd_shmem_supported end

# Main entry point: call recursive function with 2 args
@inline any_fd_shmem_supported(bc) = any_fd_shmem_supported(false, bc)

@inline _any_fd_shmem_supported_args(falsesofar, args::Tuple) =
    falsesofar ||
    any_fd_shmem_supported(falsesofar, args[1]) ||
    _any_fd_shmem_supported_args(falsesofar, Base.tail(args))

@inline _any_fd_shmem_supported_args(falsesofar, args::Tuple{Any}) =
    falsesofar || any_fd_shmem_supported(falsesofar, args[1])

@inline _any_fd_shmem_supported_args(falsesofar, args::Tuple{}) = falsesofar

@inline function _any_fd_shmem_supported(
    falsesofar,
    bc::Base.Broadcast.Broadcasted,
)
    return falsesofar || _any_fd_shmem_supported_args(falsesofar, bc.args)
end

@inline _any_fd_shmem_supported(falsesofar, bc) = falsesofar

@inline _any_fd_shmem_supported(falsesofar, _, x) = falsesofar

@inline any_fd_shmem_supported(falsesofar, x) = falsesofar # generic fallback
@inline any_fd_shmem_supported(falsesofar, bc::StencilBroadcasted) =
    falsesofar ||
    Operators.fd_shmem_is_supported(bc) ||
    _any_fd_shmem_supported_args(falsesofar, bc.args)

@inline any_fd_shmem_supported(falsesofar, bc::Base.Broadcast.Broadcasted) =
    falsesofar || _any_fd_shmem_supported_args(falsesofar, bc.args)

@inline any_fd_shmem_supported(bc::Base.Broadcast.Broadcasted) =
    _any_fd_shmem_supported_args(false, bc.args)


# Fallback is false:
@inline Operators.fd_shmem_is_supported(bc::StencilBroadcasted) =
    Operators.fd_shmem_is_supported(bc.op)


"""
    any_fd_shmem_style()

Returns a Bool indicating if any broadcasted object has a shmem style
"""
function any_fd_shmem_style end

@inline any_fd_shmem_style(bc) = any_fd_shmem_style(false, bc)

@inline _any_fd_shmem_style_args(falsesofar, args::Tuple) =
    falsesofar ||
    any_fd_shmem_style(falsesofar, args[1]) ||
    _any_fd_shmem_style_args(falsesofar, Base.tail(args))

@inline _any_fd_shmem_style_args(falsesofar, args::Tuple{Any}) =
    falsesofar || any_fd_shmem_style(falsesofar, args[1])

@inline _any_fd_shmem_style_args(falsesofar, args::Tuple{}) = falsesofar

@inline function _any_fd_shmem_style(falsesofar, bc::Base.Broadcast.Broadcasted)
    return falsesofar || _any_fd_shmem_style_args(falsesofar, bc.args)
end

@inline _any_fd_shmem_style(falsesofar, bc) = falsesofar

@inline _any_fd_shmem_style(falsesofar, _, x) = falsesofar

@inline any_fd_shmem_style(falsesofar, x) = falsesofar # generic fallback
@inline any_fd_shmem_style(
    falsesofar,
    bc::StencilBroadcasted{CUDAWithShmemColumnStencilStyle},
) = falsesofar || _any_fd_shmem_style_args(falsesofar, bc.args)

@inline any_fd_shmem_style(falsesofar, bc::StencilBroadcasted) =
    falsesofar || _any_fd_shmem_style_args(falsesofar, bc.args)

"""
    disable_shmem_style()

For high resolution cases, shmem may not work, so `disable_shmem` transforms a
the boradcast style from
`CUDAWithShmemColumnStencilStyle` to `CUDAColumnStencilStyle`.
"""
function disable_shmem_style end

@inline disable_shmem_style_args(args::Tuple) =
    (disable_shmem_style(args[1]), disable_shmem_style_args(Base.tail(args))...)
@inline disable_shmem_style_args(args::Tuple{Any}) =
    (disable_shmem_style(args[1]),)
@inline disable_shmem_style_args(args::Tuple{}) = ()

@inline function disable_shmem_style(
    bc::StencilBroadcasted{CUDAWithShmemColumnStencilStyle},
)
    StencilBroadcasted{CUDAColumnStencilStyle}(
        bc.op,
        disable_shmem_style_args(bc.args),
        bc.axes,
        nothing,
    )
end

@inline function disable_shmem_style(
    bc::Base.Broadcast.Broadcasted{CUDAWithShmemColumnStencilStyle},
)
    Base.Broadcast.Broadcasted{CUDAColumnStencilStyle}(
        bc.f,
        disable_shmem_style_args(bc.args),
        bc.axes,
    )
end
@inline disable_shmem_style(x) = x

##### MatrixFields
@inline Operators.fd_shmem_is_supported(
    bc::MatrixFields.LazyOperatorBroadcasted,
) = false

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

##### DivergenceF2C
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

##### GradientC2F
@inline Operators.fd_shmem_is_supported(op::Operators.GradientC2F) =
    Operators.fd_shmem_is_supported(op, op.bcs)
@inline Operators.fd_shmem_is_supported(
    op::Operators.GradientC2F,
    ::@NamedTuple{},
) = false
@inline Operators.fd_shmem_is_supported(
    op::Operators.GradientC2F,
    bcs::NamedTuple,
) =
    all(values(bcs)) do bc
        all(supported_bc -> bc isa supported_bc, (Operators.SetValue,))
    end
