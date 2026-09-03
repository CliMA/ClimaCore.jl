# Compatibility aliases for the pre-`Tensor` Geometry API.
#
# `AxisTensor` / `AxisVector` / `Axis2Tensor` and the `*Axis{I}` aliases are spelled with
# the `Tensor` / `Components` types. Downstream packages can keep referring to the names
# below, which are plain aliases without deprecation warnings.

# --- Type aliases (parameter order matches the deprecated API) ---

# Deprecated: AxisTensor{T, N, B, S}. Current: Tensor{N, T, B, S}. Parameters T and N
# are swapped relative to the current form.
const AxisTensor{T, N, B, S} = Tensor{N, T, B, S}
const AxisVector{T, A, S} = Tensor{1, T, Tuple{A}, S}
const Axis2Tensor{T, B, S} = Tensor{2, T, B, S}

const CovariantAxis{I} = Components{Covariant, I}
const ContravariantAxis{I} = Components{Contravariant, I}
const LocalAxis{I} = Components{Orthonormal, I}
const CartesianAxis{I} = Components{Orthonormal, I}

@inline AxisTensor(bases::Tuple, components) = Tensor(components, bases)
@inline components(x::AbstractTensor) = parent(x)
