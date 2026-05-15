# Backwards-compatibility shims for the pre-`Tensor` Geometry API.
#
# The PR collapses `AxisTensor` / `AxisVector` / `Axis2Tensor` and the
# `*Axis{I}` aliases into the single `Tensor` / `Basis` types. Downstream
# packages (ClimaAtmos, user code) that haven't migrated yet can keep
# referring to the old names through the aliases below.
#
# These are not deprecated with warnings — they're plain aliases — so old
# code continues to type-check without noise. Remove this file when all
# downstream consumers have migrated to the new names.

# --- Type aliases (parameter order matches the old API) ---

# Old: AxisTensor{T, N, B, S}. New: Tensor{N, T, B, S}. Parameters T and N
# are swapped relative to the new form.
const AxisTensor{T, N, B, S} = Tensor{N, T, B, S}
const AxisVector{T, A, S} = Tensor{1, T, Tuple{A}, S}
const Axis2Tensor{T, B, S} = Tensor{2, T, B, S}

const CovariantAxis{I} = Basis{Covariant, I}
const ContravariantAxis{I} = Basis{Contravariant, I}
const LocalAxis{I} = Basis{Orthonormal, I}
const CartesianAxis{I} = Basis{Orthonormal, I}

@inline AxisTensor(bases::Tuple, components) = Tensor(components, bases)
@inline components(x::AbstractTensor) = parent(x)
