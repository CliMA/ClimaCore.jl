# Backwards-compatibility aliases for the pre-FormType spectral element
# operators (e.g., WeakDivergence for Divergence{WeakForm}), without deprecation
# warnings, as in src/DataLayouts/deprecated.jl. Remove the aliases below when
# all downstream consumers have migrated to the parameterized names (type-alias
# cleanup, Phase 4 of https://github.com/CliMA/ClimaCore.jl/issues/2554);
# get_node at the end of this file is live API used by the Remapping module.

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
const WeakDivergence = Divergence{WeakForm}

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
const WeakGradient = Gradient{WeakForm}

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
const WeakCurl = Curl{WeakForm}

# Nodal getter for Fields, used by the Remapping module and its CUDA extension.
# A 1-dimensional index is a column of a 1D space, so its second index is 1.
Base.@propagate_inbounds function get_node(
    parent_space, field::Fields.Field, ij::CartesianIndex{N}, slabidx,
) where {N}
    space = reconstruct_placeholder_space(axes(field), parent_space)
    _v =
        if space isa Spaces.FaceExtrudedFiniteDifferenceSpace ||
           space isa Spaces.FaceFiniteDifferenceSpace
            slabidx.v + half
        elseif space isa Spaces.CenterExtrudedFiniteDifferenceSpace ||
               space isa Spaces.AbstractSpectralElementSpace ||
               space isa Spaces.CenterFiniteDifferenceSpace
            slabidx.v
        else
            error("invalid space")
        end
    (i, j) = N == 1 ? (ij[1], 1) : Tuple(ij)
    return Fields.field_values(field)[isnothing(_v) ? 1 : _v, i, j, slabidx.h]
end
