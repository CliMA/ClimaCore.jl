# Backwards-compatibility aliases for the pre-FormType spectral element
# operators. The weak-form operators are now the WeakForm variants of their
# strong-form counterparts (e.g., Divergence{I, WeakForm} instead of a separate
# WeakDivergence type); the aliases below keep the old names working. As with
# src/DataLayouts/deprecated.jl, these are plain aliases without deprecation
# warnings. Remove this file when all downstream consumers have migrated to the
# parameterized names (type-alias cleanup, Phase 4 of
# https://github.com/CliMA/ClimaCore.jl/issues/2554).

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
const WeakDivergence{I} = Divergence{I, WeakForm}
WeakDivergence() = WeakDivergence{()}()

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
const WeakGradient{I} = Gradient{I, WeakForm}
WeakGradient() = WeakGradient{()}()

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
const WeakCurl{I} = Curl{I, WeakForm}
WeakCurl() = WeakCurl{()}()

"""
    SlabData{T}

A [`DataLayouts.DataLayout`](@ref) that stores a single slab of values of type
`T` (a `VIJHWithF` layout with `Nv = Nh = 1`).
"""
const SlabData{T} = DataLayouts.VIJHWithF{T, 1, <:Any, <:Any, 1}

# Index for one node in a slab of data, with v = h = 1.
@inline slab_node_index(ij::CartesianIndex{1}) = CartesianIndex(1, ij[1], 1, 1)
@inline slab_node_index(ij::CartesianIndex{2}) =
    CartesianIndex(1, ij[1], ij[2], 1)

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
    return fv[v, i, 1, h]
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
    return fv[v, i, j, h]
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
Base.@propagate_inbounds function get_node(space, data::SlabData, ij, slabidx)
    data[slab_node_index(ij)]
end
Base.@propagate_inbounds function get_node(
    space,
    field::Fields.Field{<:SlabData},
    ij::CartesianIndex{1},
    slabidx,
)
    Fields.field_values(field)[slab_node_index(ij)]
end
Base.@propagate_inbounds function get_node(
    space,
    field::Fields.Field{<:SlabData},
    ij::CartesianIndex{2},
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
