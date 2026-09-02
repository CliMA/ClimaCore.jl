# Operators: spectral element

```@meta
CurrentModule = ClimaCore.Operators
```

Element-local operators on spectral-element spaces. Each acts inside a
broadcast expression and returns a field on the same space; a weak-form
result is complete only after DSS on a CG space or an interface flux on a DG
space ([Operators and broadcasting](../explanation/operators.md),
[DSS and numerical fluxes](../explanation/interelement.md)).

## Differential operators

```@docs
Gradient
Divergence
SplitDivergence
Curl
```

## Strong and weak forms

`Divergence`, `Gradient`, and `Curl` each have a strong and a weak variant,
selected by the [`FormType`](@ref) type parameter: `Divergence()` is the strong
form (`Divergence{StrongForm}`) and `Divergence{WeakForm}()` the weak form, and
likewise for `Gradient` and `Curl`.

```@docs
FormType
StrongForm
WeakForm
```

The names `WeakDivergence`, `WeakGradient`, and `WeakCurl` are aliases of the
weak-form types, kept for downstream packages.

```@docs
WeakDivergence
WeakGradient
WeakCurl
```

## Laplacians

Building blocks of hyperdiffusion on both discretizations.

```@docs
scalar_laplacian
scalar_laplacian!
vector_laplacian
```

## Interpolation operators

```@docs
Interpolate
Restrict
```
