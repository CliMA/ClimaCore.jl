# MatrixFields

```@meta
CurrentModule = ClimaCore.MatrixFields
```

```@docs
MatrixFields
```

## Matrix Field Element Type

```@docs
BandMatrixRow
```

## Matrix Field Multiplication

```@docs
MultiplyColumnwiseBandMatrixField
```

## Operator Matrices

```@docs
operator_matrix
```

## Vectors and Matrices of Fields

```@docs
FieldNameDict
identity_field_matrix
field_vector_view
concrete_field_vector
```

## Linear Solvers

```@docs
FieldMatrixSolverAlgorithm
FieldMatrixSolver
FieldMatrixWithSolver
field_matrix_solve!
BlockDiagonalSolve
BlockLowerTriangularSolve
BlockArrowheadSolve
SchurComplementReductionSolve
LazyFieldMatrixSolverAlgorithm
StationaryIterativeSolve
ApproximateBlockArrowheadIterativeSolve
```

## Preconditioners

```@docs
PreconditionerAlgorithm
MainDiagonalPreconditioner
BlockDiagonalPreconditioner
BlockArrowheadPreconditioner
BlockArrowheadSchurComplementPreconditioner
WeightedPreconditioner
CustomPreconditioner
```

## Utilities

```@docs
column_field2array
column_field2array_view
field2arrays
field2arrays_view
scalar_field_matrix
```

## Indexing a FieldMatrix

An entry of a `FieldMatrix` is one of

  - a `UniformScaling`, holding a `Number`;
  - a `DiagonalMatrixRow`, holding a `Number` or a `Geometry.Tensor{2}` in
    whatever basis the user supplies;
  - a `ColumnwiseBandMatrixField`: a `Field` whose values are
    [`BandMatrixRow`](@ref)s, one banded matrix per column, with entries of any
    type built from the field's base number type.

The keys are pairs of `@name`s. When an entry's element type is a composite
type, indexing with a longer name reaches into it, recursively:

```@example 1
using ClimaCore.CommonSpaces # hide
import ClimaCore: MatrixFields, Quadratures # hide
import ClimaCore.MatrixFields: @name # hide
space = Box3DSpace(; # hide
    z_elem = 3, # hide
    x_min = 0, # hide
    x_max = 1, # hide
    y_min = 0, # hide
    y_max = 1, # hide
    z_min = 0, # hide
    z_max = 10, # hide
    periodic_x = false, # hide
    periodic_y = false, # hide
    n_quad_points = 1, # hide
    quad = Quadratures.GL{1}(), # hide
    x_elem = 1, # hide
    y_elem = 2, # hide
    staggering = CellCenter(), # hide
) # hide
nt_entry_field = fill(MatrixFields.DiagonalMatrixRow((; foo = 1.0, bar = 2.0)), space)
nt_fieldmatrix = MatrixFields.FieldMatrix((@name(a), @name(b)) => nt_entry_field)
nt_fieldmatrix[(@name(a), @name(b))]
```

```@example 1
nt_fieldmatrix[(@name(a.foo), @name(b))]
```

```@example 1
nt_fieldmatrix[(@name(a.bar), @name(b))]
```

### Indexing rules

Let `(@name(name1), @name(name2))` be a key of `A` paired with `entry`, and
consider `A[(@name(name1.foo.bar), @name(name2.biz.bop))]`. `getindex` first
finds the key of `A` that contains the requested key; here
`(@name(name1), @name(name2))` is the *parent key* and
`(@name(foo.bar), @name(biz.bop))` the *internal key*. The entry is then
indexed by the internal key, which for a name pair `(n₁, n₂)` and an entry
whose bands have element type `T` proceeds as follows:

 1. If both names are empty, return the entry.
 2. If `T` is a `Geometry.Tensor{2}` and the pair has the form
    `(@name(components.data.i…), @name(components.data.j…))`, extract component
    `(i, j)` and recurse with the remaining names.
 3. If `T` is the `Adjoint` of a rank-1 tensor, recurse on its parent.
 4. If the first name of `n₁` is a field of `T`, extract it and recurse with
    the rest of `n₁` and all of `n₂`.
 5. Likewise for the first name of `n₂`.
 6. Otherwise both names are nonempty and neither is a field of `T`, and the
    entry is taken to represent a tensor implicitly, as a scaling of the
    identity (see below): if the first names of `n₁` and `n₂` agree, drop them
    and recurse on the entry; if they differ, drop them and recurse on its
    zero.

Indexing a `ColumnwiseBandMatrixField` returns a `Broadcasted` object rather
than a `Field` when the internal key reaches a type other than the entry's
base type or a zero created in rule 6; `Base.Broadcast.materialize` turns it
into a field.

```@setup 2
using ClimaCore.CommonSpaces
using ClimaCore.Geometry
using ClimaCore.Fields
import ClimaCore: MatrixFields
import ClimaCore.MatrixFields: @name
FT = Float64
space = ColumnSpace(FT ;
           z_elem = 6,
           z_min = 0,
           z_max = 10,
           staggering = CellCenter()
       )
f = map(x -> rand(Geometry.Covariant12Vector{Float64}), Fields.local_geometry_field(space))
g = map(x -> rand(Geometry.Covariant12Vector{Float64}), Fields.local_geometry_field(space))
identity_axis2tensor = Geometry.Covariant12Vector(FT(1), FT(0)) *
                   Geometry.Contravariant12Vector(FT(1), FT(0))' +
                   Geometry.Covariant12Vector(FT(0), FT(1)) *
                   Geometry.Contravariant12Vector(FT(0), FT(1))'
∂f_∂g = fill(MatrixFields.TridiagonalMatrixRow(-0.5 * identity_axis2tensor, identity_axis2tensor, -0.5 * identity_axis2tensor), space)
J = MatrixFields.FieldMatrix((@name(f), @name(g))=> ∂f_∂g)
```

## Storage optimizations

A `FieldMatrix` entry may be stored more compactly than as a field of band
rows when its structure allows it. Let `f` and `g` be fields on a column space
with `Nv` levels and element types `T_f`, `T_g`, and let `M` with `M_ij = ∂f_i/∂g_j`
be the `Nv × Nv` banded matrix of an entry.

### Scaling entries

When `M = k I` for a value `k` of type `T_k`, the entry

```julia
entry = fill(DiagonalMatrixRow(k), space)
```

is replaced by a single value, `entry = DiagonalMatrixRow(k)`, or, for a
scalar `k`, `entry = k * LinearAlgebra.I`. Both are `ScalingFieldMatrixEntry`s
and cut the memory by a factor of `Nv`.

### Implicit tensor structure

When `T_f = T_g` is a vector type and `∂f/∂g` is a multiple of the identity
tensor at every band, the tensor need not be stored. Writing `f_n[i]` for the
`i`th component at level `n`, take the tridiagonal example

```math
\frac{\partial f_n[i]}{\partial g_m[j]} = \begin{cases}
  -0.5, & \text{if } i = j \text{ and }  m = n-1 \text{ or } m = n+1 \\
  1, & \text{if } i = j \text{ and } m = n \\
  0, & \text{if } i \neq j \text{ or } m < n -1 \text{ or } m > n +1
\end{cases}
```

for `Covariant12Vector`s. Stored explicitly, each band holds the identity
tensor times a scalar:

```julia
∂f_∂g = fill(
    MatrixFields.TridiagonalMatrixRow(
        -0.5 * identity_axis2tensor,
        identity_axis2tensor,
        -0.5 * identity_axis2tensor,
    ),
    space,
)
J = MatrixFields.FieldMatrix((@name(f), @name(g)) => ∂f_∂g)
```

and indexing by component extracts the diagonal and off-diagonal blocks:

```@example 2
J[(@name(f.components.data.:(1)), @name(g.components.data.:(1)))]
```

```@example 2
J[(@name(f.components.data.:(2)), @name(g.components.data.:(1)))]
```

The same entry stored with scalar bands, by rule 6 of the indexing rules, is
read as the scalar times the identity tensor:

```@setup 2
∂f_∂g = fill(MatrixFields.TridiagonalMatrixRow(-0.5, 1.0, -0.5), space)
J = MatrixFields.FieldMatrix((@name(f), @name(g))=> ∂f_∂g)
```

```julia
∂f_∂g = fill(MatrixFields.TridiagonalMatrixRow(-0.5, 1.0, -0.5), space)
J = MatrixFields.FieldMatrix((@name(f), @name(g)) => ∂f_∂g)
```

```@example 2
J[(@name(f.components.data.:(1)), @name(g.components.data.:(1)))]
```

```@example 2
Base.Broadcast.materialize(
    J[(@name(f.components.data.:(2)), @name(g.components.data.:(1)))],
)
```

When in addition the scalar is the same at every level and only on the
diagonal, `∂f_n[i]/∂g_m[j] = k δ_ij δ_nm`, both optimizations apply and the
entry is `k * LinearAlgebra.I`.
