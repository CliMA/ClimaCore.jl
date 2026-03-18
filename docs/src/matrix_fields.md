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

## Internals

```@docs
outer_diagonals
band_matrix_row_type
matrix_shape
column_axes
AbstractLazyOperator
replace_lazy_operator
FieldName
@name
FieldNameTree
FieldNameSet
is_lazy
lazy_main_diagonal
lazy_mul
LazySchurComplement
field_matrix_solver_cache
check_field_matrix_solver
run_field_matrix_solver!
solver_algorithm
lazy_preconditioner
preconditioner_cache
check_preconditioner
lazy_or_concrete_preconditioner
apply_preconditioner
get_scalar_keys
field_offset_and_type
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

A FieldMatrix entry can be:

- A `UniformScaling`, which contains a `Number`
- A `DiagonalMatrixRow`, which can contain either a `Number` or a tensor (represented as a `Geometry.Axis2Tensor`)
- A `ColumnwiseBandMatrixField`, where each value is a [`BandMatrixRow`](@ref) with entries of any type that can be represented using the field's base number type.

If an entry contains a composite type, the fields of that type can be extracted.
This is also true for nested composite types.

For example:

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
           staggering = CellCenter() # hide
       ) # hide
nt_entry_field = fill(MatrixFields.DiagonalMatrixRow((; foo = 1.0, bar = 2.0)), space)
nt_fieldmatrix = MatrixFields.FieldMatrix((@name(a), @name(b)) => nt_entry_field)
nt_fieldmatrix[(@name(a), @name(b))]
```

The internal values of the named tuples can be extracted with

```@example 1
nt_fieldmatrix[(@name(a.foo), @name(b))]
```

and

```@example 1
nt_fieldmatrix[(@name(a.bar), @name(b))]
```

### Further Indexing Details

Let key `(@name(name1), @name(name2))` correspond to entry `sample_entry` in `FieldMatrix` `A`.
An example of this is:

```julia
 A = MatrixFields.FieldMatrix((@name(name1), @name(name2)) => sample_entry)
```

Now consider what happens when indexing `A` with the key `(@name(name1.foo.bar.buz), @name(name2.biz.bop.fud))`.

First, `getindex` finds a key in `A` that contains the key being indexed. In this example, `(@name(name1.foo.bar.buz), @name(name2.biz.bop.fud))` is contained within `(@name(name1), @name(name2))`, so `(@name(name1), @name(name2))` is called the "parent key" and `(@name(foo.bar.buz), @name(biz.bop.fud))` is referred to as the "internal key".

Next, the entry that `(@name(name1), @name(name2))` is paired with is recursively indexed
by the internal key.

The recursive indexing of an internal entry given some entry `entry` and internal key `internal_name_pair`
works as follows:

1. If the  `internal_name_pair` is blank, return `entry`
2. If the element type of each band of `entry` is an `Axis2Tensor`, and `internal_name_pair` is of the form `(@name(components.data.1...), @name(components.data.2...))` (potentially with different numbers), then extract the specified component, and recurse on it with the remaining `internal_name_pair`.
3. If the element type of each band of `entry` is a `Geometry.AdjointAxisVector`, then recurse on the parent of the adjoint.
4. If `internal_name_pair[1]` is not empty, and the first name in it is a field of the element type of each band of `entry`, extract that field from `entry`, and recurse into it with the remaining names of `internal_name_pair[1]` and all of `internal_name_pair[2]`
5. If `internal_name_pair[2]` is not empty, and the first name in it is a field of the element type of each band of `entry`, extract that field from `entry`, and recurse into it with all of `internal_name_pair[1]` and the remaining names of `internal_name_pair[2]`
6. At this point, if none of the previous cases are true, both `internal_name_pair[1]` and `internal_name_pair[2]` should be non-empty, and it is assumed that `entry` is being used to implicitly represent some tensor structure. If the first name in `internal_name_pair[1]` is equivalent to `internal_name_pair[2]`, then both the first names are dropped, and entry is recursed onto. If the first names are different, both the first names are dropped, and the zero of entry is recursed onto.

When the entry is a `ColumnWiseBandMatrixField`, indexing it will return a broadcasted object in
the following situations:

1. The internal key indexes to a type different than the basetype of the entry
2. The internal key indexes to a zero-ed value

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

## Optimizations

Each entry of a `FieldMatrix` can be a `ColumnwiseBandMatrixField`, a `DiagonalMatrixRow`, or an
`UniformScaling`.

A `ColumnwiseBandMatrixField` is a `Field` with a `BandMatrixRow` at each point. It is intended
to represent a collection of banded matrices, where there is one band matrix for each column
of the space the `Field` is on. Beyond only storing the diagonals of the band matrix, an `entry`
can be optimized to use less memory. Each optimized representation can be indexed equivalently to
non optimized representations, and used in addition, subtraction, matrix-vector multiplication,
matrix-matrix multiplication, and solving linear systems via `FieldMatrixSolver`.

For the following sections, `space` is a column space with $N_v$ levels. A column space is
used for simplicity in this example, but the optimizations work with any space with columns.

Let $f$ and $g$ be `Fields` on `space` with elements of type with elements of type
`T_f` and `T_g`. $f_i$ and $g_i$ refers to the values of $f$ and $g$ at the $ 0 < i \leq N_v$ level.

Let $M$ be a $N_v \times N_v$ banded matrix with lower and upper bandwidth of $b_1$ and $b_2$.
$M$ represents $\frac{\partial f}{\partial g}$, so $M_{i,j} = \frac{\partial f_i}{\partial g_j}$

### `ScalingFieldMatrixEntry` Optimization

Consider the case where $b_1 = 0$ and $b_2 = 0$, i.e $M$ is a diagonal matrix, and
where $M = k * I$, and $k$ is of type `T_k`. This would happen if
$\frac{\partial f_i}{\partial g_j} = \delta_{ij} * k$. Instead of storing
each element on the diagonal, the `FieldMatrix` can store a single value that represents a scaling of the identity matrix, reducing memory usage by a factor of $N_v$:

```julia
entry = fill(DiagonalMatrixRow(k), space)
```

can also be represented by

```julia
entry = DiagonalMatrixRow(k)
```

or, if `T_k` is a scalar, then

```julia
entry = I * k
```

### Implicit Tensor Structure Optimization

The functions that index an entry with an internal key assume the implicit tensor structure optimization is being used
when all of the following are true for `entry` where `T_k` is the element type of each band, and
`(internal_key_1, internal_key_2)` is the internal key indexing `entry`.

- the `internal_key_1` name chain is not empty and its first name is not a field of `T_k`
- the `internal_key_2` name chain is not empty and its first name is not a field of `T_k`

For most use cases, `T_k` is a scalar.

If the above conditions are met, the optimization assumes that the user intends the
entry to have an implicit tensor structure, with the values of type `T_k` representing a scaling of the
identity tensor. If both the first and second names in the name pair are equivalent, then they index onto the diagonal,
and the scalar value of `k` is returned. Otherwise, they index off the diagonal, and a zero value
is returned.

This optimization is intended to be used when `T_f = T_g`.
The notation $f_{n}[i]$ where $0 < n \leq N_v$  refers to the $i$-th component of the element
at the $n$-th vertical level of $f$. In the following example, `T_f` and `T_g` are both `Covariant12Vector`s, and
$b_1 = b_2 = 1$, and

```math
\frac{\partial f_n[i]}{\partial g_m[j]} = \begin{cases}
  -0.5, & \text{if } i = j \text{ and }  m = n-1 \text{ or } m = n+1 \\
  1, & \text{if } i = j \text{ and } m = n \\
  0, & \text{if } i \neq j \text{ or } m < n -1 \text{ or } m > n +1
\end{cases}
```

The non-zero values of each row of `M` are equivalent in this example, but they can also vary in value.

```julia
∂f_∂g = fill(MatrixFields.TridiagonalMatrixRow(-0.5 * identity_axis2tensor, identity_axis2tensor, -0.5 * identity_axis2tensor), space)
J = MatrixFields.FieldMatrix((@name(f), @name(g))=> ∂f_∂g)
```

`∂f_∂g` can be indexed into to get the partial derrivatives of individual components.


```@example 2
J[(@name(f.components.data.:(1)), @name(g.components.data.:(1)))]
```

```@example 2
J[(@name(f.components.data.:(2)), @name(g.components.data.:(1)))]
```

This can be more optimally stored with the implicit tensor structure optimization:

```@setup 2
∂f_∂g = fill(MatrixFields.TridiagonalMatrixRow(-0.5, 1.0, -0.5), space)
J = MatrixFields.FieldMatrix((@name(f), @name(g))=> ∂f_∂g)
```

```julia
∂f_∂g = fill(MatrixFields.TridiagonalMatrixRow(-0.5, 1.0, -0.5), space)
J = MatrixFields.FieldMatrix((@name(f), @name(g))=> ∂f_∂g)
```

```@example 2
J[(@name(f.components.data.:(1)), @name(g.components.data.:(1)))]
```

```@example 2
Base.Broadcast.materialize(J[(@name(f.components.data.:(2)), @name(g.components.data.:(1)))])
```

If it is the case that

```math
\frac{\partial f_n[i]}{\partial g_m[j]} = \begin{cases}
  k, & \text{if } i = j \text{ and } m = n \\
  0, & \text{if } i \neq j \text{ or } m \neq n
\end{cases}
```

where $k$ is a constant scalar, the implicit tensor structure optimization and
`ScalingFieldMatrixEntry` optimization can both be applied.
