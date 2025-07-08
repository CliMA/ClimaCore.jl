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

- An `UniformScaling`, which contains a `Number`
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

Now consider what happens indexing `A` with the key `(@name(name1.foo.bar.buz), @name(name2.biz.bop.fud))`.

First, `getindex` finds a key in `A` that contains the key being indexed. In this example, `(@name(name1.foo.bar.buz), @name(name2.biz.bop.fud))` is contained within `(@name(name1), @name(name2))`, so `(@name(name1), @name(name2))` is called the "parent key" and `(@name(foo.bar.buz), @name(biz.bop.fud))` is referred to as the "internal key".

Next, the entry that `(@name(name1), @name(name2))` is paired with is recursively indexed
by the internal key.

The recursive indexing of an internal entry given some entry `entry` and internal_key `internal_name_pair`
works as follows:

1. If the  `internal_name_pair` is blank, return `entry`
2. If the element type of each band of `entry` is an `Axis2Tensor`, and `internal_name_pair` is of the form
`(@name(components.data.1...), @name(components.data.2...))` (potentially with different numbers),
then extract the specified component, and recurse on it with the remaining `internal_name_pair`.
3. If the element type of each band of `entry` is a `Geometry.AdjointAxisVector`, then recurse on the parent of the adjoint.
4. If `internal_name_pair[1]` is not empty, and the first name in it is a field of the element type of each band of `entry`,
extract that field from `entry`, and recurse into it with the remaining names of `internal_name_pair[1]` and all of `internal_name_pair[2]`
5. If `internal_name_pair[2]` is not empty, and the first name in it is a field of the element type of each band of `entry`,
extract that field from `entry`, and recurse into it with all of `internal_name_pair[1]` and the remaining names of `internal_name_pair[2]`
6. At this point, if none of the previous cases are true, both `internal_name_pair[1]` and `internal_name_pair[2]` should be
non-empty, and it is assumed that `entry` is being used to implicitly represent some tensor structure. If the first name in
`internal_name_pair[1]` is equivalent to `internal_name_pair[2]`, then both the first names are dropped, and entry is recursed onto.
If the first names are different, both the first names are dropped, and the zero of entry is recursed onto.

When the entry is a `ColumnWiseBandMatrixField`, indexing it will return a broadcasted object in
the following situations:

1. The internal key indexes to a type different than the basetype of the entry
2. The internal key indexes to a zero-ed value
3. The internal key slices an `AxisTensor`

### Implicit Tensor Structure Optimization

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
```

If using a `FieldMatrix` to represent a jacobian, entries with certain structures
can be stored in an optimized manner.

The optimization assumes that if indexing into an entry of scalars, the user intends the
entry to have an implicit tensor structure, with the scalar values representing a scaling of the
tensor identity. If both the first and second name in the name pair are equivalent, then they index onto the diagonal,
and the scalar value is returned. Otherwise, they index off the diagonal, and a zero value
is returned.

The following sections refer the `Field`s
$f$ and $g$, which both have values of type `Covariant12Vector` and are defined on a column domain, which is discretized with $N_v$ layers.
The notation $f_{n}[i]$ where $ 0 < n \leq N_v$ and $i \in (1,2)$ refers to the $i$ component of the element of $f$
at the $i$ vertical level. $g$ is indexed similarly. Although $f$ and $g$ have values of type
`Covariant12Vector`, this optimization works for any two `Field`s of `AxisVector`s

```@example 2
f = map(x -> rand(Geometry.Covariant12Vector{Float64}), Fields.local_geometry_field(space))
g = map(x -> rand(Geometry.Covariant12Vector{Float64}), Fields.local_geometry_field(space))
```

#### Uniform Scaling Case

If $\frac{\partial f_n[i]}{\partial g_n[j]} = [i = j]$ for some scalar $k$, then the
non-optimized entry would be represented by a diagonal matrix with values of an identity 2d tensor. If $k=2$, then

```@example 2
identity_axis2tensor = Geometry.Covariant12Vector(FT(1), FT(0)) * # hide
                   Geometry.Contravariant12Vector(FT(1), FT(0))' + # hide
                   Geometry.Covariant12Vector(FT(0), FT(1)) * # hide
                   Geometry.Contravariant12Vector(FT(0), FT(1))' # hide
k = 2
∂f_∂g = fill(MatrixFields.DiagonalMatrixRow(k * identity_axis2tensor), space)
```

Individual components can be indexed into:

```@example 2
J = MatrixFields.FieldMatrix((@name(f), @name(g))=> ∂f_∂g)
J[[(@name(f.components.data.:(1)), @name(g.components.data.:(1)))]]
```

```@example 2
J[[(@name(f.components.data.:(2)), @name(g.components.data.:(1)))]]
```

The above example indexes into $\frac{\partial f_n[1]}{\partial g_n[1]}$ where $ 0 < n \leq N_v$

The entry can
also be represeted with a single `DiagonalMatrixRow`, as follows:

```@example 2
∂f_∂g_optimized = MatrixFields.DiagonalMatrixRow(k * identity_axis2tensor)
```

`∂f_∂g_optimized` is a single `DiagonalMatrixRow`, which represents a diagonal matrix with the
same tensor along the diagonal. In this case, that tensor is $k$ multiplied by the identity matrix, and that can be
represented with `k * I` as follows

```@example 2
∂f_∂g_more_optimized = MatrixFields.DiagonalMatrixRow(k * identity_axis2tensor)
```

Individual components of `∂f_∂g_optimized` can be indexed in the same way as `∂f_∂g`.

```@example 2
J_unoptimized = MatrixFields.FieldMatrix((@name(f), @name(g)) => ∂f_∂g)
J_unoptimized[(@name(f.components.data.:(1)), @name(g.components.data.:(1)))]
```

```@example 2
J_more_optimized = MatrixFields.FieldMatrix((@name(f), @name(g)) => ∂f_∂g_optimized)
J_more_optimized[(@name(f.components.data.:(1)), @name(g.components.data.:(1)))]
```

```@example 2
J_more_optimized[(@name(f.components.data.:(1)), @name(g.components.data.:(2)))]
```

`∂f_∂g` stores $2 * 2 * N_v$ floats in memory, `∂f_∂g_optimized` stores `$2*2$ floats, and
`∂f_∂g_more_optimized` stores only one float.

#### Vertically Varying Case

The implicit tensor optimization can also be used when
$\frac{\partial f_n[i]}{\partial g_n[j]} = [i = j] * h(f_n, g_n)$.

In this case, a full `ColumnWiseBandMatrixField` must be used.

```@example 2
∂f_∂g_optimized = map(x -> MatrixFields.DiagonalMatrixRow(rand(Float64)), ∂f_∂g)
```

```@example 2
J_optimized = MatrixFields.FieldMatrix((@name(f), @name(g)) => ∂f_∂g_optimized)
J_optimized[(@name(f.components.data.:(1)), @name(g.components.data.:(1)))]
```

```@example 2
Base.Broadcast.materialize(J_optimized[(@name(f.components.data.:(2)), @name(g.components.data.:(1)))])
```

#### bandwidth > 1 case

The implicit tensor optimization can also be used when
$\frac{\partial f_n[i]}{\partial g[j]} = [i = j] * h(f_n, g_{n-k_1}, ..., g_{n+k_2})$ where
$b_1$ and $b_2$ are the lower and upper bandwidth. Say $b_1 = b_2 = 1$. Then

```@example 2
∂f_∂g_optimized = map(x -> MatrixFields.TridiagonalMatrixRow(rand(Float64), rand(Float64), rand(Float64)), ∂f_∂g)
```

```@example 2
J_optimized = MatrixFields.FieldMatrix((@name(f), @name(g)) => ∂f_∂g_optimized)
J_optimized[(@name(f.components.data.:(1)), @name(g.components.data.:(1)))]
```

```@example 2
Base.Broadcast.materialize(J_optimized[(@name(f.components.data.:(2)), @name(g.components.data.:(1)))])
```
