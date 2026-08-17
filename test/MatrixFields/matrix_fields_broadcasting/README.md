# MatrixFields broadcasting tests

Each `test_*.jl` file in this folder checks a single broadcast expression over
matrix fields (fields of `BandMatrixRow`s): it materializes the expression,
compares the result against an independent reference implementation, and — when
the opt gate is enabled (`CLIMACORE_TEST_OPT=true`, or by default on CI) — also
runs JET type-stability checks and allocation gates on it.

The files are split one-expression-per-file because each expression compiles a
distinct broadcast kernel; keeping them separate bounds per-process compilation
memory and lets CI parallelize or subset them. They share their setup via
`test_scalar_utils.jl` / `test_non_scalar_utils.jl` and are run together by two
umbrella files in the parent folder:

 - `unit_matrix_field_broadcasting.jl` (CPU, all files),
 - `gpu_matrix_field_broadcasting.jl` (GPU).

The `scalar` / `non_scalar` split refers to the **entry type of the matrix
rows**: `test_scalar_*` files use matrices of numbers, while `test_non_scalar_*`
files use matrices whose entries are vectors, covectors, tuples, or nested
`NamedTuple`s (the hard cases for broadcast inference and GPU compilation).

Field-name legend (defined in the two `*_utils.jl` files): `ᶜ`/`ᶠ` prefixes
denote cell-center/cell-face spaces, so e.g. `ᶜᶠmat` is a center-to-face matrix
(bidiagonal), `ᶠᶜmat` face-to-center (quad-diagonal), `ᶜᶜmat` diagonal, `ᶠᶠmat`
tri-diagonal, and `ᶜvec`/`ᶠvec` are plain vectors (fields). Suffixes give the
entry type: `_AC1` = adjoint `Covariant1Vector` (covector), `_C12` =
`Covariant12Vector`, `_num` pairs entries with numbers in tuples, `_NT` =
nested `NamedTuple` entries.

## Scalar-entry expressions

| file | tested expression |
|------|-------------------|
| `test_scalar_1.jl`  | `ᶜᶜmat * ᶜvec` — diagonal matrix times vector |
| `test_scalar_2.jl`  | `ᶠᶠmat * ᶠvec` — tri-diagonal matrix times vector |
| `test_scalar_3.jl`  | `ᶠᶜmat * ᶜvec` — quad-diagonal matrix times vector |
| `test_scalar_4.jl`  | `ᶜᶜmat * ᶜᶠmat` — diagonal times bi-diagonal |
| `test_scalar_5.jl`  | `ᶠᶠmat * ᶠᶠmat` — tri-diagonal times tri-diagonal |
| `test_scalar_6.jl`  | `ᶠᶜmat * ᶜᶜmat` — quad-diagonal times diagonal |
| `test_scalar_7.jl`  | `ᶜᶜmat * ᶜᶠmat * ᶠᶠmat * ᶠᶜmat` — four-matrix product, left-associated |
| `test_scalar_8.jl`  | `ᶜᶜmat * (ᶜᶠmat * (ᶠᶠmat * ᶠᶜmat))` — same product, right-associated (**expected `InvalidIRError` on CUDA**, asserted with `@test_throws`) |
| `test_scalar_9.jl`  | `ᶜᶜmat * ᶜᶠmat * ᶠᶠmat * ᶠᶜmat * ᶜvec` — four matrices times a vector, left-associated |
| `test_scalar_10.jl` | `ᶜᶜmat * (ᶜᶠmat * (ᶠᶠmat * (ᶠᶜmat * ᶜvec)))` — same, right-associated (**expected `InvalidIRError` on CUDA**, asserted with `@test_throws`) |
| `test_scalar_11.jl` | `2 * ᶠᶜmat * ᶜᶜmat * ᶜᶠmat + ᶠᶠmat * ᶠᶠmat / 3 - (4I,)` — linear combination of matrix products and `LinearAlgebra.I` |
| `test_scalar_12.jl` | `ᶠᶜmat * ᶜᶜmat * ᶜᶠmat * 2 - (ᶠᶠmat / 3) * ᶠᶠmat + (4I,)` — as 11, with scalars/`I` placed differently |
| `test_scalar_13.jl` | `ᶜᶠmat * (2 * ᶠᶜmat * ᶜᶜmat * ᶜᶠmat + ᶠᶠmat * ᶠᶠmat / 3 - (4I,))` — matrix times a linear combination |
| `test_scalar_14.jl` | `(lin. comb. of 11) * (lin. comb. of 12)` — product of two linear combinations |
| `test_scalar_15.jl` | `ᶠᶜmat * ᶜᶠmat * (lin. comb. of 11) * ᶠᶠmat * (lin. comb. of 12) * ᶠᶠmat` — matrices interleaved with linear combinations |
| `test_scalar_16.jl` | `BidiagonalMatrixRow(ᶜᶠmat * ᶠvec, ᶜᶜmat * ᶜvec) * TridiagonalMatrixRow(ᶠvec, ᶠᶜmat * ᶜvec, 1) * ᶠᶠmat * DiagonalMatrixRow(DiagonalMatrixRow(ᶠvec) * ᶠvec)` — matrix rows constructed inside the broadcast |
| `test_scalar_17.jl` | `ᶠᶠmat * DiagonalMatrixRow(1.0f0 - 0.5f0 * 2.0f0 + 0.0f0 * ᶠvec)` — mixed-precision scalars nested in a row constructor |

## Non-scalar-entry expressions

| file | tested expression |
|------|-------------------|
| `test_non_scalar_1.jl` | `ᶜᶠmat_AC1 * ᶠᶜmat_C12` — matrix of covectors times matrix of vectors |
| `test_non_scalar_2.jl` | `ᶜᶠmat_AC1 * ᶠᶜmat_C12 * ᶜᶠmat * ᶠᶜmat_AC1 * ᶜᶠmat_C12` — five-matrix chain mixing covector, vector, and scalar entries |
| `test_non_scalar_3.jl` | `ᶜᶠmat_AC1_num * ᶠᶜmat_C12_AC1 * ᶜᶠmat_num_C12 * ᶠvec` — tuple-valued entries (covector/number, vector/covector, number/vector pairs) |
| `test_non_scalar_4.jl` | `ᶜᶠmat_NT * ᶠᶜmat * ᶜᶠmat * ᶠᶜmat_NT * ᶜvec_NT` — nested-`NamedTuple` entries through a matrix chain |
| `test_non_scalar_5.jl` | `ᶜᶠmat_C12 / 2` — matrix of vectors divided by a scalar |
