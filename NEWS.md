ClimaCore.jl Release Notes
========================

main
-------

v0.15.0
-------

- ![][badge-💥breaking] support for `IntervalTopology(::Mesh)` has been dropped in favor of using `IntervalTopology(::ClimaComms.AbstractDevice, ::Mesh)`. PR [#1821](https://github.com/CliMA/ClimaCore.jl/pull/1821).

v0.14.9
-------

- ![][badge-🐛bugfix] GPU dispatching with `copyto!` and `fill!` have been fixed PR [#1802](https://github.com/CliMA/ClimaCore.jl/pull/1802).

v0.14.8
-------

- ![][badge-✨feature/enhancement] Added `FieldMatrixWithSolver`, a wrapper that helps defining implicit Jacobians. PR [#1788](https://github.com/CliMA/ClimaCore.jl/pull/1788)


v0.14.6
-------

- ![][badge-✨feature/enhancement] Added `array2field(::Field)` and `field2array(::Field)` convenience functions, to help facilitate use with RRTMGP. PR [#1768](https://github.com/CliMA/ClimaCore.jl/pull/1768)

- ![][badge-🚀performance] `Nv` is now a type parameter in DataLayouts that have vertical levels. As a result, users can use `DataLayouts.nlevels(::AbstractData)` to obtain a compile-time constant for the number of vertical levels.

- ![][badge-✨feature/enhancement] Added `interpolate(field, target_hcoords,
  target_zcoord)` convenience function so that the `Remapper` does not have to
  be explicitely constructed. PR
  [#1764](https://github.com/CliMA/ClimaCore.jl/pull/1764)

v0.14.5
-------

- ![][badge-🐛bugfix] `run_field_matrix_solver!` was fixed for column spaces, and tests were added to ensure it doesn't break in the future.
  PR [#1750](https://github.com/CliMA/ClimaCore.jl/pull/1750)
- ![][badge-🚀performance] We're now using local memory (MArrays) in the `band_matrix_solve!`, which has improved performance. PR [#1735](https://github.com/CliMA/ClimaCore.jl/pull/1735).
- ![][badge-🚀performance] We've specialized some cases in `run_field_matrix_solver!`, which results in more efficient kernels being launched. PR [#1732](https://github.com/CliMA/ClimaCore.jl/pull/1732).
- ![][badge-🚀performance] We've reduced memory reads in the `band_matrix_solve!` for tridiagonal systems, improving its performance. PR [#1731](https://github.com/CliMA/ClimaCore.jl/pull/1731).
- ![][badge-🚀performance] We've added NVTX annotations in ClimaCore functions, so that we have a more granular trace of performance. PRs [#1726](https://github.com/CliMA/ClimaCore.jl/pull/1726), [#1723](https://github.com/CliMA/ClimaCore.jl/pull/1723).

v0.14.0
-------

- ![][badge-🐛bugfix] Extend adapt_structure for all operator and boundary
  condition types. Also use `unrolled_map` in `multiply_matrix_at_index` to
  avoid the recursive inference limit when compiling nested matrix operations.
  PR [#1684](https://github.com/CliMA/ClimaCore.jl/pull/1684)
- ![][badge-🤖precisionΔ] ![][badge-💥breaking] `Remapper`s can now process
  multiple `Field`s at the same time if created with some `buffer_lenght > 1`.
  PR ([#1669](https://github.com/CliMA/ClimaCore.jl/pull/1669))
  Machine-precision differences are expected. This change is breaking because
  remappers now return the same array type as the input field.
- ![][badge-🚀performance] We inlined the `multiple_field_solve` kernels, which should improve performance. PR [#1715](https://github.com/CliMA/ClimaCore.jl/pull/1715).
- ![][badge-🚀performance] We added support for MultiBroadcastFusion, which allows users to fuse similar space point-wise broadcast expressions via `Fields.@fused_direct`. PR [#1641](https://github.com/CliMA/ClimaCore.jl/pull/1641).

v0.13.4
-------

- ![][badge-🐛bugfix] We fixed some fieldvector broadcasting on Julia 1.9. PR [#1658](https://github.com/CliMA/ClimaCore.jl/pull/1658).
- ![][badge-🚀performance] We fixed an inference failure with matrix field broadcasting. PR [#1683](https://github.com/CliMA/ClimaCore.jl/pull/1683).

v0.13.3
-------

- ![][badge-🚀performance] We now always inline for all ClimaCore kernels. PR [#1647](https://github.com/CliMA/ClimaCore.jl/pull/1647). This can result in more brittle inference (due to compiler heuristics). Technically, this is not a breaking change, but some code changes may be needed in practice.

v0.13.2
-------

- ![][badge-🐛bugfix] fixed array allocation for interpolation on CPU.
  PR [#1643](https://github.com/CliMA/ClimaCore.jl/pull/1643).

v0.13.1
-------

- ![][badge-🐛bugfix] fixed edge case in interpolation that led to incorrect
  vertical interpolation.
  PR [#1640](https://github.com/CliMA/ClimaCore.jl/pull/1640).
- ![][badge-🐛bugfix] fixed `interpolate!` for MPI runs.
  PR [#1642](https://github.com/CliMA/ClimaCore.jl/pull/1642).


v0.13.0
-------
- ![][badge-💥breaking] support for many deprecated methods have been dropped PR [#1632](https://github.com/CliMA/ClimaCore.jl/pull/1632).
- ![][badge-🤖precisionΔ]![][badge-🚀performance] Slight performance improvement by replacing `rdiv` with `rmul`. PR ([#1496](https://github.com/CliMA/ClimaCore.jl/pull/1496)) Machine-precision differences are expected.
- ![][badge-💥breaking]![][badge-🚀performance] Rewritten `distributed_remapping`. New `distributed_remapping` is non-allocating and up to 1000x faster (on GPUs). New `distributed_remapping` no longer supports the `physical_z` argument (this option is still available in `Remapping.interpolate_column`). New `interpolate!` function is available for remapping in-place. The new preferred way to define a `Rampper` is `Remapper(space, target_hcoords, target_zcoords)` (instead of `Remapper(target_hcoords, target_zcoords, space)`).
  PR ([#1630](https://github.com/CliMA/ClimaCore.jl/pull/1630))

v0.12.1
-------
- Started changelog
- Fixed matrix field iterative solver tests.
- ![][badge-🚀performance] Specialize on diagonal fieldvector broadcasts to skip uninferred `check_broadcast_axes` PR [#1615](https://github.com/CliMA/ClimaCore.jl/pull/1615), Issue [#1465](https://github.com/CliMA/ClimaCore.jl/issues/1465).
- ![][badge-🚀performance] Fixed inference errors when not debugging PR [#1617](https://github.com/CliMA/ClimaCore.jl/pull/1617), Issue [#2597](https://github.com/CliMA/ClimaAtmos.jl/issues/2597).

<!--

Contributors are welcome to begin the description of changelog items with badge(s) below. Here is a brief description of when to use badges for a particular pull request / set of changes:

 - 🔥behavioralΔ - behavioral changes. For example: a new model is used, yielding more accurate results.
 - 🤖precisionΔ - machine-precision changes. For example, swapping the order of summed arguments can result in machine-precision changes.
 - 💥breaking - breaking changes. For example: removing deprecated functions/types, removing support for functionality, API changes.
 - 🚀performance - performance improvements. For example: improving type inference, reducing allocations, or code hoisting.
 - ✨feature - new feature added. For example: adding support for a cubed-sphere grid
 - 🐛bugfix - bugfix. For example: fixing incorrect logic, resulting in incorrect results, or fixing code that otherwise might give a `MethodError`.

-->

[badge-🔥behavioralΔ]: https://img.shields.io/badge/🔥behavioralΔ-orange.svg
[badge-🤖precisionΔ]: https://img.shields.io/badge/🤖precisionΔ-black.svg
[badge-💥breaking]: https://img.shields.io/badge/💥BREAKING-red.svg
[badge-🚀performance]: https://img.shields.io/badge/🚀performance-green.svg
[badge-✨feature/enhancement]: https://img.shields.io/badge/feature/enhancement-blue.svg
[badge-🐛bugfix]: https://img.shields.io/badge/🐛bugfix-purple.svg
