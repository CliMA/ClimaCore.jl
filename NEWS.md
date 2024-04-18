ClimaCore.jl Release Notes
========================

main
-------

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
