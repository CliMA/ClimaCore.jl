ClimaCore.jl Release Notes
========================

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
