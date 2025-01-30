ClimaCore.jl Release Notes
========================

main
-------

- Fixed bug in distributed remapping with CUDA. Sometimes, `ClimaCore` would not
  properly fill the output arrays with the correct values. This is now fixed. PR
  [2169](https://github.com/CliMA/ClimaCore.jl/pull/2169)

v0.14.24
-------

 - A new `Adapt` wrapper was added, `to_device`, which allows users to adapt datalayouts, spaces, fields, and fieldvectors between the cpu and gpu. PR [2159](https://github.com/CliMA/ClimaCore.jl/pull/2159).
 - Remap interpolation cuda threading was improved. PR [2159](https://github.com/CliMA/ClimaCore.jl/pull/2159).
 - `center_space` and `face_space` are now exported from `CommonSpaces`. PR [2157](https://github.com/CliMA/ClimaCore.jl/pull/2157).
 - Limiter debug printing can now be suppressed, and printed at the end of a simulation, using `Limiters.print_convergence_stats(limiter)`. PR [2152](https://github.com/CliMA/ClimaCore.jl/pull/2152).
 - Fixed getidx for `GradientC2F` with `SetValue` bcs. PR [2148](https://github.com/CliMA/ClimaCore.jl/pull/2148).
 - Added support `data2array` for `DataF` (i.e., `PointField`s). PR [2143](https://github.com/CliMA/ClimaCore.jl/pull/2143).
 - `HDF5Reader` / `HDF5Writer` now support `do`-syntax. PR [2147](https://github.com/CliMA/ClimaCore.jl/pull/2147).

v0.14.22
-------

 - Added support for CPU<->GPU conversion of spaces, fields and fieldvectors, via Adapt. PR [2114](https://github.com/CliMA/ClimaCore.jl/pull/2114).

 - Fixed numerics of column integrals for deep atmosphere. PR [2119](https://github.com/CliMA/ClimaCore.jl/pull/2119).

 - Fixed high-resolution gpu space construction. PR [2100](https://github.com/CliMA/ClimaCore.jl/pull/2100).

### ![][badge-✨feature/enhancement] A new DebugOnly module can help find where NaNs/Inf come from
 - PRs [2115](https://github.com/CliMA/ClimaCore.jl/pull/2115) and
 [2139](https://github.com/CliMA/ClimaCore.jl/pull/2139)

A new `ClimaCore.DebugOnly` module was added, which can help users find where
NaNs or Infs come from in a simulation, interactively. Documentation, with a
simple representative example can be found [here](https://clima.github.io/ClimaCore.jl/dev/debugging/#Infiltrating).

### ![][badge-✨feature/enhancement] Various improvements to `Remapper` [2060](https://github.com/CliMA/ClimaCore.jl/pull/2060)

The `ClimaCore.Remapping` module received two improvements. First, `Remapper` is
now compatible with purely vertical `Space`s (performing a linear
interpolation), making it compatible with column setups. Second, a new set of
simplified interpolation functions are provided.

Now, interpolating a `Field` `field` is as easy as
```julia
import ClimaCore.Remapping: interpolate
output_array = interpolate(field)
```
The target coordinates are automatically determined, but can also be customized.
Refer to the [documentation](https://clima.github.io/ClimaCore.jl/dev/remapping/)
for more information.

v0.14.22
-------

 - Fixed gpu support (adapt) for van-leer limiters. PR [2112](https://github.com/CliMA/ClimaCore.jl/pull/2112).

v0.14.21
--------

 - Support for new TVD limiters were added, PR [1662]
   (https://github.com/CliMA/ClimaCore.jl/pull/1662).

### ![][badge-🐛bugfix] Bug fixes

- Fixed writing/reading purely vertical spaces. PR [2102](https://github.com/CliMA/ClimaCore.jl/pull/2102)
- Fixed correctness bug in reductions on GPUs. PR [2106](https://github.com/CliMA/ClimaCore.jl/pull/2106)

### ![][badge-✨feature/enhancement] `face_space`, `center_space` functions

`ClimaCore.Spaces` now comes with two functions, `face_space` and
`center_space`, to convert a `Space` from being cell-centered to be
face-centered (and viceversa). These functions only work for vertical and
extruded spaces.

- We've added new convenience constructors for spaces PR [2082](https://github.com/CliMA/ClimaCore.jl/pull/2082). Here are links to the new constructors:
  - [ExtrudedCubedSphereSpace](https://clima.github.io/ClimaCore.jl/dev/api/#ClimaCore.CommonSpaces.ExtrudedCubedSphereSpace)
  - [CubedSphereSpace](https://clima.github.io/ClimaCore.jl/dev/api/#ClimaCore.CommonSpaces.CubedSphereSpace)
  - [ColumnSpace](https://clima.github.io/ClimaCore.jl/dev/api/#ClimaCore.CommonSpaces.ColumnSpace)
  - [Box3DSpace](https://clima.github.io/ClimaCore.jl/dev/api/#ClimaCore.CommonSpaces.Box3DSpace)
  - [SliceXZSpace](https://clima.github.io/ClimaCore.jl/dev/api/#ClimaCore.CommonSpaces.SliceXZSpace)
  - [RectangleXYSpace](https://clima.github.io/ClimaCore.jl/dev/api/#ClimaCore.CommonSpaces.RectangleXYSpace)

v0.14.20
--------

 - We've added new convenience constructors for grids PR [1848](https://github.com/CliMA/ClimaCore.jl/pull/1848). Here are links to the new constructors:
   - [ExtrudedCubedSphereGrid](https://github.com/CliMA/ClimaCore.jl/blob/cbb193042fac3b4bef33251fbc0f232427bfe506/src/CommonGrids/CommonGrids.jl#L85-L144)
   - [CubedSphereGrid](https://github.com/CliMA/ClimaCore.jl/blob/cbb193042fac3b4bef33251fbc0f232427bfe506/src/CommonGrids/CommonGrids.jl#L200-L235)
   - [ColumnGrid](https://github.com/CliMA/ClimaCore.jl/blob/cbb193042fac3b4bef33251fbc0f232427bfe506/src/CommonGrids/CommonGrids.jl#L259-L281)
   - [Box3DGrid](https://github.com/CliMA/ClimaCore.jl/blob/cbb193042fac3b4bef33251fbc0f232427bfe506/src/CommonGrids/CommonGrids.jl#L303-L378)
   - [SliceXZGrid](https://github.com/CliMA/ClimaCore.jl/blob/cbb193042fac3b4bef33251fbc0f232427bfe506/src/CommonGrids/CommonGrids.jl#L441-L498)
   - [RectangleXYGrid](https://github.com/CliMA/ClimaCore.jl/blob/cbb193042fac3b4bef33251fbc0f232427bfe506/src/CommonGrids/CommonGrids.jl#L547-L602)

 - A `strict = true` keyword was added to `rcompare`, which checks that the types match. If `strict = false`, then `rcompare` will return `true` for `FieldVector`s and `NamedTuple`s with the same properties but permuted order. For example:
     - `rcompare((;a=1,b=2), (;b=2,a=1); strict = true)` will return `false` and
     - `rcompare((;a=1,b=2), (;b=2,a=1); strict = false)` will return `true`
 - We've added new datalayouts: `VIJHF`,`IJHF`,`IHF`,`VIHF`, to explore their performance compared to our existing datalayouts: `VIJFH`,`IJFH`,`IFH`,`VIFH`. PR [#2055](https://github.com/CliMA/ClimaCore.jl/pull/2053), PR [#2052](https://github.com/CliMA/ClimaCore.jl/pull/2055).
 - We've refactored some modules to use less internals. PR [#2053](https://github.com/CliMA/ClimaCore.jl/pull/2053), PR [#2052](https://github.com/CliMA/ClimaCore.jl/pull/2052), [#2051](https://github.com/CliMA/ClimaCore.jl/pull/2051), [#2049](https://github.com/CliMA/ClimaCore.jl/pull/2049).
 - Some work was done in attempt to reduce specializations and compile time. PR [#2042](https://github.com/CliMA/ClimaCore.jl/pull/2042), [#2041](https://github.com/CliMA/ClimaCore.jl/pull/2041)

### ![][badge-🐛bugfix] Fix lower compat bounds

`ClimaCore` had incorrect lower bounds for certain packages. PR
[#2078](https://github.com/CliMA/ClimaCore.jl/pull/2078) fixes the lower bounds
and adds a GitHub Action workflow to test it. `ClimaCore` now requires Julia
1.10 or greater.

v0.14.19
-------

 - Fixed world-age issue on Julia 1.11 issue [Julia#54780](https://github.com/JuliaLang/julia/issues/54780), PR [#2034](https://github.com/CliMA/ClimaCore.jl/pull/2034).

### ![][badge-🐛bugfix] Fix undefined behavior in `DataLayout`s

PR [#2034](https://github.com/CliMA/ClimaCore.jl/pull/2034) fixes some undefined
behavior in the `DataLayout` module. This bug was manifesting itself as a `world
age` error in some applications that are using Julia 1.11.

### ![][badge-✨feature/enhancement] New convenience constructors for `DataLayout`s

PR [#2033](https://github.com/CliMA/ClimaCore.jl/pull/2033) introduces new
constructors for `DataLayout`s. Instead of writing
```julia
array = rand(FT, Nv, Nij, Nij, 3, Nh)
data = VIJFH{S, Nv, Nij}(array)
```

You can now write
```julia
data = VIJFH{S}(ArrayType{FT}, rand; Nv, Nij, Nh)
```
and grab the `array` with `parent(data)` (if you need).

Note: These constructors are meant to be used in tests and interactive use, not
in performance sensitive modules (due to their lack of inferrability).

v0.14.18
-------

 - Fixed multiple-field solve for land simulations PR [#2025](https://github.com/CliMA/ClimaCore.jl/pull/2025).
 - Fixed Julia 1.11 PR [#2018](https://github.com/CliMA/ClimaCore.jl/pull/2018).
 - `Nh` was turned back into a dynamic parameter, in order to alleviate compile times PR [#2005](https://github.com/CliMA/ClimaCore.jl/pull/2005).
 - Defined some convenience methods [#2012](https://github.com/CliMA/ClimaCore.jl/pull/2012)

### ![][badge-🐛bugfix] Fix equality for `FieldVector`s with different type

Due to a bug, `==` was not recursively checking `FieldVector`s with different
types, which resulted in false positives. This is now fixed and `FieldVector`s
with different types are always considered different.

### ![][badge-🐛bugfix] Fix restarting simulations from `Space`s with `deep = true`

Prior to this change, the `ClimaCore.InputOutput` module did not save whether a
`Space` was constructed with `deep = true`. This meant that restarting a
simulation from a HDF5 file led to inconsistent and incorrect spaces and
`Field`s. This affected only extruded 3D spectral spaces.

We now expect `Space`s read from a file to be bitwise identical to the original
one.


PR [#2021](https://github.com/CliMA/ClimaCore.jl/pull/2021).

v0.14.17
-------

 - Fixed some type instabilities PR [#2004](https://github.com/CliMA/ClimaCore.jl/pull/2004)
 - More fixes to higher resolution column cases for the GPU [#1854](https://github.com/CliMA/ClimaCore.jl/pull/1854)

v0.14.16
-------

- Extended `create_dss_buffer` and `weighted_dss!` for `FieldVector`s, rather than
just `Field`s. PR [#2000](https://github.com/CliMA/ClimaCore.jl/pull/2000).

- ![][badge-🐛bugfix] Fix restarting simulations from `Space`s with `enable_bubble = true`

Prior to this change, the `ClimaCore.InputOutput` module did not save whether a
`Space` was constructed with `enable_bubble = true`. This meant that restarting
a simulation from a HDF5 file led to inconsistent and incorrect spaces and
`Field`s. This affected only 2D spectral spaces (and extruded ones that have
this type of horizontal space).

We now expect `Space`s read from a file to be bitwise identical to the original
one.

PR [#1999](https://github.com/CliMA/ClimaCore.jl/pull/1999).

v0.14.15
-------

- Added support for mixing extruded and horizontal spaces in GPU kernels. PR [#1987](https://github.com/CliMA/ClimaCore.jl/pull/1987).

v0.14.14
-------

- Inference was fixed for some broadcast expressions involving columns PR [#1984](https://github.com/CliMA/ClimaCore.jl/pull/1984).

v0.14.13
-------

- CUDA kernel launch configurations have been tuned to improve performance, and now allows for high resolution in the vertical direction PR [#1969](https://github.com/CliMA/ClimaCore.jl/pull/1969), issue [#1854](https://github.com/CliMA/ClimaCore.jl/issues/1854) closed.

- DSS was refactored, and machine precision changes can be expected. PR [#1958](https://github.com/CliMA/ClimaCore.jl/pull/1958).

v0.14.12
-------
- Added hyperbolic tangent stretching. PR [#1930](https://github.com/CliMA/ClimaCore.jl/pull/1930).

v0.14.11
-------

- Support for matrix fields on spectral and point spaces was added, PR [#1884](https://github.com/CliMA/ClimaCore.jl/pull/1884).
- Support for 3-component DSS transform was added, PR [#1693](https://github.com/CliMA/ClimaCore.jl/pull/1693).
- Support for column-wise "accumulate"/"reduce" operations were added, PR [#1903](https://github.com/CliMA/ClimaCore.jl/pull/1903). These abstractions will allow us to group, paralellize and optimize more column-wise work on the GPU.
- A new macro, `Fields.@rprint_diff` was added, which recursively print differences between two `FieldVector`s (of the same type) (PR [#1886](https://github.com/CliMA/ClimaCore.jl/pull/1886)).
- Julia 1.11 fixes (PR [#1883](https://github.com/CliMA/ClimaCore.jl/pull/1883))
- `Nh` has been added to the type parameter space, which allows us to more flexibly write performant backend kernels (PR [#1894](https://github.com/CliMA/ClimaCore.jl/pull/1894)). This was leveraged in PR [#1898](https://github.com/CliMA/ClimaCore.jl/pull/1898), and may result in slightly more performant kernels.


v0.14.10
-------

- Various performance tweaks (PRs [#1840](https://github.com/CliMA/ClimaCore.jl/pull/1840), [#1837](https://github.com/CliMA/ClimaCore.jl/pull/1837), [#1843](https://github.com/CliMA/ClimaCore.jl/pull/1843), [#1839](https://github.com/CliMA/ClimaCore.jl/pull/1839)).
- CPU/GPU kernels are now determined by dispatching, instead of specializing, which should (hopefully) have generally fixed GPU dispatching issues (PR [#1863](https://github.com/CliMA/ClimaCore.jl/pull/1863)).
- Matrix multiplication kernels have been improved (PR [#1880](https://github.com/CliMA/ClimaCore.jl/pull/1880)).
- Support for the following methods have been deprecated (PR [#1821](https://github.com/CliMA/ClimaCore.jl/pull/1821), ):
  - `IntervalTopology(::Mesh)` in favor of using `IntervalTopology(::ClimaComms.AbstractDevice, ::Mesh)`
  - `FaceFiniteDifferenceSpace(::Mesh)` in favor of using `FaceFiniteDifferenceSpace(::ClimaComms.AbstractDevice, ::Mesh)`
  - `CenterFiniteDifferenceSpace(::Mesh)` in favor of using `CenterFiniteDifferenceSpace(::ClimaComms.AbstractDevice, ::Mesh)`
  - `FiniteDifferenceGrid(::Mesh)` in favor of using `FiniteDifferenceGrid(::ClimaComms.AbstractDevice, ::Mesh)`

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
