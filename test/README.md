# Tests

Our test suite organization and goals are outlined here. This document was designed to be somewhat agnostic to this package.

Some goals we hope to achieve are:

 - Achieve high coverage in our unit tests
 - Run our unit tests in a very short time (<3 minutes)

In order to achieve thes goals, we need to strategically organize our tests, and partition unit tests from more complicated ones. To do so, we've organized our tests by "levels":

## Test levels

 - Level 1: Quick and high coverage unit tests (with only `Float64`). These tests are run with `MyPackage> test`, and should be as quick as possible, while exercising, and testing the logic of, a high percentage of the source code.

 - Level 2: Inference, flop-counting, and allocation tests. Since compilation incurs allocations, allocation _tests_ require running everything twice, in order to measure runtime allocations. Inference tests can be somewhat expensive, too, as this involves verifying inference. This includes, for example, running tests with `Float32`. Therefore, these tests are separate from Level 1.

 - Level 3: Convergence tests. Convergence tests are highly valuable in testing the correctness of algorithms with theoretic convergence guarantees. This requires running code multiple times (often more than 2 or 3), therefore, these tests are separate from Level 1 and 2.

 - Level 4: Examples and simulations. Computing solutions to ODEs/PDEs can be very expensive, and are separated into level 4 tests. These may be examples that one can compare with (approximate or) exact solutions, or with (e.g., numerical) data.

 - Benchmarks: Benchmarks are special in that they must be performed at target resolutions, and we often can't strictly test benchmarks due to variation / noise (which can easily reach 30%). Therefore, benchmarks will only run and log the results, so that we have a trail of their results over time.

## Test organization

We have separate modules for different components, and now different test levels. Partitioning our tests requires some level of organization.

 - Tests are firstly organized in folders based on the package's modules. e.g., `MyPackage.SubModuleA` -> `test/SubModuleA/`.
 - Files use the following prefix convention:
  - `utils_` - non-test files. These files should only define methods, and should be designed for flexibility (e.g., can create objects with `Float32` or `Float64`).
  - `unit_` -        Unit        test files
  - `inference_`   - Inference   test files
  - `allocs_`      - Allocation  test files
  - `convergence_` - Convergence test files
  - `benchmark_` -   Benchmark   test files

This should help keep including shared files between `unit`, `inference` etc. tests simple as they are still organized by the packages sub-modules.

One helpful tip on organizing tests is: try to (sensibly) split off the creation of the method arguments from testing a particular method. This is very helpful for REPL testing, and will allow developers to share lots of code between the different levels of tests.

# ClimaCore specifics:

 - Our benchmark tests must be performed at a target resolution.

