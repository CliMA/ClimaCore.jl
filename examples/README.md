## ClimaCore Examples

* `bickleyjet/`: 2D spectral element implementations of the bickleyjet testcase
* `column/`: 1D column tests for finite difference operators
* `hybrid/`: 2D slice model testcases for the hybrid spectral / fd domains
* `sphere/`: 2D dycore shallow water testcases on the cubed sphere
* `sphere3d/`: 3D dycore testcases on the hybrid spectral / fd cubed sphere domain


### Use ClimaCore Examples from the ClimaCore parent package

    cd ClimaCore
    julia --project=examples

Activate the Pkg repl mode and dev the ClimaCore repo and ClimaCorePlots packages into the examples project environment:

    (examples) pkg> dev . lib/ClimaCorePlots
    (examples) pkg> instantiate

Or alternatively:

    julia --project=examples -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path="."), Pkg.PackageSpec(path="lib/ClimaCorePlots")])'

You can now execute the examples in the example projects, and ClimaCore changes will be tracked in the env:

    julia --project=examples
    julia> include(examples/bickleyjet/bickleyjet_cg.jl)

Or alternatively:

    julia --project=examples examples/bickleyjet/bickleyjet_cg.jl