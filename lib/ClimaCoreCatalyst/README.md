## ClimaCoreCatalyst

ClimaCore [Paraview Catalyst](https://www.paraview.org/in-situ/)
adaptor for visualizing ClimaCore field data in-situ.

**Note:** Needs Paraview v5.10+

### Use ClimaCoreCatalyst from the ClimaCore parent package

    cd ClimaCore
    julia --project

Activate the Pkg repl mode and dev the ClimaCoreCatalyst subpackge:

    (ClimaCore) pkg> dev lib/ClimaCoreCatalyst

You can now use ClimaCoreCatalyst in your ClimaCore pkg environment:

    julia> using ClimaCoreCatalyst

### Development of the `ClimaCoreCatalyst` subpackage

    cd ClimaCore/lib/ClimaCoreCatalyst

    # Add ClimaCore to subpackage environment
    julia --project -e 'using Pkg; Pkg.develop(path="../../")'

    # Instantiate the ClimaCoreCatalyst project environment
    julia --project -e 'using Pkg; Pkg.instantiate()'
    julia --project -e 'using Pkg; Pkg.test()'
