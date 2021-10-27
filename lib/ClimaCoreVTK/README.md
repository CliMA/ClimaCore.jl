## ClimaCoreVTK

VTK output for ClimaCore fields for visualization with [Paraview](https://www.paraview.org/).

### Use ClimaCoreVTK from the ClimaCore parent package

    cd ClimaCore
    julia --project

Activate the Pkg repl mode and dev the ClimaCoreVTK subpackge:

    (ClimaCore) pkg> dev lib/ClimaCoreVTK

You can now use ClimaCoreVTK in your ClimaCore pkg environment:

    julia> using ClimaCoreVTK
    julia> writevtk(...)

### Development of the `ClimaCoreVTK` subpackage

    cd ClimaCore/lib/ClimaCoreVTK

    # Add ClimaCore to subpackage environment
    julia --project -e 'using Pkg; Pkg.develop("../../")

    # Instantiate ClimaCoreVTK project environment
    julia --project -e 'using Pkg; Pkg.instantiate()'
    julia --project -e 'using Pkg; Pkg.test()'
