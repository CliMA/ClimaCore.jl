## ClimaCoreMakie

Makie plot infastructure for ClimaCore fields for 2D / 3D visualization.

### Use ClimaCoreMakie from the ClimaCore parent package

    cd ClimaCore
    julia --project

Activate the Pkg repl mode and dev the ClimaCoreMakie subpackge:

    (ClimaCore) pkg> dev lib/ClimaCoreMakie

You can now use ClimaCoreMakie in your ClimaCore pkg environment:

    julia> using ClimaCoreMakie

To vizualize a clima core object, you fist need to install a Makie backend:

    (@v1.7) pkg> add GLMakie # 3D
    (@v1.7) pkg> add CairoMakie # 2D
    
Then load in the backend:

    julia> using GLMakie

Finally call `viz`:
	
    julia> ClimaCoreMakie.viz(field, ...)


### Development of the `ClimaCoreMakie` subpackage

    cd ClimaCore/lib/ClimaCoreMakie

    # Add ClimaCore to subpackage environment
    julia --project -e 'using Pkg; Pkg.develop(path="../../")

    # Instantiate ClimaCoreMakie project environment
    julia --project -e 'using Pkg; Pkg.instantiate()'
    julia --project -e 'using Pkg; Pkg.test()'
