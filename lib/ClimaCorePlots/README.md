## ClimaCorePlots

[Plots.jl recipes](http://docs.juliaplots.org/latest/recipes) for visualizing ClimaCore Field / Space datastructures.

### Use ClimaCorePlots from the ClimaCore parent package

    cd ClimaCore
    julia --project

Activate the Pkg repl mode and dev the ClimaCorePlots subpackge:

    (ClimaCore) pkg> dev lib/ClimaCorePlots

You can now use ClimaCorePlots in your ClimaCore pkg environment:

    julia> using ClimaCorePlots

To visualize a ClimaCore object, you fist need to install a [Plots.jl backend](http://docs.juliaplots.org/latest/backends/#backends):

    (@v1.8) pkg> add Plots # (default GR backend)
    (@v1.8) pkg> add Plots, PlotlyJS

Then load in the backend:

    julia> using Plots
    julia> gr() # or plotlyjs(),  ...

Finally call `plot`:

    julia> plot(field, ...)

Consult the Plots.jl documentation for more information regarding setting attributes, backends, or saving output.

### Development of the `ClimaCorePlots` subpackage

    cd ClimaCore/lib/ClimaCorePlots

    # Add ClimaCore to subpackage environment
    julia --project -e 'using Pkg; Pkg.develop(path="../../")'

    # Instantiate ClimaCorePlots project environment
    julia --project -e 'using Pkg; Pkg.instantiate()'
    julia --project -e 'using Pkg; Pkg.test()'
