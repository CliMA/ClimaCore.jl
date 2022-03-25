## ClimaCoreArrow

[Tables.jl implementation](http://docs.juliaplots.org/latest/recipes) for ClimaCore datastructures.
Can be used to ex. cast a ClimaCore Field to a Julia DataFrame efficiently, output Field data to a CSV,
or interop through Arrow with other language environments.

### Use ClimaCoreTables from the ClimaCore parent package

    cd ClimaCore
    julia --project

Activate the Pkg repl mode and dev the ClimaCoreTables subpackge:

    (ClimaCore) pkg> dev lib/ClimaCoreTables

You can now use ClimaCoreTables in your ClimaCore pkg environment:

    julia> using ClimaCoreTables

Tooling that supports the Tables.jl interface should be able to now natively work with ClimaCore datastructures:

    julia> field = ClimaCore.Fields.Field(...)
    julia> using CSV
    julia> field |> CSV.write("field_data.csv", kwargs...)

To get more over various optional representations, you can also manually construct the `ClimaCoreTable` wrapper type:

    julia table = ClimaCoreTable(field, local_geometry=true)
    julia table |> CSV.write("field_with_space_geometry.csv", kwargs...)

Columnar Table.jl descriptions are defined for ClimaCore `Space` and `Field` types.

### Development of the `ClimaCoreArrow` subpackage

    cd ClimaCore/lib/ClimaCoreTables

    # Add ClimaCore to subpackage environment
    julia --project -e 'using Pkg; Pkg.develop(path="../../")'

    # Instantiate ClimaCoreTables project environment
    julia --project -e 'using Pkg; Pkg.instantiate()'
    julia --project -e 'using Pkg; Pkg.test()'
