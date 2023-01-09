# Installation

`ClimaCore.jl` is a registered Julia package. You can install the latest version
of `ClimaCore.jl` through the built-in package manager. Press `]` in the Julia REPL
command prompt and

```julia
julia> ]
(v1.8) pkg> add ClimaCore
(v1.8) pkg> instantiate
```

This will install the latest tagged release of the package.

!!! info "But I wanna be on the bleeding edge..."
    If you want the *most recent* developer's version of the package then

    ```julia
    julia> ]
    (v1.8) pkg> add ClimaCore#main
    (v1.8) pkg> instantiate
    ```

You can run the tests via the package manager by:

```julia
julia> ]
(v1.8) pkg> test ClimaCore
```

# Running examples

We have a selection of examples, found within the `examples/` directory to demonstrate different use of our library.
Each example directory contains a `Project.toml`

To build with the latest `ClimaCore.jl` release:
```
> cd examples/
> julia --project -e 'using Pkg; Pkg.instantiate()'
> julia --project example_file_name.jl
```
If you wish to run a local modified version of `ClimaCore.jl` then try the following (starting from the `ClimaCore.jl` package root)
```
> cd examples/
> julia --project
> julia> ]
> (examples)> rm ClimaCore.jl
> (examples)> dev ../
> (examples)> instantiate
```
followed by
```
> julia --project example_file_name.jl
```

## Attribution and Credits
These instructions and how-to guides are heavily based on the excellent [`EnsembleKalmanProcesses.jl` Installation Instructions](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/installation_instructions//)
