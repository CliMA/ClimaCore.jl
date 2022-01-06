# Simple script that automates setting up the documentation project environment
# before building the docs.
import Pkg
Pkg.activate(@__DIR__)
Pkg.develop([
    Pkg.PackageSpec(path = normpath(joinpath(@__DIR__, ".."))),
    Pkg.PackageSpec(
        path = normpath(joinpath(@__DIR__, "..", "lib", "ClimaCoreVTK")),
    ),
    Pkg.PackageSpec(
        path = normpath(joinpath(@__DIR__, "..", "lib", "ClimaCorePlots")),
    ),
    Pkg.PackageSpec(
        path = normpath(joinpath(@__DIR__, "..", "lib", "ClimaCoreMakie")),
    ),
])
Pkg.instantiate()
include(joinpath(@__DIR__, "make.jl"))
