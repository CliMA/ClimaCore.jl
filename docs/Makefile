docs: Manifest.toml
	julia --project=. make.jl

Manifest.toml:
	julia --project=. -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path=".."), Pkg.PackageSpec(path="../lib/ClimaCorePlots"), Pkg.PackageSpec(path="../lib/ClimaCoreVTK"), Pkg.PackageSpec(path="../lib/ClimaCoreMakie")])'

.PHONY: docs