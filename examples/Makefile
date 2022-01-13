instantiate: Manifest.toml

Manifest.toml:
	julia --project=. -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path=".."), Pkg.PackageSpec(path="../lib/ClimaCorePlots")])'

.PHONY: instantiate