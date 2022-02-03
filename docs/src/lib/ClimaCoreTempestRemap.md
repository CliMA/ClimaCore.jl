# ClimaCoreTempestRemap.jl

```@meta
CurrentModule = ClimaCoreTempestRemap
```

ClimaCoreTempestRemap.jl provides an interfaces for using ClimaCore data with
the [TempestRemap](https://github.com/ClimateGlobalChange/tempestremap/)
remapping package, by Paul Ullrich.

# Interface

## Mesh export

```@docs
write_exodus
```

## NetCDF data export

```@docs
def_time_coord
def_space_coord
NCDatasets.defVar(::NCDatasets.NCDataset, ::Any, field::Fields.Field)
Base.setindex!(::NCDatasets.CFVariable, ::Fields.Field, ::Colon)
```

## Wrapper functions

```@docs
rll_mesh
overlap_mesh
remap_weights
apply_remap
```

# Example

The following example converts an OrdinaryDiffEq solution object `sol` to a netcdf file, and remaps it to an regular latitude-longitude (RLL) grid.

```julia
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces
using NCDatasets, ClimaCoreTempestRemap

# sol is the integrator solution
# cspace is the center extrduded space
# fspace is the face extruded space

# the issue is that the Space types changed since this changed
# we can reconstruct it by digging around a bit
Nq = Spaces.Quadratures.degrees_of_freedom(cspace.quadrature_style)

datafile_cc = "test.nc"
NCDataset(datafile_cc, "c") do nc
    # defines the appropriate dimensions and variables for a space coordinate
    def_space_coord(nc, cspace)
    def_space_coord(nc, fspace)
    # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
    nc_time = def_time_coord(nc)

    # define variables
    nc_rho = defVar(nc, "rho", Float64, cspace, ("time",))
    nc_theta = defVar(nc, "theta", Float64, cspace, ("time",))
    nc_u = defVar(nc, "u", Float64, cspace, ("time",))
    nc_v = defVar(nc, "v", Float64, cspace, ("time",))
    nc_w = defVar(nc, "w", Float64, fspace, ("time",))

    # write data to netcdf file
    for i = 1:length(sol.u)
        nc_time[i] = sol.t[i]

        # extract fields and convert to orthogonal coordinates
        Yc = sol.u[i].Yc
        uₕ = Geometry.UVVector.(sol.u[i].uₕ)
        w = Geometry.WVector.(sol.u[i].w)

        # write fields to file
        nc_rho[:,i] = Yc.ρ
        nc_theta[:,i] = Yc.ρθ ./ Yc.ρ
        nc_u[:,i] = map(u -> u.u, uₕ)
        nc_v[:,i] = map(u -> u.v, uₕ)
        nc_w[:,i] = map(u -> u.w, w)
    end
end

# write out our cubed sphere mesh
meshfile_cc = "mesh_cubedsphere.g"
write_exodus(meshfile_cc, cspace.horizontal_space.topology)

# write out RLL mesh
nlat = 90
nlon = 180
meshfile_rll = "mesh_rll.g"
rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

# construct overlap mesh
meshfile_overlap = "mesh_overlap.g"
overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

# construct remap weight file
weightfile = "remap_weights.nc"
remap_weights(
    weightfile,
    meshfile_cc,
    meshfile_rll,
    meshfile_overlap;
    in_type = "cgll",
    in_np = Spaces.degrees_of_freedom(space),
)

# apply remap
datafile_rll = "data_rll.nc"
apply_remap(datafile_rll, datafile_cc, weightfile, ["rho", "theta", "u", "v", "w"])
```