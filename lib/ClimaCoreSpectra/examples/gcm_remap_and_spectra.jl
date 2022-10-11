import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators

using NCDatasets, ClimaCoreTempestRemap, Test, FFTW, Plots

import ClimaCoreSpectra: power_spectrum_2d, compute_gaussian!

OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

FT = Float64

# Setup CC mesh
ne = 4
R = 5.0
Nq = 5
domain = Domains.SphereDomain(R)
mesh = Meshes.EquiangularCubedSphere(domain, ne)
topology = Topologies.Topology2D(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(topology, quad)
coords = Fields.coordinate_field(space)

# Define simple lat-long field
sinθ = sind.(Fields.coordinate_field(space).lat)
long = Fields.coordinate_field(space).long

scaling = long ./ maximum(parent(long)) ./ 2
field_m3_P_32 = sqrt(105 / 8) .* (sinθ .- sinθ .^ 3) .* cos.(scaling .* 2 .* 2π) # n = 3 (total wavenumber, and degree of associated Legendre polynomial), m = 2 (zonal wavenumber, which MUST be equal to the order m of the associated Legendre polynomial). This is a modified version of the associated Legendre polynomial \hat{P}_3^2 from ref: Ehrendorfer, M. (2011) Spectral Numerical Weather Prediction Models, Appendix B, Society for Industrial and Applied Mathematics, Table B.1

# Remap infrastructure:
# write mesh
meshfile_cc = joinpath(OUTPUT_DIR, "mesh_cc.g")
write_exodus(meshfile_cc, topology)

# write data
datafile_cc = joinpath(OUTPUT_DIR, "data_cc.nc")
NCDataset(datafile_cc, "c") do nc
    def_space_coord(nc, space; type = "cgll")

    nc_m3_P_32 = defVar(nc, "m3_P_32", FT, space)
    nc_m3_P_32[:] = field_m3_P_32

    nothing
end

nlat = 32 # Gauss latitudes
nlon = 64 # Gauss longitudes
meshfile_rll = joinpath(OUTPUT_DIR, "mesh_rll.g")
rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

meshfile_overlap = joinpath(OUTPUT_DIR, "mesh_overlap.g")
overlap_mesh(meshfile_overlap, meshfile_cc, meshfile_rll)

weightfile = joinpath(OUTPUT_DIR, "remap_weights.nc")
remap_weights(
    weightfile,
    meshfile_cc,
    meshfile_rll,
    meshfile_overlap;
    in_type = "cgll",
    in_np = Nq,
)

datafile_rll = joinpath(OUTPUT_DIR, "data_rll.nc")
apply_remap(datafile_rll, datafile_cc, weightfile, ["m3_P_32"])

nt = NCDataset(datafile_rll) do nc_rll
    lat = Array(nc_rll["lat"])
    lon = Array(nc_rll["lon"])
    m3_P_32 = Array(nc_rll["m3_P_32"])
    (; lat, lon, m3_P_32)
end

(; lat, lon, m3_P_32) = nt

mass_weight = ones(FT, 1)
spectrum, wave_numbers, spherical, mesh_info =
    power_spectrum_2d(FT, m3_P_32, mass_weight)

# Plot the spectrum
Plots.contourf(
    collect(0:1:(mesh_info.num_fourier))[:],
    collect(0:1:(mesh_info.num_spherical))[:],
    (spectrum[:, :, 1])',
    xlabel = "m",
    ylabel = "n",
    clim = (0, 0.25),
    ylims = (0, 5),
    xlims = (0, 5),
    c = :roma, # this palette was tested for color-blindness safety using the online simulator https://www.color-blindness.com/coblis-color-blindness-simulator/
)

# Verification of the spectrum:
# Since we started with the  associated Legendre polynomial P_3^2, we now check that the max of the spectrum happens at (m, n) = (2, 3) indices.
# We need add 1 b/c Julia indexing starts from 1, whereas the order and degree of associated Legendre polynomials start from 0 by convention.
@assert (3, 4) == Tuple(argmax(spectrum[:, :, 1]))
