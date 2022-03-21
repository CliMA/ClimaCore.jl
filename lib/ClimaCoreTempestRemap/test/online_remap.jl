import ClimaCore
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Fields
using NCDatasets
using TempestRemap_jll
using Test
using ClimaCoreTempestRemap
using LinearAlgebra

OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

"""
    reshape_sparse_to_field!(field, in_array, R)

reshapes and broadcasts a sparse matrix data array (e.g., output from TempestRemap) into a Field object
"""
function reshape_sparse_to_field!(field::Fields.Field, in_array::Array, R)
    field_array = parent(field)

    fill!(field_array, zero(eltype(field_array)))
    Nf = size(field_array, 3)

    f = 1
    for (
        source_idx_i,
        source_idx_j,
        source_idx_e,
        it,
        jt,
        et,
        row,
    ) in zip(R.source_idxs_i, R.source_idxs_j, R.source_idxs_e, R.target_idxs_i, R.target_idxs_j, R.target_idxs_e, R.row)
        #(it, jt), et = target_idx
        for f in 1:Nf
            field_array[it, jt, f, et] = in_array[row]
        end
    end
    # broadcast to the redundant nodes using unweighted dss
    Spaces.horizontal_dss!(field)
    return field
end

@testset "online remap 2D sphere data" begin

    # domain
    R = 1.0 # unit sphere
    domain = ClimaCore.Domains.SphereDomain(R)

    # source grid params
    ne_i = 20 # #elements
    nq_i = 3 # polynomial order for SE discretization

    # target grid params
    ne_o = 5
    nq_o = 3

    # construct source mesh
    mesh_i = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_i)
    topology_i = ClimaCore.Topologies.Topology2D(mesh_i)
    space_i = Spaces.SpectralElementSpace2D(
        topology_i,
        Spaces.Quadratures.GLL{nq_i}(),
    )
    coords_i = Fields.coordinate_field(space_i)

    # construct target mesh
    mesh_o = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_o)
    topology_o = ClimaCore.Topologies.Topology2D(mesh_o)
    space_o = Spaces.SpectralElementSpace2D(
        topology_o,
        Spaces.Quadratures.GLL{nq_o}(),
    )
    coords_o = Fields.coordinate_field(space_o)

    # generate test data in the Field format
    field_i = sind.(Fields.coordinate_field(space_i).long)
    field_o = similar(field_i, space_o, eltype(field_i))

    # use TempestRemap to generate map weights
    weightfile = tempname()
    R = ClimaCoreTempestRemap.generate_map(
        space_o,
        space_i,
        weightfile = weightfile,
    )

    # apply the remap
    ClimaCoreTempestRemap.remap!(field_o, field_i, R)

    # TEST_1: error between our `apply!` in ClimaCoe and `apply_remap` in TempestRemap

    # write test data for offline map apply for comparison
    datafile_in = joinpath(OUTPUT_DIR, "data_in.nc")

    NCDataset(datafile_in, "c") do nc
        def_space_coord(nc, space_i)
        nc_time = def_time_coord(nc)
        nc_sinlong = defVar(nc, "sinlong", Float64, space_i, ("time",))
        nc_sinlong[:, 1] = field_i
        nothing
    end

    ## for test below, apply offline map, read in the resulting field and reshape it to the IJFH format
    datafile_out = joinpath(OUTPUT_DIR, "data_out.nc")
    apply_remap(datafile_out, datafile_in, weightfile, ["sinlong"])

    ds_wt = NCDataset(datafile_out, "r")
    offline_outarray = ds_wt["sinlong"][:][:, 1]
    close(ds_wt)
    offline_outarray = Float64.(offline_outarray)

    offline_field = similar(field_o)
    reshape_sparse_to_field!(offline_field, offline_outarray, R)

    error_1 = maximum(sqrt.((offline_field .- field_o) .^ 2))

    @test error_1 < 1e-6

    # TEST_2: error compared to the analytical solution

    # reference
    field_ref = sind.(Fields.coordinate_field(space_o).long)
    @test field_ref ≈ field_o atol = 0.1

    # OTHER VIZ TESTS: can be performed manually using commented code below 

end


#=
using Plots
using ClimaCorePlots
heatmap(field_ref)
heatmap(field_o)

field_ref = sind.(Fields.coordinate_field(space_o).long)
norm(field_ref .- field_o)
heatmap(field_ref .- field_o)

heatmap(offline_field .- field_o)

using BenchmarkTools
@btime ClimaCoreTempestRemap.remap!(field_o, field_i, R)
128.883 ms (2589602 allocations: 68.91 MiB)

109.237 ms (2589602 allocations: 68.91 MiB) < no dss

267.977 ms (1294803 allocations: 31.36 MiB)

270.560 ms (1294801 allocations: 30.59 MiB)

11.833 ms (405608 allocations: 24.76 MiB) < split out and view indices

9.998 ms (304208 allocations: 19.34 MiB) < split out and view indices + view source array
=#

# domain
R = 1.0 # unit sphere
domain = ClimaCore.Domains.SphereDomain(R)

# source grid params
ne_i = 20 # #elements
nq_i = 3 # polynomial order for SE discretization

# target grid params
ne_o = 5
nq_o = 3

# construct source mesh
mesh_i = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_i)
topology_i = ClimaCore.Topologies.Topology2D(mesh_i)
space_i = Spaces.SpectralElementSpace2D(
    topology_i,
    Spaces.Quadratures.GLL{nq_i}(),
)
coords_i = Fields.coordinate_field(space_i)

# construct target mesh
mesh_o = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_o)
topology_o = ClimaCore.Topologies.Topology2D(mesh_o)
space_o = Spaces.SpectralElementSpace2D(
    topology_o,
    Spaces.Quadratures.GLL{nq_o}(),
)
coords_o = Fields.coordinate_field(space_o)

# generate test data in the Field format
field_i = sind.(Fields.coordinate_field(space_i).long)
field_o = similar(field_i, space_o, eltype(field_i))

# use TempestRemap to generate map weights
weightfile = tempname()
R = ClimaCoreTempestRemap.generate_map(
    space_o,
    space_i,
    weightfile = weightfile,
)

# apply the remap
ClimaCoreTempestRemap.remap!(field_o, field_i, R)

using BenchmarkTools
@btime ClimaCoreTempestRemap.remap!(field_o, field_i, R)