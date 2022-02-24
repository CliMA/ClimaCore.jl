import ClimaCore
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Fields
using NCDatasets
using TempestRemap_jll
using Test
using ClimaCoreTempestRemap

include("../src/connectivity.jl")

OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

@testset "online remap 2D sphere data" begin

    # domain 
    R = 1.0 # unit sphere 
    domain = ClimaCore.Domains.SphereDomain(R)

    # source grid params
    ne_i = 20 # #elements
    nq_i = 3

    # target grid params
    ne_o = 5  
    nq_o = 3   

    # construct source mesh
    mesh_i = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_i) 
    topology_i = ClimaCore.Topologies.Topology2D(mesh_i)
    space_i = Spaces.SpectralElementSpace2D(topology_i, Spaces.Quadratures.GLL{nq_i}())
    coords_i = Fields.coordinate_field(space_i)

    # construct target mesh
    mesh_o = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_o) 
    topology_o = ClimaCore.Topologies.Topology2D(mesh_o)
    space_o = Spaces.SpectralElementSpace2D(topology_o, Spaces.Quadratures.GLL{nq_o}())
    coords_o = Fields.coordinate_field(space_o)

    # use TempestRemap to generate map weights
    weightfile = joinpath(OUTPUT_DIR, "remap_data.nc")
    map, col, row = generate_map(weightfile, topology_i, topology_o, nq_i, nq_o)

    # get connectivity for projections from ClimaCore Fields to Tempest sparse matrix
    _, n_gll_i, connect_i = get_cc_gll_connect(ne_i, nq_i)
    _, n_gll_o, connect_o = get_cc_gll_connect(ne_o, nq_o)

    # initialize the remap operator
    RemapInfo = LinearTempestRemap(map, col, row, nq_i, ne_i, n_gll_i, connect_i, nq_o, ne_o, n_gll_o, connect_o )

    # generate test data in the Field format
    field_i = sind.(Fields.coordinate_field(space_i).long)
    field_o = similar(field_i, space_o, eltype(field_i))
    
    # apply the remap
    remap!(field_o, field_i, RemapInfo)

    ## write test data for offline map apply for comparison
    datafile_in = joinpath(OUTPUT_DIR, "data_in.nc")
    
    NCDataset(datafile_in, "c") do nc
        def_space_coord(nc, space_i)
        nc_time = def_time_coord(nc)

        # nc_xlat = defVar(nc, "xlat", Float64, space)
        nc_sinlong = defVar(nc, "sinlong", Float64, space_i, ("time",))

        # nc_time[:] = time[:]
        # nc_xlat[:] = xlat[:]
        nc_sinlong[:, 1] = field_i

        nothing
    end
    
    ## for test below, apply offline map, read in the resulting field and reshape it to the IJFH format 
    datafile_out = joinpath(OUTPUT_DIR, "data_out.nc")
    apply_remap(datafile_out, datafile_in, weightfile, ["sinlong"])

    ds_wt = NCDataset(datafile_out,"r")
    field_o_offline = ds_wt["sinlong"][:][:,1]
    close(ds_wt)

    field_o_offline = Float64.(field_o_offline)
    field_o_offline_reshaped = project_sparse_to_IJFH(field_o_offline, connect_o, nq_o, ne_o)

    err = maximum(sqrt.((field_o_offline_reshaped[:,:,1,:,:] - parent(field_o)) .^ 2))

    @test err < 1e-6

end

#=
using Plots
function plot_flatmesh(Psi,nelem)
    plots = []
    tiltes = ["Eq1" "Eq2" "Eq3" "Eq4" "Po1" "Po2"]
    for f in collect(1:1:6)
        Psi_reshape = reshape(Psi[(f-1)*nelem^2+1:f*nelem^2],(nelem,nelem))

        push!(plots, contourf(Psi_reshape))
    end
    plot(plots..., layout = (6), title = tiltes )
end

plot_flatmesh(parent(field_i)[1,1,1,:],ne_i)
png(joinpath(OUTPUT_DIR,"in.png"))

plot_flatmesh(parent(field_o)[1,1,1,:],ne_o)
png(joinpath(OUTPUT_DIR,"out.png"))

plot_flatmesh(parent(field_o_offline_reshaped)[1,1,1,1,:],ne_o)
png(joinpath(OUTPUT_DIR,"out_offline.png"))


using BenchmarkTools
@btime  apply_remap --> 53.477 ms (304 allocations: 19.78 KiB)
@btime remap! --> 1.398 s (65163 allocations: 1.55 GiB)
@btime remap!(Simon_fix) --> 363.599 μs (623 allocations: 150.81 KiB) 



=#