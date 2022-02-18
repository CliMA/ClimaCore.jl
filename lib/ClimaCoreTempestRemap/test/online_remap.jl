import ClimaCore
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Fields
using NCDatasets
using TempestRemap_jll
using Test
using ClimaCoreTempestRemap

OUTPUT_DIR = "."#mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

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

    field_i = sind.(Fields.coordinate_field(space_i).long)
    field_o = similar(field_i, space_o, eltype(field_i))
    
    # initialize the remap operator
    RemapInfo = LinearTempestRemap(field_o, field_i, map, col, row)
    
    # apply the remap
    remap!(RemapInfo)


    # offline map apply for comparison

    ## write test data for offline map apply for comparison
    datafile_in = joinpath(OUTPUT_DIR, "data_in.nc")
    NCDataset(datafile_cc, "c") do nc
        def_space_coord(nc, space)
        nc_time = def_time_coord(nc)

        # nc_xlat = defVar(nc, "xlat", Float64, space)
        nc_sinlong = defVar(nc, "sinlong", Float64, space, ("time",))

        nc_time[:] = time[:]
        # nc_xlat[:] = xlat[:]
        nc_sinlong[:] = field_i[:]

        nothing
    end
    
    ## offline map apply for comparison
    datafile_out = joinpath(OUTPUT_DIR, "data_out.nc")
    apply_remap(datafile_out, datafile_in, weightfile, vars)

    #@test Array(nc_rll["xlat"]) ≈ ones(nlon) * lats' rtol = 0.1
end

