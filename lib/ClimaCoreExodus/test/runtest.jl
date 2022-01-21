import ClimaCore
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces
using NCDatasets
using TempestRemap_jll
using Test 
using IntervalSets

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

dir = mktempdir(".")

include("../src/ClimaCoreExodus.jl")

@testset "mesh coordinates ClimaCore > TempestRemap" begin
    """
    ensures that Equiangular cubed sphere node Cartesian coords are identical in ClimaCore (CC) and TempestRemap (TR)
    """
    # generate CC mesh
    ne = 9
    R = 1 # unit sphere to compare with TR
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne) # ne×ne×6 
    grid_topology = Topologies.Topology2D(mesh)
    nc_name_cc = dir*"/test_cc.nc"
    write_exodus_identical(nc_name_cc, grid_topology)

    # generate TR mesh
    nc_name_tr = dir*"/test_tr.nc"
    run(`$(TempestRemap_jll.GenerateCSMesh_exe()) --res $ne --alt --file $nc_name_tr`)

    # open both files and compare Cartesian coords point-by-point
    ds_cc = NCDataset(nc_name_cc, "r");
    coord_cc = ds_cc["coord"][:]

    ds_tr = NCDataset(nc_name_tr, "r");
    coord_tr = ds_tr["coord"][:]

    @test sqrt(sum((coord_cc .- coord_tr) .^2)) < 10e-15
end

@testset "regrid fv" begin
    """
    regrid fv CS generated in CC
    """
    # input mesh
    nq = 4
    ne_i = 20
    R = 1 # unit sphere 
    domain = ClimaCore.Domains.SphereDomain(R)
    mesh = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_i) # ne×ne×6 
    grid_topology = ClimaCore.Topologies.Topology2D(mesh)
    nc_name_in = dir*"/test_in.nc"
    write_exodus_identical(nc_name_in, grid_topology)

    # generate fake input data (replace with CC variable; NB this requires write_exodus_identical() above)
    nc_name_data_in = dir*"/Psi_in.nc"
    run(`$(GenerateTestData_exe()) --mesh $nc_name_in --test 1 --out $nc_name_data_in`) # var: Psi

    # output mesh
    ne_o = 3
    R = 1 
    domain = ClimaCore.Domains.SphereDomain(R)
    mesh = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_o) # ne×ne×6 
    grid_topology = ClimaCore.Topologies.Topology2D(mesh)
    nc_name_out = dir*"/test_out.nc"
    write_exodus_identical(nc_name_out, grid_topology)

    # overlap mesh
    nc_name_ol = dir*"/test_ol.g"
    run(`$(TempestRemap_jll.GenerateOverlapMesh_exe()) --a $nc_name_in --b $nc_name_out --out $nc_name_ol`)
    
    # map weights 
    nc_name_wgt = dir*"/test_wgt.g"
    run(`$(TempestRemap_jll.GenerateOfflineMap_exe()) --in_mesh $nc_name_in --out_mesh $nc_name_out --ov_mesh $nc_name_ol --in_np 1 --out_map $nc_name_wgt`) # FV > FV, parallelized sparse matrix multiply (SparseMatrix.h)

    # apply map (this to be done by CC at each timestep)
    nc_name_data_out = dir*"/Psi_out.nc"
    run(`$(TempestRemap_jll.ApplyOfflineMap_exe()) --map $nc_name_wgt --var Psi --in_data $nc_name_data_in --out_data $nc_name_data_out`)
    
    # # viz
    # ds_indata = NCDataset(nc_name_data_in,"r")
    # Psi_in = ds_indata["Psi"][:]
    # close(ds_indata)
    # ds_inmesh = NCDataset(nc_name_in,"r")
    # connect1_in = ds_inmesh["connect1"][:]
    # coord_in = ds_inmesh["coord"][:]
    # plot_flatmesh(Psi_in,ne_i)
    # png(dir*"/in.png")

    # ## get regriddded field and its mapping
    # ds_outdata = NCDataset(nc_name_data_out,"r")
    # Psi_out = ds_outdata["Psi"][:]
    # close(ds_outdata)
    # ds_outmesh = NCDataset(nc_name_out,"r")
    # connect1_out = ds_outmesh["connect1"][:]
    # coord_in = ds_outmesh["coord"][:]
    # plot_flatmesh(Psi_out,ne_o)
    # png(dir*"/out.png")

end    

@testset "regrid fe" begin

    # input mesh
    nq = 5
    ne_i = 20
    R = 1 # unit sphere 
    domain = ClimaCore.Domains.SphereDomain(R)
    mesh = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_i) # ne×ne×6 
    grid_topology = ClimaCore.Topologies.Topology2D(mesh)
    nc_name_in = dir*"/test_in.nc"
    write_exodus_identical(nc_name_in, grid_topology)

    # generate fake input data (replace with CC variable; NB this requires write_exodus_identical() above)
    #nc_name_data_in = dir*"/Psi_in.nc"
    #run(`$(GenerateTestData_exe()) --mesh $nc_name_in --test 1 --out $nc_name_data_in --gll --np $nq`) # var: Psi
    #run(`$(GenerateTestData_exe()) --mesh $nc_name_in --test 1 --out $nc_name_data_in`) # var: Psi

    # output mesh
    ne_o = 3
    R = 1 
    domain = ClimaCore.Domains.SphereDomain(R)
    mesh = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne_o) # ne×ne×6 
    grid_topology = ClimaCore.Topologies.Topology2D(mesh)
    nc_name_out = dir*"/test_out.nc"
    write_exodus_identical(nc_name_out, grid_topology)

    #run(`$(TempestRemap_jll.GenerateCSMesh_exe()) --res 6 --alt --file $nc_name_in`)
    #run(`$(TempestRemap_jll.GenerateCSMesh_exe()) --res 9 --alt --file $nc_name_out`)

    # overlap mesh
    nc_name_ol = dir*"/test_ol.g"
    run(`$(TempestRemap_jll.GenerateOverlapMesh_exe()) --a $nc_name_in --b $nc_name_out --out $nc_name_ol`)
    
    # map weights 
    nc_name_wgt = dir*"/test_wgt.g"
    #run(`$(TempestRemap_jll.GenerateOfflineMap_exe()) --in_mesh $nc_name_in --out_mesh $nc_name_out --ov_mesh $nc_name_ol --in_np 1 --out_map $nc_name_wgt`) # FV > FV, parallelized sparse matrix multiply (SparseMatrix.h)
    run(`$(TempestRemap_jll.GenerateOfflineMap_exe()) --in_mesh $nc_name_in --out_mesh $nc_name_out --ov_mesh $nc_name_ol --in_type cgll --out_type cgll --in_np $nq --out_np $nq --out_map $nc_name_wgt`) # GLL > GLL - crashing 

    # apply map (this to be done by CC at each timestep)
    nc_name_data_out = dir*"/Psi_out.nc"
    run(`$(TempestRemap_jll.ApplyOfflineMap_exe()) --map $nc_name_wgt --var Psi --in_data $nc_name_data_in --out_data $nc_name_data_out`)

    # quad = Spaces.Quadratures.GLL{Nq}()
    # space = ClimaCore.Spaces.SpectralElementSpace2D(grid_topology, quad) #float_type(::AbstractPoint{FT}) where {FT} = FT

    
    # coords = ClimaCore.Fields.coordinate_field(space)
    # u = map(coords) do coord
    #     u0 = 20.0
    #     α0 = 45.0
    #     ϕ = coord.lat
    #     λ = coord.long
    
    #     uu = u0 * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
    #     uv = -u0 * sind(α0) * sind(λ)
    #     ClimaCore.Geometry.UVVector(uu, uv)
    # end
    
    # u = u.components.data.:1
    # u_vals=getfield(u, :values) #IJFH : 4,4,1,1,216
end
    
# run(`$(TempestRemap_jll.GenerateCSMesh_exe()) --res 64 --alt --file $nc_name_in`)
# run(`$(TempestRemap_jll.GenerateRLLMesh_exe()) --lon 256 --lat 128 --file $nc_name_out`)
# run(`$(TempestRemap_jll.GenerateOverlapMesh_exe()) --a $nc_name_in --b $nc_name_out --out $nc_name_ol`)
    
# run(`$(TempestRemap_jll.GenerateOfflineMap_exe()) --in_mesh $nc_name_in --out_mesh $nc_name_out --ov_mesh $nc_name_ol --in_np 2 --out_map $nc_name_wgt`) # FV > FV, parallelized sparse matrix multiply (SparseMatrix.h)

# TEMPESTREMAP=.
# $TEMPESTREMAP/GenerateCSMesh --res 64 --alt --file gravitySam.000000.3d.cubedSphere.g
# $TEMPESTREMAP/GenerateRLLMesh --lon 256 --lat 128 --file gravitySam.000000.3d.latLon.g
# $TEMPESTREMAP/GenerateOverlapMesh --a gravitySam.000000.3d.cubedSphere.g --b gravitySam.000000.3d.latLon.g --out gravitySam.000000.3d.overlap.g
# $TEMPESTREMAP/GenerateOfflineMap --in_mesh gravitySam.000000.3d.cubedSphere.g --out_mesh gravitySam.000000.3d.latLon.g --ov_mesh gravitySam.000000.3d.overlap.g --in_np 2 --out_map gravitySam.000000.3d.mapping.nc 
