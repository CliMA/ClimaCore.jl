import ClimaCore
using ClimaComms
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Quadratures
using NCDatasets
using TempestRemap_jll
using Test
using ClimaCoreTempestRemap

OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

@testset "mesh coordinates ClimaCore > TempestRemap" begin
    # generate CC mesh
    ne = 9
    R = 1
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    topology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)
    meshfile_cc = joinpath(OUTPUT_DIR, "test_cc.g")
    write_exodus(meshfile_cc, topology)

    # generate TR mesh
    meshfile_tr = joinpath(OUTPUT_DIR, "test_tr.g")
    run(
        `$(TempestRemap_jll.GenerateCSMesh_exe()) --res $ne --alt --file $meshfile_tr`,
    )
    # tempest remap-generated mesh should have the same number of elements
    @test Meshes.nelements(mesh) ==
          NCDataset(nc -> nc.dim["num_elem"], meshfile_tr)

    # compute overlap mesh
    meshfile_overlap = joinpath(OUTPUT_DIR, "test_overlap.g")
    run(
        `$(TempestRemap_jll.GenerateOverlapMesh_exe()) --a $meshfile_cc --b $meshfile_tr --out $meshfile_overlap`,
    )
    # the overlap mesh should have the same number of elements as the input meshes
    @test Meshes.nelements(mesh) ==
          NCDataset(nc -> nc.dim["num_elem"], meshfile_overlap)

    # compute weight file
    weightfile = joinpath(OUTPUT_DIR, "test_weight.nc")
    run(
        `$(TempestRemap_jll.GenerateOfflineMap_exe()) --in_mesh $meshfile_cc --out_mesh $meshfile_tr --ov_mesh $meshfile_overlap --in_np 1 --out_map $weightfile`,
    )
    # fractions should be close to 1
    frac_a = NCDataset(nc -> Array(nc["frac_a"]), weightfile)
    @test all(x -> x ≈ 1, frac_a)

    # compute overlap
    run(
        `$(TempestRemap_jll.GenerateOfflineMap_exe()) --in_mesh $meshfile_cc --out_mesh $meshfile_tr --ov_mesh $meshfile_overlap --in_np 2 --out_map $weightfile`,
    )
end
