import ClimaCore
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Fields
using NCDatasets
using TempestRemap_jll
using Test
using ClimaCoreTempestRemap

OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

@testset "write remap sphere data $node_type" for node_type in ["cgll", "dgll"]
    # generate CC mesh
    ne = 4
    R = 5.0
    Nq = 5
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(topology, quad)
    coords = Fields.coordinate_field(space)

    # write mesh
    meshfile_cc = joinpath(OUTPUT_DIR, "mesh_cc.g")
    write_exodus(meshfile_cc, topology)

    # write data
    datafile_cc = joinpath(OUTPUT_DIR, "data_cc.nc")
    NCDataset(datafile_cc, "c") do nc
        def_space_coord(nc, space; type = node_type)
        nc_time = def_time_coord(nc)

        nc_xlat = defVar(nc, "xlat", Float64, space)
        nc_sinlong = defVar(nc, "sinlong", Float64, space, ("time",))

        nc_xlat[:] = Fields.coordinate_field(space).lat
        for (t_i, t) in enumerate(0:30:180)
            nc_time[t_i] = t
            nc_sinlong[:, t_i] = sind.(Fields.coordinate_field(space).long .+ t)
        end
        nothing
    end

    nlat = 90
    nlon = 180
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
        in_type = node_type,
        in_np = Nq,
    )

    datafile_rll = joinpath(OUTPUT_DIR, "data_rll.nc")
    apply_remap(datafile_rll, datafile_cc, weightfile, ["xlat", "sinlong"])

    NCDataset(datafile_rll) do nc_rll
        lats = Array(nc_rll["lat"])
        lons = Array(nc_rll["lon"])
        @test lats ≈ -89:2:89
        @test lons ≈ 1:2:359
        @test Array(nc_rll["xlat"]) ≈ ones(nlon) * lats' rtol = 0.1
        @test nc_rll["sinlong"][:, :, 1] ≈ sind.(lons) * ones(nlat)' rtol = 0.1
        @test nc_rll["sinlong"][:, :, 2] ≈ sind.(lons .+ 30) * ones(nlat)' rtol =
            0.1
    end
end

@testset "write remap 3d sphere data $node_type" for node_type in
                                                     ["cgll", "dgll"]
    # generate CC mesh
    ne = 4
    R = 1000.0
    nlevels = 10
    Nq = 5
    hdomain = Domains.SphereDomain(R)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, ne)
    htopology = Topologies.Topology2D(hmesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)

    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(50.0);
        boundary_tags = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = nlevels)
    vspace = Spaces.CenterFiniteDifferenceSpace(vmesh)

    hvspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    fhvspace = Spaces.FaceExtrudedFiniteDifferenceSpace(hvspace)

    # write mesh
    meshfile_cc = joinpath(OUTPUT_DIR, "mesh_cc_3d.g")
    write_exodus(meshfile_cc, htopology)

    # write data
    datafile_cc = joinpath(OUTPUT_DIR, "data_cc.nc")
    NCDataset(datafile_cc, "c") do nc
        def_space_coord(nc, hvspace; type = node_type)
        def_space_coord(nc, fhvspace; type = node_type)

        nc_xlat = defVar(nc, "xlat", Float64, hvspace)
        nc_xz = defVar(nc, "xz", Float64, hvspace)
        nc_xz_half = defVar(nc, "xz_half", Float64, fhvspace)

        nc_xlat[:] = Fields.coordinate_field(hvspace).lat
        nc_xz[:] = Fields.coordinate_field(hvspace).z
        nc_xz_half[:] = Fields.coordinate_field(fhvspace).z
        nothing
    end

    nlat = 90
    nlon = 180
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
        in_type = node_type,
        in_np = Nq,
    )

    datafile_rll = joinpath(OUTPUT_DIR, "data_rll_3d.nc")
    apply_remap(datafile_rll, datafile_cc, weightfile, ["xlat", "xz"])

    NCDataset(datafile_rll) do nc_rll
        lats = Array(nc_rll["lat"])
        lons = Array(nc_rll["lon"])
        zs = Array(nc_rll["z"])
        @test lats ≈ -89:2:89
        @test lons ≈ 1:2:359
        @test zs == 2.5:5:47.5
        @test Array(nc_rll["xlat"]) ≈ ones(nlon, nlat, nlevels) .* lats' rtol =
            0.1
        @test Array(nc_rll["xz"]) ≈
              ones(nlon, nlat, nlevels) .* reshape(zs, (1, 1, nlevels)) rtol =
            0.1
    end
end
