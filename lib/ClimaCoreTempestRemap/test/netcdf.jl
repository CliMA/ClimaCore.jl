import ClimaCore
using ClimaCore:
    Geometry, Meshes, Domains, Topologies, Spaces, Fields, Hypsography
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

@testset "write remap 3d sphere data $node_type to rll and back" for node_type in
                                                                     [
    "cgll",
    "dgll",
]
    # generate CC mesh
    ne = 4
    R = 1000.0
    nlevels = 10
    Nq = 4
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
        mono = true,
    )

    datafile_rll = joinpath(OUTPUT_DIR, "data_rll_3d.nc")
    apply_remap(datafile_rll, datafile_cc, weightfile, ["xlat", "xz"])

    NCDataset(weightfile) do weights
        # test monotonicity
        @test maximum(weights["S"]) <= 1
        @test minimum(weights["S"]) >= 0 ||
              isapprox(minimum(weights["S"]), 0; atol = 10^(-15))
    end

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

    # convert newly-created rll data to new node_type (cgll/dgll)
    # write mesh
    meshfile_cc_post = joinpath(OUTPUT_DIR, "mesh_cc_3d_post.g")
    write_exodus(meshfile_cc_post, htopology)

    meshfile_overlap_post = joinpath(OUTPUT_DIR, "mesh_overlap_post.g")
    overlap_mesh(meshfile_overlap_post, meshfile_rll, meshfile_cc_post)

    weightfile_post = joinpath(OUTPUT_DIR, "remap_weights_post.nc")
    remap_weights(
        weightfile_post,
        meshfile_rll,
        meshfile_cc_post,
        meshfile_overlap_post;
        out_type = node_type,
        out_np = Nq - 1,
        in_np = 1,
        mono = true,
    )

    datafile_cc_post = joinpath(OUTPUT_DIR, "data_cc_post.nc")
    apply_remap(datafile_cc_post, datafile_rll, weightfile_post, ["xlat", "xz"])

    NCDataset(weightfile_post) do weights_post
        # test monotonicity
        @test maximum(weights_post["S"]) <= 1
        @test minimum(weights_post["S"]) >= 0 ||
              isapprox(minimum(weights_post["S"]), 0; atol = 10^(-15))
    end
end

function test_warp(coords)
    λ, ϕ = coords.long, coords.lat
    FT = eltype(λ)
    ϕₘ = FT(0) # degrees (equator)
    λₘ = FT(3 / 2 * 180)  # degrees
    rₘ = @. FT(acos(sind(ϕₘ) * sind(ϕ) + cosd(ϕₘ) * cosd(ϕ) * cosd(λ - λₘ))) # Great circle distance (rads)
    Rₘ = FT(3π / 4) # Moutain radius
    ζₘ = FT(π / 16) # Mountain oscillation half-width
    h₀ = FT(10)
    if rₘ < Rₘ
        zₛ = FT(h₀ / 2) * (1 + cospi(rₘ / Rₘ))
    else
        zₛ = @. FT(0)
    end
    return zₛ
end

@testset "write remap warped 3d sphere data $node_type" for node_type in
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
    vfspace = Spaces.FaceFiniteDifferenceSpace(vmesh)
    z_surface = test_warp.(Fields.coordinate_field(hspace))
    fhvspace = Spaces.ExtrudedFiniteDifferenceSpace(
        hspace,
        vfspace,
        Hypsography.LinearAdaption(),
        z_surface,
    )
    chvspace = Spaces.CenterExtrudedFiniteDifferenceSpace(fhvspace)

    # write mesh
    meshfile_cc = joinpath(OUTPUT_DIR, "mesh_cc_3d.g")
    write_exodus(meshfile_cc, htopology)

    # write data
    datafile_cc = joinpath(OUTPUT_DIR, "data_cc.nc")
    NCDataset(datafile_cc, "c") do nc
        def_space_coord(nc, chvspace; type = node_type)
        def_space_coord(nc, fhvspace; type = node_type)

        nc_xlat = defVar(nc, "xlat", Float64, chvspace)
        nc_xz = defVar(nc, "xz", Float64, chvspace)
        nc_xz_half = defVar(nc, "xz_half", Float64, fhvspace)

        nc_xlat[:] = Fields.coordinate_field(chvspace).lat
        nc_xz[:] = Fields.coordinate_field(chvspace).z
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
    apply_remap(
        datafile_rll,
        datafile_cc,
        weightfile,
        ["xlat", "xz", "xz_half"],
    )

    NCDataset(datafile_rll) do nc_rll
        lats = Array(nc_rll["lat"])
        lons = Array(nc_rll["lon"])
        zs = Array(nc_rll["z"])
        xzs = Array(nc_rll["xz"])
        xzhalfs = Array(nc_rll["xz_half"])
        @test size(xzs) == (180, 90, 10)
        @test size(xzhalfs) == (180, 90, 11)
        @test lats ≈ -89:2:89
        @test lons ≈ 1:2:359
        @test maximum(xzs[:, :, 1]) ≈ 12.0 rtol = 0.1
        @test maximum(xzhalfs[:, :, 1]) ≈ 10.0 rtol = 0.1
        @test Array(nc_rll["xlat"]) ≈ ones(nlon, nlat, nlevels) .* lats' rtol =
            0.1
        @test Array(nc_rll["xz"]) ≈
              ones(nlon, nlat, nlevels) .* reshape(zs, (1, 1, nlevels)) rtol =
            0.1
    end
end

@testset "write nc data for column with $node_type" for node_type in
                                                        ["cgll", "dgll"]

    node_type = "cgll"
    FT = Float64
    x_max = FT(0)
    y_max = FT(0)
    x_elem = 1
    y_elem = 1
    # generate CC mesh
    nlevels = 10
    x_domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint(zero(y_max)),
        Geometry.YPoint(y_max);
        periodic = true,
    )
    domain = Domains.RectangleDomain(x_domain, y_domain)
    hmesh = Meshes.RectilinearMesh(domain, x_elem, y_elem)

    quad = Spaces.Quadratures.GL{1}()
    htopology = Topologies.Topology2D(hmesh)
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

    # write data
    datafile_cc = joinpath(OUTPUT_DIR, "data_cc.nc")
    NCDataset(datafile_cc, "c") do nc
        def_space_coord(nc, hvspace; type = node_type)
        def_space_coord(nc, fhvspace; type = node_type)

        nc_x = defVar(nc, "xx", Float64, hvspace)
        nc_z = defVar(nc, "xz", Float64, hvspace)
        nc_z_half = defVar(nc, "xz_half", Float64, fhvspace)

        nc_x[:] = Fields.coordinate_field(hvspace).x
        nc_z[:] = Fields.coordinate_field(hvspace).z
        nc_z_half[:] = Fields.coordinate_field(fhvspace).z
        nothing
    end
    @test isfile(datafile_cc)
    rm(datafile_cc; force = true)
end
