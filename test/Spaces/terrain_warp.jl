using Test
using Adapt
using ClimaComms
using IntervalSets

import ClimaCore:
    ClimaCore,
    Domains,
    Geometry,
    Grids,
    Fields,
    Operators,
    Meshes,
    Spaces,
    Quadratures,
    Topologies,
    Hypsography

using ClimaCore.Utilities: half

function warp_sin_2d(coord)
    x = Geometry.component(coord, 1)
    eltype(x)(0.5) * sin(x)
end
function warp_sinsq_2d(coord)
    x = Geometry.component(coord, 1)
    eltype(x)(0.5) * sin(x)^2
end
function flat_test_2d(coord)
    x = Geometry.component(coord, 1)
    eltype(x)(0) * sin(x)
end
function warp_sincos_3d(coord)
    x = Geometry.component(coord, 1)
    y = Geometry.component(coord, 2)
    eltype(x)(0.5) * sin(x)^2 * cos(y)^2
end
function warp_sinsq_3d(coord)
    x = Geometry.component(coord, 1)
    y = Geometry.component(coord, 2)
    eltype(x)(0.5) * sin(x)^2 * sin(y)^2
end
function generate_base_spaces(
    xlim,
    zlim,
    helem,
    velem,
    npoly,
    stretch = Meshes.Uniform();
    ndims = 3,
)
    device = ClimaComms.CUDADevice()
    comms_context = ClimaComms.SingletonCommsContext(device)
    FT = eltype(xlim)

    # Horizontal Grid Construction
    quad = Quadratures.GLL{npoly + 1}()
    if ndims == 2
	horzdomain = Domains.IntervalDomain(
            Geometry.XPoint{FT}(xlim[1]),
            Geometry.XPoint{FT}(xlim[2]);
            periodic = true,
        )
        horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
        horz_topology = Topologies.IntervalTopology(comms_context, horzmesh)
    	h_space = Spaces.SpectralElementSpace1D(horz_topology, quad);
    elseif ndims == 3
        horzdomain = Domains.RectangleDomain(
            Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
            Geometry.YPoint{FT}(xlim[1]) .. Geometry.YPoint{FT}(xlim[2]),
            x1periodic = true,
            x2periodic = true,
        )
        # Assume same number of elems (helem) in (x,y) directions
        horzmesh = Meshes.RectilinearMesh(horzdomain, helem, helem)
	horz_topology = Topologies.Topology2D(comms_context,
                                              horzmesh,
                                              Topologies.spacefillingcurve(horzmesh));
	h_space = Spaces.SpectralElementSpace2D(horz_topology, quad, enable_bubble=true);
    end
    
    horz_grid = Spaces.grid(h_space)

    # Vertical Grid Construction
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = velem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)
    vert_topology = Topologies.IntervalTopology(ClimaComms.SingletonCommsContext(device), vertmesh)
    vert_grid = Grids.FiniteDifferenceGrid(vert_topology)


    ArrayType = ClimaComms.array_type(device)
    horz_grid = Adapt.adapt(ArrayType, horz_grid)
    vert_grid = Adapt.adapt(ArrayType, vert_grid)
    adaption = Adapt.adapt(ArrayType, Hypsography.Flat())

    grid = Grids.ExtrudedFiniteDifferenceGrid(horz_grid,
                                              vert_grid,
                                              adaption;
                                              )

    cent_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)

    return h_space, face_space, horz_grid, vert_grid
end
function generate_smoothed_orography(
    h_space,
    warp_fn::Function,
    helem;
    test_smoothing::Bool = false,
)
    # Extrusion
    z_surface = warp_fn.(Fields.coordinate_field(h_space))
    # An Euler step defines the diffusion coefficient 
    # (See e.g. cfl condition for diffusive terms).
    x_array = parent(Fields.coordinate_field(h_space).x)
    dx = x_array[2] - x_array[1]
    FT = eltype(x_array)
    κ = FT(1 / helem)
    test_smoothing ?
    Hypsography.diffuse_surface_elevation!(
        z_surface;
        κ,
        maxiter = 10^5,
        dt = FT(dx / 100),
    ) : nothing
    return z_surface
end

function get_adaptation(adaption, z_surface::Fields.Field)
    if adaption <: Hypsography.LinearAdaption
        return adaption(z_surface)
    elseif adaption <: Hypsography.SLEVEAdaption
        return adaption(
            z_surface,
            eltype(z_surface.z)(0.75),
            eltype(z_surface.z)(0.60),
        )
    end
end

function warpedspace_2D(
    FT = Float64,
    xlim = (0, π),
    zlim = (0, 1),
    helem = 2,
    velem = 10,
    npoly = 5,
    stretch = Meshes.Uniform();
    warp_fn = warp_sin_2d,
    test_smoothing = false,
    adaption = Hypsography.LinearAdaption,
)
    h_space, vert_face_space, horz_grid, vert_grid = 
        generate_base_spaces(xlim, zlim, helem, velem, npoly, ndims = 2)
    z_surface =
        Geometry.ZPoint.(generate_smoothed_orography(h_space, warp_fn, helem; test_smoothing))
    grid_adapt = get_adaptation(adaption, z_surface)
    grid = Grids.ExtrudedFiniteDifferenceGrid(horz_grid, vert_grid, grid_adapt)
    c_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
    f_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)
    return (c_space, f_space)
end
function hybridspace_2D(
    FT = Float64,
    xlim = (0, π),
    zlim = (0, 1),
    helem = 2,
    velem = 10,
    npoly = 5;
    stretch = Meshes.Uniform(),
)
    h_space, vert_face_space, horz_grid, vert_grid = 
        generate_base_spaces(xlim, zlim, helem, velem, npoly, ndims = 2)
    grid = Grids.ExtrudedFiniteDifferenceGrid(horz_grid, vert_grid, Hypsography.Flat())
    c_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
    f_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)
    return (c_space, f_space)
end
function warpedspace_3D(
    FT = Float64,
    xlim = (0, π),
    ylim = (0, π),
    zlim = (0, 1),
    helem = 2,
    velem = 10,
    npoly = 5;
    stretch = Meshes.Uniform(),
    warp_fn = warp_sincos_3d,
    test_smoothing = false,
    adaption = Hypsography.LinearAdaption,
)
    h_space, vert_face_space, horz_grid, vert_grid = 
        generate_base_spaces(xlim, zlim, helem, velem, npoly, ndims = 3)
    z_surface =
        Geometry.ZPoint.(generate_smoothed_orography(h_space, warp_fn, helem; test_smoothing))
    grid_adapt = get_adaptation(adaption, z_surface)
    grid = Grids.ExtrudedFiniteDifferenceGrid(horz_grid, vert_grid, grid_adapt)
    c_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
    f_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)
    return (c_space, f_space)
end

# 2D Tests
@testset "2D Extruded Terrain Warped Space" begin
    # Generated "negative space" should be unity
    for FT in (Float32, Float64)
        # Extruded FD-Spectral Hybrid
        xmin, xmax = FT(0), FT(π)
        zmin, zmax = FT(0), FT(1)
        levels = 5:10
        polynom = 2:2:10
        horzelem = 2:2:10
        for nl in levels, np in polynom, nh in horzelem
            ʷhv_center_space, ʷhv_face_space =
                warpedspace_2D(FT, (xmin, xmax), (zmin, zmax), nh, nl, np;)
            ʷᶜcoords = Fields.coordinate_field(ʷhv_center_space)
            ʷᶠcoords = Fields.coordinate_field(ʷhv_face_space)
            z₀ = ClimaCore.Fields.level(ʷᶜcoords.z, 1)
            # Check ∫ₓ(z_sfc)dx == known value from warp_sin_2d
            @test sum(z₀ .- zmax / 2nl) - FT(1) <= FT(0.1 / np * nh * nl)
            @test abs(maximum(z₀) - FT(0.5)) <= FT(0.125)
        end
    end
end

@testset "2D Extruded Terrain Laplacian Smoothing" begin
    # Test smoothing for known parameters
    for FT in (Float32, Float64)
        # Extruded FD-Spectral Hybrid
        xmin, xmax = FT(0), FT(π)
        zmin, zmax = FT(0), FT(1)
        levels = [5, 10]
        polynom = 3:2:10
        horzelem = 5:2:10
        for nl in levels, np in polynom, nh in horzelem
            # Test Against Steady State Analytical Solution
            ʷhv_center_space, ʷhv_face_space = warpedspace_2D(
                FT,
                (xmin, xmax),
                (zmin, zmax),
                nh,
                nl,
                np;
                warp_fn = warp_sinsq_2d,
                test_smoothing = true,
            )
            ʳhv_center_space, ʳhv_face_space = warpedspace_2D(
                FT,
                (xmin, xmax),
                (zmin, zmax),
                nh,
                nl,
                np;
                warp_fn = warp_sinsq_2d,
                test_smoothing = false,
            )
            ʷᶠcoords = Fields.coordinate_field(ʷhv_face_space)
            ʷᶠʳcoords = Fields.coordinate_field(ʳhv_face_space)
            ᶠz₀ = ClimaCore.Fields.level(ʷᶠcoords.z, half)
            @test minimum(ᶠz₀) >= FT(0)
            @test maximum(ᶠz₀) <= FT(0.5)
            @test maximum(@. abs.(ᶠz₀ .- one(ᶠz₀) .* FT.(1 / 4))) <= FT(1e-2)
        end
    end
end

@testset "2D Warped Mesh RHS Integration Test" begin
    for FT in (Float64,)
        xmin, xmax = FT(0), FT(π)
        zmin, zmax = FT(0), FT(1)
        levels = 10
        polynom = 4
        horzelem = 5
        ⁿhv_center_space, ⁿhv_face_space = warpedspace_2D(
            FT,
            (xmin, xmax),
            (zmin, zmax),
            horzelem,
            levels,
            polynom;
            warp_fn = flat_test_2d,
        )
        ⁿᶜcoords = Fields.coordinate_field(ⁿhv_center_space)
        ⁿᶠcoords = Fields.coordinate_field(ⁿhv_face_space)

        uₕ = map(_ -> Geometry.UVector(1.0), ⁿᶜcoords)
        w = map(_ -> Geometry.WVector(0.0), ⁿᶠcoords)

        uₕ = @. Geometry.Covariant1Vector(uₕ)
        w = @. Geometry.Covariant3Vector(w)
        Y = Fields.FieldVector(uₕ = uₕ, w = w)
        dY = similar(Y)
        function rhs(dY, Y, _, t)
            dY.uₕ = uₕ
            dY.w = w
            Spaces.weighted_dss!(dY.uₕ)
            Spaces.weighted_dss!(dY.w)
            return (dY, Y)
        end
        (dY, Y) = rhs(dY, Y, nothing, 0.0)
        @test maximum(
            abs.(dY.uₕ.components.data.:1 .- uₕ.components.data.:1),
        ) <= eps(FT)
        @test maximum(abs.(dY.w.components.data.:1 .- w.components.data.:1)) <=
              eps(FT)
    end
end

# 3D Tests
@testset "3D Extruded Terrain Warped Space" begin
    # Generated "negative space" should be unity
    for FT in (Float32, Float64)
        # Extruded FD-Spectral Hybrid
        xmin, xmax = FT(0), FT(π)
        ymin, ymax = FT(0), FT(π)
        zmin, zmax = FT(0), FT(1)
        levels = 5:10
        polynom = 2:2:10
        horzelem = 2:2:10
        for nl in levels, np in polynom, nh in horzelem
            hv_center_space, hv_face_space = warpedspace_3D(
                FT,
                (xmin, xmax),
                (ymin, ymax),
                (zmin, zmax),
                nh,
                nl,
                np;
            )
            ᶜcoords = Fields.coordinate_field(hv_center_space)
            ᶠcoords = Fields.coordinate_field(hv_face_space)
            z₀ = ClimaCore.Fields.level(ᶜcoords.z, 1)
            # Check ∫ₛ(z_sfc)dS == known value from warp_sincos_3d
            # Assumes uniform stretching
            @test sum(z₀ .- zmax / 2nl) - FT(π^2 / 8) <= FT(0.1 / np * nh * nl)
            @test abs(maximum(z₀) - FT(0.5)) <= FT(0.125)
        end
    end
end

@testset "3D Extruded Terrain Laplacian Smoothing" begin
    # Test smoothing for known parameters
    for FT in (Float32, Float64)
        # Extruded FD-Spectral Hybrid
        xmin, xmax = FT(0), FT(π)
        ymin, ymax = FT(0), FT(π)
        zmin, zmax = FT(0), FT(1)
        levels = [5]
        polynom = 3:2:10
        horzelem = 5:2:10
        for nl in levels, np in polynom, nh in horzelem
            # Test Against Steady State Analytical Solution
            ʷhv_center_space, ʷhv_face_space = warpedspace_3D(
                FT,
                (xmin, xmax),
                (ymin, ymax),
                (zmin, zmax),
                nh,
                nl,
                np;
                warp_fn = warp_sinsq_3d,
                test_smoothing = true,
            )
            ʳhv_center_space, ʳhv_face_space = warpedspace_3D(
                FT,
                (xmin, xmax),
                (ymin, ymax),
                (zmin, zmax),
                nh,
                nl,
                np;
                warp_fn = warp_sinsq_3d,
                test_smoothing = false,
            )
            ʷᶠcoords = Fields.coordinate_field(ʷhv_face_space)
            ʷᶠʳcoords = Fields.coordinate_field(ʳhv_face_space)
            ᶠz₀ = ClimaCore.Fields.level(ʷᶠcoords.z, half)
            @test minimum(ᶠz₀) >=
                  minimum(ClimaCore.Fields.level(ʷᶠʳcoords.z, half))
            @test maximum(ᶠz₀) <=
                  maximum(ClimaCore.Fields.level(ʷᶠʳcoords.z, half))
            @test maximum(@. abs.(ᶠz₀ .- one(ᶠz₀) .* FT.(1 / 8))) <= FT(1e-2)
        end
    end
end

@testset "Interior Mesh `Adaption` ηₕ Test" begin
    # Test interior mesh in different adaptation types
    for meshadapt in (Hypsography.SLEVEAdaption,)
        for FT in (Float32, Float64)
            xmin, xmax = FT(0), FT(π)
            zmin, zmax = FT(0), FT(1)
            nl = 10
            np = 3
            nh = 4
            ʷhv_center_space, ʷhv_face_space = warpedspace_2D(
                FT,
                (xmin, xmax),
                (zmin, zmax),
                nh,
                nl,
                np;
                warp_fn = warp_sin_2d,
                adaption = meshadapt,
            )
            hv_center_space, hv_face_space =
                hybridspace_2D(FT, (xmin, xmax), (zmin, zmax), nh, nl, np)
            ʷᶜcoords = Fields.coordinate_field(ʷhv_center_space)
            ʷᶠcoords = Fields.coordinate_field(ʷhv_face_space)
            ᶜcoords = Fields.coordinate_field(hv_center_space)
            ᶠcoords = Fields.coordinate_field(hv_face_space)
            # Check ηₛ = 0.75 is correctly applied. 
            # Expectation: ≈zero difference between unwarped and warped coordinates for η >= ηₕ, where η = z / zₜ
            r1 =
                (
                    parent(ʷᶜcoords)[8:10, :, 2, :] .-
                    parent(ᶜcoords)[8:10, :, 2, :]
                ) ./ parent(ᶜcoords)[8:10, :, 2, :]
            @test maximum(r1) <= FT(0.015)
        end
    end
end

@testset "Interior Mesh `Adaption` (ηₕ=1, s=1) Test" begin
    # Test interior mesh in different adaptation types
    for meshadapt in (Hypsography.SLEVEAdaption,)
        for FT in (Float32, Float64)
	    device = ClimaComms.CUDADevice()
	    comms_context = ClimaComms.SingletonCommsContext(device)

            xlim = (FT(0), FT(π))
            zlim = (FT(0), FT(1))
            nl = 10
            np = 3
            nh = 4

	    quad = Quadratures.GLL{np + 1}()
		horzdomain = Domains.IntervalDomain(
		    Geometry.XPoint{FT}(xlim[1]),
		    Geometry.XPoint{FT}(xlim[2]);
		    periodic = true,
		)
		horzmesh = Meshes.IntervalMesh(horzdomain; nelems =  nh)
		horz_topology = Topologies.IntervalTopology(comms_context, horzmesh)
		h_space = Spaces.SpectralElementSpace1D(horz_topology, quad);

            # Generate surface elevation profile
            z_surface = warp_sin_2d.(Fields.coordinate_field(h_space))
	    horz_grid = Spaces.grid(h_space)

	    # Vertical Grid Construction
	    vertdomain = Domains.IntervalDomain(
		Geometry.ZPoint{FT}(zlim[1]),
		Geometry.ZPoint{FT}(zlim[2]);
		boundary_tags = (:bottom, :top),
	    )
	    vertmesh = Meshes.IntervalMesh(vertdomain, Meshes.Uniform(), nelems = nl)
	    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)
	    vert_topology = Topologies.IntervalTopology(ClimaComms.SingletonCommsContext(device), vertmesh)
	    vert_grid = Grids.FiniteDifferenceGrid(vert_topology)

	    grid = Grids.ExtrudedFiniteDifferenceGrid(horz_grid,
						      vert_grid,
						      Hypsography.SLEVEAdaption(z_surface, FT(1), FT(1));
						      )

	    cspace = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
	    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)

            for i in 1:(nl + 1)
                z_extracted = Fields.Field(
                    Fields.level(fspace.face_local_geometry.coordinates.z, i),
                    fspace,
                )
                η = FT((i - 1) / 10)
                z_surface_known =
                    @. FT(η) + z_surface * FT(sinh(1 - η) / sinh(1))
                @test maximum(
                    abs.(
                        Fields.field_values(z_extracted) .-
                        Fields.field_values(z_surface_known)
                    ),
                ) <= FT(1e-6)
            end
        end
    end
end
@testset "Interior Mesh `Adaption`: Test Error" begin
    # Test interior mesh in different adaptation types
    for meshadapt in (Hypsography.SLEVEAdaption,)
        for FT in (Float32, Float64)
	    device = ClimaComms.CUDADevice()
	    comms_context = ClimaComms.SingletonCommsContext(device)

            xlim = (FT(0), FT(π))
            zlim = (FT(0), FT(1))
            nl = 10
            np = 3
            nh = 4

	    quad = Quadratures.GLL{np + 1}()
		horzdomain = Domains.IntervalDomain(
		    Geometry.XPoint{FT}(xlim[1]),
		    Geometry.XPoint{FT}(xlim[2]);
		    periodic = true,
		)
		horzmesh = Meshes.IntervalMesh(horzdomain; nelems =  nh)
		horz_topology = Topologies.IntervalTopology(comms_context, horzmesh)
		h_space = Spaces.SpectralElementSpace1D(horz_topology, quad);

            # Generate surface elevation profile
            z_surface = warp_sin_2d.(Fields.coordinate_field(h_space))
	    horz_grid = Spaces.grid(h_space)

	    # Vertical Grid Construction
	    vertdomain = Domains.IntervalDomain(
		Geometry.ZPoint{FT}(zlim[1]),
		Geometry.ZPoint{FT}(zlim[2]);
		boundary_tags = (:bottom, :top),
	    )
	    vertmesh = Meshes.IntervalMesh(vertdomain, Meshes.Uniform(), nelems = nl)
	    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)
	    vert_topology = Topologies.IntervalTopology(ClimaComms.SingletonCommsContext(device), vertmesh)
	    vert_grid = Grids.FiniteDifferenceGrid(vert_topology)
	    @test_throws ErrorException Grids.ExtrudedFiniteDifferenceGrid(horz_grid,
						      vert_grid,
						      Hypsography.SLEVEAdaption(z_surface, FT(1), FT(0.1));
						      )
        end
    end
end
