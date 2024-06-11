include("utils_3d.jl")
device = ClimaComms.device()

@testset "sphere divergence" begin
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(1.0);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 10)
    verttopology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        vertmesh,
    )
    vertgrid = Grids.FiniteDifferenceGrid(verttopology)

    radius = 30.0
    horzdomain = Domains.SphereDomain(radius)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, 4)
    horztopology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(device),
        horzmesh,
    )
    quad = Quadratures.GLL{3 + 1}()
    horzgrid = Grids.SpectralElementGrid2D(horztopology, quad)

    # shallow
    shallow_grid = Grids.ExtrudedFiniteDifferenceGrid(horzgrid, vertgrid)

    coords = Fields.coordinate_field(
        Spaces.CenterExtrudedFiniteDifferenceSpace(shallow_grid),
    )
    x = Geometry.UVWVector.(cosd.(coords.lat), 0.0, 0.0)
    div = Operators.Divergence()
    @test div.(x) ≈ zero(coords.z) atol = 1e-4

    fcoords = Fields.coordinate_field(
        Spaces.FaceExtrudedFiniteDifferenceSpace(shallow_grid),
    )
    y = map(coord -> Geometry.WVector(0.7), fcoords)
    divf2c = Operators.DivergenceF2C()
    @test divf2c.(y) ≈ zero(coords.z) atol = 100 * eps(FT)

    # deep
    deep_grid =
        Grids.ExtrudedFiniteDifferenceGrid(horzgrid, vertgrid; deep = true)

    coords = Fields.coordinate_field(
        Spaces.CenterExtrudedFiniteDifferenceSpace(deep_grid),
    )
    x = Geometry.UVWVector.(cosd.(coords.lat), 0.0, 0.0)
    div = Operators.Divergence()
    @test div.(x) ≈ zero(coords.z) atol = 1e-4

    # divergence of a constant outward vector field = 2 w / (r + z)
    fcoords = Fields.coordinate_field(
        Spaces.FaceExtrudedFiniteDifferenceSpace(deep_grid),
    )
    y = map(coord -> Geometry.WVector(0.7), fcoords)
    divf2c = Operators.DivergenceF2C()
    @test divf2c.(y) ≈ (2 * 0.7) ./ (radius .+ coords.z) atol = 100 * eps(FT)
end

@testset "2D SE, 1D FD Extruded Domain level extraction" begin
    hv_center_space, hv_face_space = hvspace_3D()

    coord = Fields.coordinate_field(hv_face_space)

    @test parent(Fields.field_values(level(coord.x, half))) == parent(
        Fields.field_values(
            Fields.coordinate_field(Spaces.horizontal_space(hv_face_space)).x,
        ),
    )
    @test parent(Fields.field_values(level(coord.z, half))) ==
          parent(
        Fields.field_values(
            Fields.coordinate_field(Spaces.horizontal_space(hv_face_space)).x,
        ),
    ) .* 0
end

@testset "bycolumn fuse" begin
    hv_center_space, hv_face_space =
        hvspace_3D((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    fz = Fields.coordinate_field(hv_face_space).z
    ∇ = Operators.GradientF2C()
    ∇z = map(coord -> WVector(0.0), Fields.coordinate_field(hv_center_space))
    Fields.bycolumn(hv_center_space) do colidx
        @. ∇z[colidx] = WVector(∇(fz[colidx]))
    end
    @test ∇z == WVector.(∇.(fz))
end
