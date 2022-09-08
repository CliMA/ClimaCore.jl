using Test

import ClimaCore:
    ClimaCore,
    Domains,
    Geometry,
    Fields,
    Operators,
    Meshes,
    Spaces,
    Topologies,
    Hypsography

function warp_test_2d(coord)
    x = Geometry.component(coord, 1)
    eltype(x)(0.5) * sin(x)
end
function nowarp_test_2d(coord)
    x = Geometry.component(coord, 1)
    eltype(x)(0) * sin(x)
end
function warp_test_3d(coord)
    x = Geometry.component(coord, 1)
    y = Geometry.component(coord, 2)
    eltype(x)(0.5) * sin(x)^2 * cos(y)^2
end
function warpedspace_2D(
    FT = Float64,
    xlim = (0, π),
    zlim = (0, 1),
    helem = 2,
    velem = 10,
    npoly = 5;
    stretch = Meshes.Uniform(),
    warp_fn = warp_test_2d,
)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = velem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)

    # Generate Horizontal Space
    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    # Extrusion
    z_surface = warp_fn.(Fields.coordinate_field(hspace))
    f_space = Spaces.ExtrudedFiniteDifferenceSpace(
        hspace,
        vert_face_space,
        Hypsography.LinearAdaption(z_surface),
    )
    c_space = Spaces.CenterExtrudedFiniteDifferenceSpace(f_space)

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
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = velem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)

    # Generate Horizontal Space
    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    # Extrusion
    f_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vert_face_space)
    c_space = Spaces.CenterExtrudedFiniteDifferenceSpace(f_space)

    return (c_space, f_space)
end

function warpedspace_3D(
    FT = Float64,
    xlim = (0, π),
    ylim = (0, π),
    zlim = (0, 1),
    xelem = 2,
    yelem = 2,
    zelem = 10,
    npoly = 5;
    stretch = Meshes.Uniform(),
    warp_fn = warp_test_3d,
)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = zelem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)

    # Generate Horizontal Space
    xdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    ydomain = Domains.IntervalDomain(
        Geometry.YPoint{FT}(ylim[1]),
        Geometry.YPoint{FT}(ylim[2]),
        periodic = true,
    )
    horzdomain = Domains.RectangleDomain(xdomain, ydomain)
    horzmesh = Meshes.RectilinearMesh(horzdomain, xelem, yelem)
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    # Extrusion
    z_surface = warp_fn.(Fields.coordinate_field(hspace))
    f_space = Spaces.ExtrudedFiniteDifferenceSpace(
        hspace,
        vert_face_space,
        Hypsography.LinearAdaption(z_surface),
    )
    c_space = Spaces.CenterExtrudedFiniteDifferenceSpace(f_space)

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
            hv_center_space, hv_face_space =
                hybridspace_2D(FT, (xmin, xmax), (zmin, zmax), nh, nl, np)
            ⁿhv_center_space, ⁿhv_face_space = warpedspace_2D(
                FT,
                (xmin, xmax),
                (zmin, zmax),
                nh,
                nl,
                np;
                warp_fn = warp_test_2d,
            )
            ʷᶜcoords = Fields.coordinate_field(ʷhv_center_space)
            ʷᶠcoords = Fields.coordinate_field(ʷhv_face_space)

            z₀ = ClimaCore.Fields.level(ʷᶜcoords.z, 1)
            # Check ∫ₓ(z_sfc)dx == known value from warp_test_2d
            @test sum(z₀ .- zmax / 2nl) - FT(1) <= FT(0.1 / np * nh * nl)
            @test abs(maximum(z₀) - FT(0.5)) <= FT(0.125)
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
            warp_fn = nowarp_test_2d,
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
                nh,
                nl,
                np;
            )
            ᶜcoords = Fields.coordinate_field(hv_center_space)
            ᶠcoords = Fields.coordinate_field(hv_face_space)
            z₀ = ClimaCore.Fields.level(ᶜcoords.z, 1)
            # Check ∫ₛ(z_sfc)dS == known value from warp_test_3d
            # Assumes uniform stretching
            @test sum(z₀ .- zmax / 2nl) - FT(π^2 / 8) <= FT(0.1 / np * nh * nl)
            @test abs(maximum(z₀) - FT(0.5)) <= FT(0.125)
        end
    end
end
