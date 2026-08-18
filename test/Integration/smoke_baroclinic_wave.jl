using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Grids,
    Spaces,
    Quadratures,
    Fields,
    Geometry,
    Operators

@testset "3D Hybrid Extruded Hydrostatic / Baroclinic Wave Smoke Test" begin
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext()
        radius = FT(6.371e6)
        zmax = FT(30e3)
        helem = 4
        zelem = 10
        Nq = 4

        # Vertical domain and grid
        vdomain = Domains.IntervalDomain(
            Geometry.ZPoint(zero(FT)),
            Geometry.ZPoint(zmax);
            boundary_names = (:bottom, :top),
        )
        vmesh = Meshes.IntervalMesh(vdomain, nelems = zelem)
        vtopology = Topologies.IntervalTopology(context, vmesh)
        vgrid = Grids.FiniteDifferenceGrid(vtopology)

        # Horizontal cubed-sphere domain and grid
        hdomain = Domains.SphereDomain(radius)
        hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
        htopology = Topologies.Topology2D(context, hmesh)
        quad = Quadratures.GLL{Nq}()
        hgrid = Grids.SpectralElementGrid2D(htopology, quad)

        # 3D Extruded hybrid grid & spaces
        grid3d = Grids.ExtrudedFiniteDifferenceGrid(hgrid, vgrid)
        center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid3d)
        face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid3d)

        coords_c = Fields.coordinate_field(center_space)
        coords_f = Fields.coordinate_field(face_space)

        # Initialize background atmosphere
        p0 = FT(1e5)
        R_d = FT(287.0)
        g = FT(9.81)
        T0 = FT(300.0)

        # Hydrostatic density profile: ρ(z) = ρ₀ * exp(-g z / (R_d T₀))
        ρ0 = p0 / (R_d * T0)
        scale_height = R_d * T0 / g

        ρ_c = @. ρ0 * exp(-coords_c.z / scale_height) *
           (FT(1) + FT(0.01) * cosd(coords_c.lat) * cosd(coords_c.long))
        u_c = @. Geometry.UVWVector(
            FT(20) * cosd(coords_c.lat) * (coords_c.z / zmax),
            zero(FT),
            zero(FT),
        )
        w_f = fill(Geometry.WVector(zero(FT)), face_space)

        # Operators
        hdiv = Operators.Divergence()
        vdiv = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.WVector(zero(FT))),
            top = Operators.SetValue(Geometry.WVector(zero(FT))),
        )
        vinterp_c2f = Operators.InterpolateC2F(
            bottom = Operators.SetValue(zero(FT)),
            top = Operators.SetValue(zero(FT)),
        )

        # Compute initial total mass
        mass0 = sum(ρ_c)
        @test !isnan(mass0)
        @test mass0 > 0

        function continuity_rhs!(dρdt, ρ, u, w)
            flux_h = @. ρ * u
            ρ_f = vinterp_c2f.(ρ)
            flux_v = @. ρ_f * w
            @. dρdt = -(hdiv(flux_h))
            @. dρdt -= vdiv(flux_v)
            Spaces.weighted_dss!(dρdt)
            return dρdt
        end

        dρdt = similar(ρ_c)
        continuity_rhs!(dρdt, ρ_c, u_c, w_f)

        # Check no NaNs/Infs
        @test all(!isnan, parent(dρdt))
        @test all(!isinf, parent(dρdt))

        # 5-step time integration
        dt = FT(1.0)
        ρ = copy(ρ_c)
        for step in 1:5
            continuity_rhs!(dρdt, ρ, u_c, w_f)
            @. ρ += dt * dρdt
        end

        # Verify mass conservation
        mass_final = sum(ρ)
        tol = FT == Float32 ? 1e-4 : 1e-10
        @test isapprox(mass_final, mass0, rtol = tol)
    end
end
