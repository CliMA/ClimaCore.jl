using Test
using IntervalSets
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Quadratures, Spaces, Topologies
import ClimaCore.Geometry: ⊗

include("utils_dg.jl")               # dg_sphere_space
include("utils_tensor_divergence.jl")

# `Operators.cartesian_tensor_divergence` against the identity
# ∇ₕ·(v⊗m) = (∇ₕ·v) m, which holds for a transport field `v` of any form once
# `m` is constant in the global Cartesian frame. It is algebraic, so it holds
# to roundoff at any resolution — including on the terrain-following extruded
# sphere, where the rotation meets `LatLongZPoint` coordinates and a warped
# `∂ξ∂x`.

@testset "Cartesian tensor divergence: exact v⊗m identity (sphere) [$FT]" for FT in
                                                                              (
    Float32,
    Float64,
)
    central = Operators.CentralNumericalFlux(identity)
    # Algebraic identity, so the tolerance sits at a few ULP.
    rtol = FT == Float32 ? FT(1e-4) : FT(1e-11)
    for helem in (3, 6)
        space = dg_sphere_space(FT; helem, Nq = 4)
        @test !Spaces.is_continuous(space)
        coords = Fields.coordinate_field(space)

        # constant global-Cartesian momentum, expressed in the local frame
        mcart = Geometry.Cartesian123Vector(FT(0.3), FT(-0.7), FT(0.5))
        mloc = local_cartesian_field(space, mcart)

        # arbitrary smooth tangent transport velocity
        v = @. Geometry.UVVector(
            sind(coords.long) * cosd(coords.lat),
            cosd(coords.long),
        )
        T = @. v ⊗ mloc  # (UVAxis transport, UVWAxis momentum)

        completion = tensor_div_completion(space; numflux = central)
        dT = Operators.cartesian_tensor_divergence(T, completion)

        dv = completed_divergence(v)
        ref = @. dv * mloc
        maxref = maximum(abs, parent(ref))
        @test maximum(abs, parent(@. dT - ref)) / maxref < rtol
    end
end

@testset "Cartesian tensor divergence: exact v⊗m identity (topography) [$FT]" for FT in
                                                                                  (
    Float32,
    Float64,
)
    central = Operators.CentralNumericalFlux(identity)
    rtol = FT == Float32 ? FT(1e-4) : FT(1e-11)
    for discretization in (Spaces.CG(), Spaces.DG())
        space = tensor_div_topography_space(FT; helem = 3, nz = 6, discretization)
        coords = Fields.coordinate_field(space)

        mcart = Geometry.Cartesian123Vector(FT(0.3), FT(-0.7), FT(0.5))
        mloc = local_cartesian_field(space, mcart)
        v = @. Geometry.UVVector(
            sind(coords.long) * cosd(coords.lat),
            cosd(coords.long),
        )
        T = @. v ⊗ mloc

        completion = tensor_div_completion(space; numflux = central)
        dT = Operators.cartesian_tensor_divergence(T, completion)

        dv = completed_divergence(v)
        ref = @. dv * mloc
        maxref = maximum(abs, parent(ref))
        @test maximum(abs, parent(@. dT - ref)) / maxref < rtol
    end
end

@testset "Cartesian tensor divergence: dropped connection term (sphere) [$FT]" for FT in
                                                                                   (
    Float32,
    Float64,
)
    space = dg_sphere_space(FT; helem = 6, Nq = 4)
    coords = Fields.coordinate_field(space)
    mcart = Geometry.Cartesian123Vector(FT(0.3), FT(-0.7), FT(0.5))
    mloc = local_cartesian_field(space, mcart)
    v = @. Geometry.UVVector(
        sind(coords.long) * cosd(coords.lat),
        cosd(coords.long),
    )
    T = @. v ⊗ mloc

    completion = tensor_div_completion(
        space;
        numflux = Operators.CentralNumericalFlux(identity),
    )
    dT = Operators.cartesian_tensor_divergence(T, completion)
    dT_naive = naive_tensor_divergence(T)
    maxref = maximum(abs, parent(dT))
    # The connection term contributes at O(1), far above roundoff.
    @test maximum(abs, parent(@. dT_naive - dT)) / maxref > 1e-2
end

# A `UVAxis` momentum axis is read as a `UVWAxis` one with `w == 0`, so a model
# carrying horizontal momentum needs no promotion before forming the flux.
@testset "Cartesian tensor divergence: two-component momentum axis" begin
    FT = Float64
    central = Operators.CentralNumericalFlux(identity)
    for discretization in (Spaces.CG(), Spaces.DG())
        space = tensor_div_sphere_space(FT; helem = 3, discretization)
        coords = Fields.coordinate_field(space)
        v = @. Geometry.UVVector(
            sind(coords.long) * cosd(coords.lat),
            cosd(coords.long),
        )
        m2 = @. Geometry.UVVector(cosd(coords.lat), sind(coords.long))
        m3 = @. Geometry.UVWVector(cosd(coords.lat), sind(coords.long), FT(0))
        completion = tensor_div_completion(space; numflux = central)
        T2 = @. v ⊗ m2
        d2 = Operators.cartesian_tensor_divergence(T2, completion)
        d3 = Operators.cartesian_tensor_divergence((@. v ⊗ m3), completion)
        # `T*G'` contracts three components with the padded `w` and two
        # without, so the two agree to rounding: 3.5 ULP relative measured at
        # Ne = 3, ~280x inside this bound.
        @test maximum(abs, parent(d2) .- parent(d3)) <
              1000 * eps(FT) * maximum(abs, parent(d3))

        # The rotation widens the momentum axis, so `similar(T2)` is too
        # narrow: assigning into it drops the Cartesian `w` column.
        out = Fields.Field(Geometry.UVWVector{FT}, space)
        @test_throws ErrorException Operators.cartesian_tensor_divergence!(
            out,
            similar(T2),
            T2,
            completion,
        )
    end
end

@testset "Cartesian tensor divergence: planar identity path" begin
    FT = Float64
    central = Operators.CentralNumericalFlux(identity)
    # Doubly-periodic plane: global_geometry is CartesianGlobalGeometry, so the
    # rotations are the identity and the naive form is already exact.
    L = FT(2π)
    errs = zeros(FT, 2)
    nelems = (4, 8)
    for (i, nelem) in enumerate(nelems)
        context = ClimaComms.SingletonCommsContext()
        domain = Domains.RectangleDomain(
            Geometry.XPoint{FT}(zero(L)) .. Geometry.XPoint{FT}(L),
            Geometry.YPoint{FT}(zero(L)) .. Geometry.YPoint{FT}(L);
            x1periodic = true,
            x2periodic = true,
        )
        mesh = Meshes.RectilinearMesh(domain, nelem, nelem)
        topology = Topologies.Topology2D(context, mesh)
        space = Spaces.SpectralElementSpace2D(
            topology,
            Quadratures.GLL{4}();
            discretization = Spaces.DG(),
        )
        coords = Fields.coordinate_field(space)

        # constant momentum (identity rotation on a plane) with a w-component
        mcart = Geometry.Cartesian123Vector(FT(0.3), FT(-0.7), FT(0.5))
        mloc = local_cartesian_field(space, mcart)

        v = @. Geometry.UVVector(sin(coords.x), sin(coords.y))
        T = @. v ⊗ mloc
        completion = tensor_div_completion(space; numflux = central)
        dT = Operators.cartesian_tensor_divergence(T, completion)

        # On a plane the rotation is the identity, so the naive form and this
        # one run the same arithmetic and agree bit for bit.
        @test parent(dT) == parent(naive_tensor_divergence(T))

        div_v_exact = @. cos(coords.x) + cos(coords.y)
        dT_exact = @. div_v_exact * mloc
        errs[i] = maximum(abs, parent(@. dT - dT_exact))
    end
    @test errs[2] < errs[1] / 4
end
