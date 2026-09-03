using Test
using LinearAlgebra
using StaticArrays
using IntervalSets
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry, Quadratures
import ClimaCore.Geometry: ⊗

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

include("utils_dg.jl") # dg_sphere_space, dg_divergence

# Christoffel-free Cartesian tensor divergence ∇·T on a DG space
# (Operators.cartesian_tensor_divergence). The weak Divergence drops the
# connection term Γⁱ_jk Tʲᵏ ; rotating that axis into the global Cartesian basis makes it exact.
# Verified three ways: (1) an exact algebraic identity that holds to roundoff at
# any resolution, (2) the connection term being O(1) when omitted, and (3)
# h-refinement convergence to an analytic divergence.

# Field of a constant global-Cartesian vector expressed in the local frame. The
# closure captures the (typed) constant so the bare Tensor does not drag
# StaticArrays' broadcast style into the Field broadcast.
function local_cartesian_field(space, vcart::Geometry.Cartesian123Vector)
    gg = Spaces.global_geometry(space)
    coords = Fields.coordinate_field(space)
    rgg = Ref(gg)
    f(geom, coord) = Geometry.LocalVector(vcart, geom, coord)
    return f.(rgg, coords)
end

# Naive un-rotated tensor divergence: the same weak volume term + central
# interface flux as cartesian_tensor_divergence, but WITHOUT the momentum-axis
# rotation. On a curved space this omits the connection term.
function naive_tensor_divergence(T)
    space = axes(T)
    lgeom = Fields.local_geometry_field(space)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    r = @. wdiv(T) * (-(lgeom.WJ))
    Operators.add_numerical_flux_interior!(
        Operators.CentralNumericalFlux(identity), r, T,
    )
    return @. -r / lgeom.WJ
end

# The exact identity holds at any precision, so it runs in Float32 and Float64
# (with a precision-aware tolerance).
@testset "Cartesian tensor divergence: exact v⊗m identity (sphere) [$FT]" for FT in
                                                                              (
    Float32,
    Float64,
)
    central = Operators.CentralNumericalFlux(identity)
    # Algebraic identity ⇒ agreement to a few ULP, not the analytic error.
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

        dT = Operators.cartesian_tensor_divergence(T, central)

        dv = dg_divergence(v)
        ref = @. dv * mloc
        maxref = maximum(abs, parent(ref))
        @test maximum(abs, parent(@. dT - ref)) / maxref < rtol
    end
end

@testset "Cartesian tensor divergence: magnitude of missing christoffel terms (sphere) [$FT]" for FT in
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

    dT = Operators.cartesian_tensor_divergence(
        T, Operators.CentralNumericalFlux(identity),
    )
    dT_naive = naive_tensor_divergence(T)
    maxref = maximum(abs, parent(dT))
    # Dropping the connection term is an O(1) error, not roundoff.
    @test maximum(abs, parent(@. dT_naive - dT)) / maxref > 1e-2
end

@testset "Cartesian tensor divergence: in-place is allocation-free (sphere)" begin
    FT = Float64
    space = dg_sphere_space(FT; helem = 6, Nq = 4)
    coords = Fields.coordinate_field(space)
    mcart = Geometry.Cartesian123Vector(FT(0.3), FT(-0.7), FT(0.5))
    mloc = local_cartesian_field(space, mcart)
    v = @. Geometry.UVVector(
        sind(coords.long) * cosd(coords.lat),
        cosd(coords.long),
    )
    T = @. v ⊗ mloc
    central = Operators.CentralNumericalFlux(identity)

    # caller-owned buffers
    Tc = similar(T)
    out = Fields.Field(Geometry.UVWVector{FT}, space)
    # function barrier: type the captured buffers so @allocated sees the true
    # (heap-boxing-free) cost rather than boxing from @testset soft scope.
    function alloc_bytes(out, Tc, T, numflux)
        Operators.cartesian_tensor_divergence!(out, Tc, T, numflux)  # warm up
        return @allocated Operators.cartesian_tensor_divergence!(out, Tc, T, numflux)
    end
    @test alloc_bytes(out, Tc, T, central) == 0
end

@testset "Cartesian tensor divergence: h-convergence to analytic (sphere)" begin
    FT = Float64
    R = FT(6.371e6)
    central = Operators.CentralNumericalFlux(identity)
    helems = (3, 6, 12)
    errs = zeros(FT, length(helems))
    for (i, helem) in enumerate(helems)
        space = dg_sphere_space(FT; radius = R, helem, Nq = 4)
        coords = Fields.coordinate_field(space)

        mcart = Geometry.Cartesian123Vector(FT(0.3), FT(-0.7), FT(0.5))
        mloc = local_cartesian_field(space, mcart)

        # v = (0, cosφ): purely meridional, analytic surface divergence
        # ∇·v = -2 sinφ / R, so ∇·(v⊗m) = (-2 sinφ / R) m.
        v = @. Geometry.UVVector(FT(0), cosd(coords.lat))
        T = @. v ⊗ mloc
        dT = Operators.cartesian_tensor_divergence(T, central)

        div_v_exact = @. -2 * sind(coords.lat) / R
        dT_exact = @. div_v_exact * mloc
        errs[i] = maximum(abs, parent(@. dT - dT_exact))
    end
    Δh = [FT(1) / helem for helem in helems]
    rates = TU.convergence_rate(errs, Δh)
    @info "Cartesian tensor divergence convergence (sphere)" errs rates
    @test all(rates .>= 2.5)
    @test errs[end] < errs[1] / 50
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
        dT = Operators.cartesian_tensor_divergence(T, central)

        # On a plane the rotation is the identity: cartesian == naive exactly.
        @test parent(dT) ≈ parent(naive_tensor_divergence(T))

        div_v_exact = @. cos(coords.x) + cos(coords.y)
        dT_exact = @. div_v_exact * mloc
        errs[i] = maximum(abs, parent(@. dT - dT_exact))
    end
    @test errs[2] < errs[1] / 4
end
