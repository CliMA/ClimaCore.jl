using Test
using LinearAlgebra
using IntervalSets
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Quadratures,
    Topologies

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

include("utils_dg.jl")  # dg_central_flux

# `Operators.tendency_completion` / `Operators.complete_tendency!`, the
# model-level CG↔DG switch: the completion built for a continuous space must
# reproduce `weighted_dss!` bitwise, and the one built for a discontinuous
# space must reproduce the manual mass-weighted surface-term sequence
# (WJ-weight → interface/boundary flux → unweight) bitwise, so a tendency
# written against the completion is identical to the hand-written form of
# either discretization.

function completion_channel_space(
    ::Type{FT};
    discretization,
    nelem = 4,
    Nq = 4,
) where {FT}
    context = ClimaComms.SingletonCommsContext()
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(0) .. Geometry.XPoint{FT}(2π),
        Geometry.YPoint{FT}(-1) .. Geometry.YPoint{FT}(1);
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, nelem, nelem)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(
        topology,
        Quadratures.GLL{Nq}();
        discretization,
    )
end

# Element-local weak tendency of ∂q/∂t = -∇·(q u) for the state (; q, uv).
function local_weak_tendency(y)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    F = @. Geometry.transform(Geometry.Contravariant12Axis(), y.q * y.uv)
    return @. -wdiv(F)
end

state(space) =
    map(Fields.coordinate_field(space)) do coord
        FT = typeof(coord.x)
        (;
            q = sin(coord.x) + cos(coord.y),
            uv = Geometry.UVVector(cos(coord.x), sin(coord.y) / FT(2)),
        )
    end

# One-sided outflow closure at domain-boundary faces.
outflow_flux(normal, argvals⁻) = dg_central_flux(normal, argvals⁻, argvals⁻)

@testset "tendency completion (CG↔DG switch)" begin
    TU.@test_precisions FT begin
        @testset "CG completion is weighted_dss! [$FT]" begin
            space = completion_channel_space(FT; discretization = Spaces.CG())
            y = state(space)
            dydt = local_weak_tendency(y)
            completion = Operators.tendency_completion(
                dydt;
                numflux = dg_central_flux,
            )
            @test completion isa Operators.DSSCompletion

            r_api = copy(dydt)
            Operators.complete_tendency!(completion, r_api, y)
            r_dss = copy(dydt)
            Spaces.weighted_dss!(r_dss)
            @test parent(r_api) == parent(r_dss)
        end

        @testset "DG completion is the mass-weighted surface term [$FT]" begin
            space = completion_channel_space(FT; discretization = Spaces.DG())
            lgeom = Fields.local_geometry_field(space)
            y = state(space)
            dydt = local_weak_tendency(y)
            completion = Operators.tendency_completion(
                dydt;
                numflux = dg_central_flux,
            )
            @test completion isa Operators.NumericalFluxCompletion

            r_api = copy(dydt)
            Operators.complete_tendency!(completion, r_api, y)

            r_manual = copy(dydt)
            @. r_manual = r_manual * lgeom.WJ
            Operators.add_numerical_flux_interior!(
                dg_central_flux,
                r_manual,
                y,
            )
            @. r_manual = r_manual / lgeom.WJ
            @test parent(r_api) == parent(r_manual)
        end

        @testset "DG completion with a boundary flux [$FT]" begin
            space = completion_channel_space(FT; discretization = Spaces.DG())
            lgeom = Fields.local_geometry_field(space)
            y = state(space)
            dydt = local_weak_tendency(y)
            completion = Operators.tendency_completion(
                dydt;
                numflux = dg_central_flux,
                boundary_numflux = outflow_flux,
            )

            r_api = copy(dydt)
            Operators.complete_tendency!(completion, r_api, y)

            r_manual = copy(dydt)
            @. r_manual = r_manual * lgeom.WJ
            Operators.add_numerical_flux_interior!(
                dg_central_flux,
                r_manual,
                y,
            )
            Operators.add_numerical_flux_boundary!(outflow_flux, r_manual, y)
            @. r_manual = r_manual / lgeom.WJ
            @test parent(r_api) == parent(r_manual)

            # The boundary term contributes: the two completions differ.
            r_interior = copy(dydt)
            Operators.complete_tendency!(
                Operators.tendency_completion(dydt; numflux = dg_central_flux),
                r_interior,
                y,
            )
            @test parent(r_api) != parent(r_interior)
        end

        @testset "DG completion requires a numflux [$FT]" begin
            space = completion_channel_space(FT; discretization = Spaces.DG())
            dydt = state(space).q
            @test_throws ErrorException Operators.tendency_completion(dydt)
        end

        # A `FieldVector` state is what a model holds, so the completion has to
        # accept one wherever the underlying mechanism does.
        @testset "FieldVector tendency [$FT]" begin
            cg_space =
                completion_channel_space(FT; discretization = Spaces.CG())
            dg_space =
                completion_channel_space(FT; discretization = Spaces.DG())
            q_cg = state(cg_space).q
            q_dg = state(dg_space).q

            # On CG the completion is one batched DSS over every component.
            fv = Fields.FieldVector(; a = copy(q_cg), b = copy(q_cg))
            expected_a = copy(q_cg)
            expected_b = copy(q_cg)
            Spaces.weighted_dss!(expected_a)
            Spaces.weighted_dss!(expected_b)
            Operators.complete_tendency!(
                Operators.tendency_completion(fv),
                fv,
            )
            @test parent(fv.a) == parent(expected_a)
            @test parent(fv.b) == parent(expected_b)

            # On DG the interface flux needs the whole state at a face node,
            # so a FieldVector is rejected with an explanatory error rather
            # than a MethodError.
            fv_dg = Fields.FieldVector(; a = copy(q_dg))
            @test_throws ErrorException Operators.tendency_completion(
                fv_dg;
                numflux = dg_central_flux,
            )

            # One completion applies one interface treatment, so components
            # may not disagree on the discretization.
            mixed = Fields.FieldVector(; a = copy(q_cg), b = copy(q_dg))
            @test_throws ErrorException Operators.tendency_completion(mixed)
        end
    end
end
