#=
CPU-only validation of the single-kernel spectral-operator arithmetic in
fused_kernels.jl against ClimaCore's operators. No GPU or cuTile needed:

    julia +1.11 --project=benchmarks/cutile benchmarks/cutile/test_fused_cpu.jl

Three geometries: a shallow sphere without topography and one with linear
terrain-following coordinates (both have level-uniform horizontal metrics,
so the one-level and the per-level metric paths must agree), and a deep
sphere (metrics scale with radius, so only the per-level path applies).
Checks the plain and weighted/accumulated scalar Laplacian, the
specific-enthalpy prologue used by the energy hyperdiffusion, and the fused
pressure gradient.
=#

import ClimaComms
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Grids,
    Hypsography,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies
using Test

include(joinpath(@__DIR__, "fused_kernels.jl"))

const FT = Float64
const Nq = 4
const K = Nq * Nq
const helem = 3
const zelem = 5
const zmax = FT(30e3)

const C12 = Geometry.Covariant12Vector
const grad = Operators.Gradient()
const wdiv = Operators.Divergence{Operators.WeakForm}()

function make_space(; deep = false, topography = false)
    context = ClimaComms.context(ClimaComms.CPUSingleThreaded())
    hdomain = Domains.SphereDomain(FT(6.37122e6))
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    hspace = Spaces.SpectralElementSpace2D(htopology, Quadratures.GLL{Nq}())
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(zmax);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vtopology = Topologies.IntervalTopology(context, vertmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    hyps = if topography
        hcoords = Fields.coordinate_field(hspace)
        z_sfc = @. Geometry.ZPoint(
            FT(2e3) * (1 + sind(2 * hcoords.long) * cosd(3 * hcoords.lat)) / 2,
        )
        Hypsography.LinearAdaption(z_sfc)
    else
        Grids.Flat()
    end
    return Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace, hyps; deep)
end

# (Nv, Nq², Nh) array of a scalar field, or of component `f` of a vector field.
arr3(field, f = 1) = component3(parent(Fields.field_values(field)), f)

function state(space)
    coords = Fields.coordinate_field(space)
    p = @. FT(1e5) * (1 + FT(0.1) * sind(coords.long) * cosd(coords.lat))
    ρ = @. 1 + FT(0.05) * cosd(coords.lat) + coords.z / zmax
    Kin = @. FT(100) * sind(coords.long)^2
    Φ = @. FT(9.81) * coords.z
    χ = @. sind(coords.long) * cosd(coords.lat) * (1 + coords.z / zmax)
    ρe = @. ρ * (FT(2e5) + FT(1e4) * cosd(2 * coords.long) * sind(coords.lat))
    return (; p, ρ, Kin, Φ, χ, ρe)
end

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs, b), floatmin(FT))

function check_geometry(name; levels_choices, deep = false, topography = false)
    space = make_space(; deep, topography)
    (; p, ρ, Kin, Φ, χ, ρe) = state(space)
    D = Quadratures.differentiation_matrix(FT, Quadratures.GLL{Nq}())
    w = spectral_weights(D)
    κ = FT(3e15)

    coords = Fields.coordinate_field(space)
    lap_ref = arr3(@. wdiv(grad(χ)))
    acc0 = @. sind(3 * coords.long) * FT(0.5)
    acc_ref = arr3(@. acc0 - κ * wdiv(ρ * grad(χ)))
    h_tot = @. (ρe + p) / ρ
    energy_ref = arr3(@. wdiv(grad(h_tot)))
    du_ref = @. C12(grad(p) / ρ + grad(Kin + Φ))
    ut0 = @. C12(FT(0.25) * cosd(coords.lat), FT(-0.75))
    ut_ref = @. ut0 - du_ref

    @testset "$name" begin
        for levels in levels_choices
            m = horizontal_metrics(space; levels)
            @test size(m.g11, 1) == levels

            out = similar(lap_ref)
            cpu_scalar_laplacian!(out, arr3(χ), w, m)
            @test relerr(out, lap_ref) < 1e-11

            out = copy(arr3(acc0))
            cpu_scalar_laplacian!(
                out,
                arr3(χ),
                w,
                m;
                ρ = arr3(ρ),
                weighted = true,
                scale = -κ,
                accumulate = true,
            )
            @test relerr(out, acc_ref) < 1e-11

            out = similar(lap_ref)
            cpu_scalar_laplacian!(out, arr3(ρe), w, m; p = arr3(p), ρ = arr3(ρ))
            @test relerr(out, energy_ref) < 1e-11
        end

        out1 = copy(arr3(ut0, 1))
        out2 = copy(arr3(ut0, 2))
        cpu_pressure_gradient!(
            out1,
            out2,
            arr3(p),
            arr3(Kin),
            arr3(Φ),
            arr3(ρ),
            w;
            scale = -one(FT),
            accumulate = true,
        )
        # The gate is absolute: the exact horizontal gradient of the O(1e5)
        # node-constant part of p is zero, so the result is a cancellation
        # whose rounding floor is eps · ‖D‖ · ‖p‖ / ρ.
        floor = 8 * eps(FT) * maximum(sum(abs, Matrix(D); dims = 2)) * FT(1.1e5)
        @test maximum(abs.(out1 .- arr3(ut_ref, 1))) < floor
        @test maximum(abs.(out2 .- arr3(ut_ref, 2))) < floor
    end
    return space
end

@testset "fused spectral kernels (CPU reference)" begin
    space = check_geometry("shallow flat sphere"; levels_choices = (1, zelem))
    @test metrics_are_level_uniform(space)

    space = check_geometry("deep sphere"; levels_choices = (zelem,), deep = true)
    @test !metrics_are_level_uniform(space)

    # Linear adaption rescales ξ³ only: g¹¹, g¹², g²² are unchanged and the
    # Jacobian's horizontal variation is the same on every level.
    space = check_geometry(
        "linear terrain-following";
        levels_choices = (1, zelem),
        topography = true,
    )
    @test metrics_are_level_uniform(space)
end
