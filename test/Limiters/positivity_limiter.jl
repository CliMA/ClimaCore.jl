# Tests the Zhang–Shu PositivityLimiter on an extruded space, in both state
# shapes: the moist 6-tuple (ρ, ρe, ρu1, ρu2, ρu3, ρq) and the dry 5-tuple
# without the tracer. Pins the properties the limiter exists for: it is a
# no-op on admissible states, it restores the ρ/ρq/p floors, and it preserves
# every WJ-weighted element mean exactly (a redistribution, not a clamp).
using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields, Limiters, Spaces

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

const FT = Float64
const γ = FT(1.4)

# Ideal-gas pressure of a conserved node: p = (γ−1)(ρe − |ρu|²/2ρ − ρ·off),
# with `off` carrying the geopotential-like unscaled part. The tracer is
# passed through unused — `nothing` in the dry 5-tuple case.
zs_pressure(ρ, ρe, u1, u2, u3, ρq, off) =
    (γ - 1) * (ρe - (u1^2 + u2^2 + u3^2) / (2 * ρ) - ρ * off)

# WJ-weighted integrals per element slab, one entry per (v, h) — the
# conserved quantities the limiter must preserve. Uses the parent arrays,
# laid out (Nv, Ni, Nj, Nf, Nh) on an extruded space.
function element_integrals(f)
    space = axes(f)
    wj = Fields.Field(FT, space)
    wj .= Fields.local_geometry_field(space).WJ
    pf = parent(f)
    pw = parent(wj)
    return dropdims(sum(pf .* pw; dims = (2, 3, 4)); dims = (2, 3, 4))
end

function constant_field(space, val)
    f = Fields.Field(FT, space)
    fill!(parent(f), FT(val))
    return f
end

function admissible_state(space)
    ρ = constant_field(space, 1)
    ρe = constant_field(space, 10)
    u1 = constant_field(space, 1 / 10)
    u2 = constant_field(space, -1 / 5)
    u3 = constant_field(space, 3 / 10)
    ρq = constant_field(space, 1 / 1000)
    off = constant_field(space, 2)
    return (ρ, ρe, u1, u2, u3, ρq, off)
end

@testset "PositivityLimiter: no-op on admissible states" begin
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    lim = Limiters.PositivityLimiter(FT; ρ_min = FT(1e-6), p_min = FT(1e-2))
    (ρ, ρe, u1, u2, u3, ρq, off) = admissible_state(space)
    before = deepcopy.((ρ, ρe, u1, u2, u3, ρq))
    Limiters.apply_positivity_limiter!(
        lim,
        zs_pressure,
        (ρ, ρe, u1, u2, u3, ρq),
        off,
    )
    for (f, f0) in zip((ρ, ρe, u1, u2, u3, ρq), before)
        @test parent(f) == parent(f0)
    end
end

@testset "PositivityLimiter: moist 6-tuple restores floors, preserves means" begin
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    ρ_min = FT(1e-6)
    p_min = FT(1e-2)
    lim = Limiters.PositivityLimiter(FT; ρ_min, p_min, maxiter = 30)
    (ρ, ρe, u1, u2, u3, ρq, off) = admissible_state(space)
    # Inadmissible nodes in three distinct elements: a density undershoot, a
    # tracer undershoot, and a kinetic-energy spike driving p < p_min. Element
    # means stay admissible, so a valid θ exists in each.
    parent(ρ)[1, 1, 1, 1, 1] = -FT(1) / 10
    parent(ρq)[2, 1, 1, 1, 2] = -FT(1) / 1000
    parent(u1)[3, 2, 2, 1, 3] = FT(4)
    ints_before = map(element_integrals, (ρ, ρe, u1, u2, u3, ρq))
    Limiters.apply_positivity_limiter!(
        lim,
        zs_pressure,
        (ρ, ρe, u1, u2, u3, ρq),
        off,
    )
    ints_after = map(element_integrals, (ρ, ρe, u1, u2, u3, ρq))
    for (b, a) in zip(ints_before, ints_after)
        @test all(isapprox.(b, a; rtol = 1e-12))
    end
    @test minimum(parent(ρ)) >= ρ_min - eps(FT)
    @test minimum(parent(ρq)) >= -eps(FT)
    p = @. zs_pressure(ρ, ρe, u1, u2, u3, ρq, off)
    # The pressure floor is met up to the bisection tolerance in θ.
    @test minimum(parent(p)) >= p_min - 1e-6
    # The limiter actually engaged (states differ from the perturbed input).
    @test parent(ρ)[1, 1, 1, 1, 1] >= ρ_min
end

@testset "PositivityLimiter: dry 5-tuple" begin
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    ρ_min = FT(1e-6)
    p_min = FT(1e-2)
    lim = Limiters.PositivityLimiter(FT; ρ_min, p_min, maxiter = 30)
    (ρ, ρe, u1, u2, u3, _, off) = admissible_state(space)
    parent(ρ)[1, 1, 1, 1, 1] = -FT(1) / 10
    parent(u1)[3, 2, 2, 1, 3] = FT(4)
    ints_before = map(element_integrals, (ρ, ρe, u1, u2, u3))
    Limiters.apply_positivity_limiter!(
        lim,
        zs_pressure,
        (ρ, ρe, u1, u2, u3),
        off,
    )
    ints_after = map(element_integrals, (ρ, ρe, u1, u2, u3))
    for (b, a) in zip(ints_before, ints_after)
        @test all(isapprox.(b, a; rtol = 1e-12))
    end
    @test minimum(parent(ρ)) >= ρ_min - eps(FT)
    p = @. zs_pressure(ρ, ρe, u1, u2, u3, FT(0), off)
    @test minimum(parent(p)) >= p_min - 1e-6
end
