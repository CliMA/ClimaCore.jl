# Tests the Zhang–Shu PositivityLimiter on an extruded space, in all state
# shapes: moist Euler (; ρ, ρe, ρu1, ρu2, ρu3, ρq) with a scalar or
# NamedTuple-valued tracer field (several variables in one field), dry Euler without the
# tracer, and a shallow-water-like (; h, hu1, hu2) system with no `g`.
# Pins the properties the limiter exists for: it is a
# no-op on admissible states, it restores the ρ/ρq/p floors, and it preserves
# every WJ-weighted element mean exactly (a redistribution, not a clamp).
using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields, Limiters, Spaces
import ForwardDiff

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

const FTs = (Float64, Float32)
# The spaces below are built on this device; on GPU the tests run the CUDA
# kernel, and single-node reads and writes go through `allowscalar`.
const device = ClimaComms.device()

# Ideal-gas pressure of a conserved node: p = (γ−1)(ρe − |ρu|²/2ρ − ρ·aux),
# with `aux` carrying the geopotential-like unscaled part and γ = 1.4 in the
# state's float type. `zs_pressure` is the limiter's `g(U, aux)`; it never
# reads a tracer, so one function serves the moist and dry states.
ideal_gas_p(ρ, ρe, u1, u2, u3, aux) =
    (oftype(ρ, 1.4) - 1) * (ρe - (u1^2 + u2^2 + u3^2) / (2 * ρ) - ρ * aux)
zs_pressure(U, aux) = ideal_gas_p(U.ρ, U.ρe, U.ρu1, U.ρu2, U.ρu3, aux)

# Mean preservation is exact up to roundoff in the slab sums, so the
# tolerance scales with the float type.
ints_rtol(::Type{FT}) where {FT} = 1000 * eps(FT)

# WJ-weighted integrals per element slab, one entry per (v, f, h) — the
# conserved quantities the limiter must preserve, kept per field component so
# a NamedTuple-valued tracer is checked per variable. Uses the parent arrays,
# laid out (Nv, Ni, Nj, Nf, Nh) on an extruded space.
function element_integrals(f)
    space = axes(f)
    FT = Spaces.undertype(space)
    wj = Fields.Field(FT, space)
    wj .= Fields.local_geometry_field(space).WJ
    pf = parent(f)
    pw = parent(wj)
    return dropdims(sum(pf .* pw; dims = (2, 3)); dims = (2, 3))
end

function constant_field(space, val)
    FT = Spaces.undertype(space)
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
    aux = constant_field(space, 2)
    return (ρ, ρe, u1, u2, u3, ρq, aux)
end

@testset "PositivityLimiter: no-op on admissible states ($FT)" for FT in FTs
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    lim = Limiters.PositivityLimiter(
        FT;
        g_min = FT(1e-2),
        floors = (; ρ = FT(1e-6), ρq = 0),
        g = zs_pressure,
    )
    (ρ, ρe, u1, u2, u3, ρq, aux) = admissible_state(space)
    before = deepcopy.((ρ, ρe, u1, u2, u3, ρq))
    Limiters.apply_limiter!((; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3, ρq), aux, lim)
    for (f, f0) in zip((ρ, ρe, u1, u2, u3, ρq), before)
        @test parent(f) == parent(f0)
    end
end

@testset "PositivityLimiter: moist Euler floors and means ($FT)" for FT in FTs
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    ρ_min = FT(1e-6)
    p_min = FT(1e-2)
    lim = Limiters.PositivityLimiter(
        FT;
        g_min = p_min,
        maxiter = 30,
        floors = (; ρ = ρ_min, ρq = 0),
        g = zs_pressure,
    )
    (ρ, ρe, u1, u2, u3, ρq, aux) = admissible_state(space)
    # Inadmissible nodes in three distinct elements: a density undershoot, a
    # tracer undershoot, and a kinetic-energy spike driving p < p_min. Element
    # means stay admissible, so a valid θ exists in each.
    ClimaComms.allowscalar(device) do
        parent(ρ)[1, 1, 1, 1, 1] = -FT(1) / 10
        parent(ρq)[2, 1, 1, 1, 2] = -FT(1) / 1000
        parent(u1)[3, 2, 2, 1, 3] = FT(4)
    end
    fields = (ρ, ρe, u1, u2, u3, ρq)
    # (v, h) of each perturbed element => index in `fields` of the field
    # perturbed there.
    perturbed = ((1, 1) => 1, (2, 2) => 6, (3, 3) => 3)
    ints_before = map(element_integrals, fields)
    before = map(f -> Array(parent(f)), fields)
    Limiters.apply_limiter!((; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3, ρq), aux, lim)
    ints_after = map(element_integrals, fields)
    for (b, a) in zip(ints_before, ints_after)
        @test all(isapprox.(b, a; rtol = ints_rtol(FT)))
    end
    @test minimum(parent(ρ)) >= ρ_min - eps(FT)
    @test minimum(parent(ρq)) >= -eps(FT)
    p = @. ideal_gas_p(ρ, ρe, u1, u2, u3, aux)
    # Bisection keeps θ on the admissible side: the floor holds to roundoff.
    @test minimum(parent(p)) >= p_min - eps(FT)
    # The limiter engaged in each perturbed element, and nowhere else.
    after = map(f -> Array(parent(f)), fields)
    elem(x, (v, h)) = x[v, :, :, :, h]
    for (vh, k) in perturbed
        @test elem(after[k], vh) != elem(before[k], vh)
    end
    (Nv, _, _, _, Nh) = size(before[1])
    others = [(v, h) for v in 1:Nv, h in 1:Nh if (v, h) ∉ first.(perturbed)]
    @test all(others) do vh
        all(k -> elem(after[k], vh) == elem(before[k], vh), eachindex(fields))
    end
end

@testset "PositivityLimiter: NamedTuple multi-tracer field ($FT)" for FT in FTs
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    ρ_min = FT(1e-6)
    p_min = FT(1e-2)
    lim = Limiters.PositivityLimiter(
        FT;
        g_min = p_min,
        maxiter = 30,
        floors = (; ρ = ρ_min, ρq = 0),
        g = zs_pressure,
    )
    (ρ, ρe, u1, u2, u3, _, aux) = admissible_state(space)
    # Nonequilibrium-style tracer set: one field, NamedTuple element type.
    ρq = Fields.Field(NamedTuple{(:tot, :liq, :rai), NTuple{3, FT}}, space)
    fill!(parent(ρq), FT(1e-3))
    # Undershoots in two different variables/elements (4th parent axis is the
    # variable index), plus a density undershoot in a third element.
    ClimaComms.allowscalar(device) do
        parent(ρq)[2, 1, 1, 2, 2] = -FT(1e-3)  # liq
        parent(ρq)[1, 2, 1, 3, 3] = -FT(1) / 2000  # rai
        parent(ρ)[1, 1, 1, 1, 1] = -FT(1) / 10
    end
    ints_before = map(element_integrals, (ρ, ρe, u1, u2, u3, ρq))
    Limiters.apply_limiter!((; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3, ρq), aux, lim)
    ints_after = map(element_integrals, (ρ, ρe, u1, u2, u3, ρq))
    for (b, a) in zip(ints_before, ints_after)
        @test all(isapprox.(b, a; rtol = ints_rtol(FT)))
    end
    @test minimum(parent(ρ)) >= ρ_min - eps(FT)
    @test minimum(parent(ρq)) >= -eps(FT)  # every variable nonnegative
end

@testset "PositivityLimiter: shallow-water-like 3-field ($FT)" for FT in FTs
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    h_min = FT(1e-4)
    lim = Limiters.PositivityLimiter(FT; maxiter = 30, floors = (; h = h_min))
    h = constant_field(space, 2)
    hu1 = constant_field(space, 1 / 2)
    hu2 = constant_field(space, -1 / 3)
    ClimaComms.allowscalar(device) do
        parent(h)[1, 2, 2, 1, 1] = -FT(1) / 2
    end
    ints_before = map(element_integrals, (h, hu1, hu2))
    Limiters.apply_limiter!((; h, hu1, hu2), nothing, lim)
    ints_after = map(element_integrals, (h, hu1, hu2))
    for (b, a) in zip(ints_before, ints_after)
        @test all(isapprox.(b, a; rtol = ints_rtol(FT)))
    end
    @test minimum(parent(h)) >= h_min - eps(FT)
end

@testset "PositivityLimiter: dry Euler ($FT)" for FT in FTs
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    ρ_min = FT(1e-6)
    p_min = FT(1e-2)
    lim = Limiters.PositivityLimiter(
        FT;
        g_min = p_min,
        maxiter = 30,
        floors = (; ρ = ρ_min),
        g = zs_pressure,
    )
    (ρ, ρe, u1, u2, u3, _, aux) = admissible_state(space)
    ClimaComms.allowscalar(device) do
        parent(ρ)[1, 1, 1, 1, 1] = -FT(1) / 10
        parent(u1)[3, 2, 2, 1, 3] = FT(4)
    end
    ints_before = map(element_integrals, (ρ, ρe, u1, u2, u3))
    Limiters.apply_limiter!((; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3), aux, lim)
    ints_after = map(element_integrals, (ρ, ρe, u1, u2, u3))
    for (b, a) in zip(ints_before, ints_after)
        @test all(isapprox.(b, a; rtol = ints_rtol(FT)))
    end
    @test minimum(parent(ρ)) >= ρ_min - eps(FT)
    p = @. ideal_gas_p(ρ, ρe, u1, u2, u3, aux)
    @test minimum(parent(p)) >= p_min - eps(FT)
end

# A floor must name a state: a typo must not silently drop the constraint.
@testset "PositivityLimiter: unknown floor key throws" begin
    space = TU.CenterExtrudedFiniteDifferenceSpace(Float64)
    lim = Limiters.PositivityLimiter(Float64; floors = (; ρ = 0, ρq = 0))
    (ρ, ρe, u1, u2, u3, _, aux) = admissible_state(space)
    @test_throws ArgumentError Limiters.apply_limiter!(
        (; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3),
        aux,
        lim,
    )
end

# `g` reads the state by name, so the order of the state keys is irrelevant.
@testset "PositivityLimiter: state key order does not matter" begin
    FT = Float64
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    lim = Limiters.PositivityLimiter(
        FT;
        g_min = FT(1e-2),
        maxiter = 30,
        floors = (; ρ = FT(1e-6)),
        g = zs_pressure,
    )
    results = map((false, true)) do reversed
        (ρ, ρe, u1, u2, u3, _, aux) = admissible_state(space)
        ClimaComms.allowscalar(device) do
            parent(ρ)[1, 1, 1, 1, 1] = -FT(1) / 10
            parent(u1)[3, 2, 2, 1, 3] = FT(4)
        end
        states = (; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3)
        states = reversed ? NamedTuple{reverse(keys(states))}(states) : states
        Limiters.apply_limiter!(states, aux, lim)
        map(parent, (ρ, ρe, u1, u2, u3))
    end
    @test results[1] == results[2]
end

@testset "PositivityLimiter: float type mismatch throws" begin
    space = TU.CenterExtrudedFiniteDifferenceSpace(Float64)
    lim = Limiters.PositivityLimiter(Float32; floors = (; ρ = 0, ρq = 0), g = zs_pressure)
    (ρ, ρe, u1, u2, u3, ρq, aux) = admissible_state(space)
    @test_throws ArgumentError Limiters.apply_limiter!(
        (; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3, ρq),
        aux,
        lim,
    )
end

# Paper step 1 (Zhang–Shu (2.4)–(2.5)) scales only the field whose floor is
# violated: a tracer undershoot must not modify the dynamics, and a depth
# undershoot must not modify the momentum.
@testset "PositivityLimiter: linear step touches only constrained fields ($FT)" for FT in
                                                                                    FTs

    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    lim = Limiters.PositivityLimiter(
        FT;
        g_min = FT(1e-2),
        maxiter = 30,
        floors = (; ρ = FT(1e-6), ρq = 0),
        g = zs_pressure,
    )
    (ρ, ρe, u1, u2, u3, ρq, aux) = admissible_state(space)
    ClimaComms.allowscalar(device) do
        parent(ρq)[2, 1, 1, 1, 2] = -FT(1) / 1000
    end
    before = deepcopy.((ρ, ρe, u1, u2, u3))
    Limiters.apply_limiter!((; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3, ρq), aux, lim)
    @test minimum(parent(ρq)) >= -eps(FT)
    for (f, f0) in zip((ρ, ρe, u1, u2, u3), before)
        @test parent(f) == parent(f0)
    end

    # Shallow water: depth undershoot, momentum untouched.
    h = constant_field(space, 2)
    hu1 = constant_field(space, 1 / 2)
    hu2 = constant_field(space, -1 / 3)
    ClimaComms.allowscalar(device) do
        parent(h)[1, 2, 2, 1, 1] = -FT(1) / 2
    end
    before = deepcopy.((hu1, hu2))
    Limiters.apply_limiter!(
        (; h, hu1, hu2),
        nothing,
        Limiters.PositivityLimiter(FT; floors = (; h = FT(1e-4))),
    )
    @test minimum(parent(h)) >= FT(1e-4) - eps(FT)
    @test parent(hu1) == parent(before[1])
    @test parent(hu2) == parent(before[2])
end

# Step 1 is per component of a NamedTuple-valued field: an undershoot in one
# variable leaves the other variables of the same element unchanged.
@testset "PositivityLimiter: per-variable linear step ($FT)" for FT in FTs
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    lim = Limiters.PositivityLimiter(FT; floors = (; ρq = 0))
    ρq = Fields.Field(NamedTuple{(:tot, :liq), NTuple{2, FT}}, space)
    fill!(parent(ρq), FT(1e-3))
    ClimaComms.allowscalar(device) do
        parent(ρq)[2, 1, 1, 2, 2] = -FT(1e-3)  # liq only
        parent(ρq)[2, 2, 2, 1, 2] = FT(2e-3)   # tot varies but stays admissible
    end
    tot_before = copy(parent(ρq)[:, :, :, 1, :])
    Limiters.apply_limiter!((; ρq), nothing, lim)
    @test minimum(parent(ρq)) >= -eps(FT)
    @test parent(ρq)[:, :, :, 1, :] == tot_before
end

# θ₁ stays in [0, 1]: an inadmissible mean or a constant element below the
# floor collapses to the mean rather than extrapolating (or dividing by 0).
@testset "PositivityLimiter: θ₁ clamped to [0, 1] ($FT)" for FT in FTs
    θ = Limiters._θ_floor
    @test θ(FT(1), FT(2), FT(0)) == 1                 # admissible
    @test θ(FT(1), FT(-1), FT(0)) == FT(1 / 2)        # interior
    @test θ(FT(-1), FT(-2), FT(0)) == 0               # mean below floor
    @test θ(FT(-1), FT(-1), FT(0)) == 0               # constant, below floor
    @test θ(FT(-1), FT(-1 + eps(FT)), FT(0)) == 0     # roundoff: m < xmin
end

# Step 2 (the paper's (2.8)–(2.12)) runs on the state after step 1: a node
# whose density undershoots and whose pressure is low is admissible after
# limiting, with θ₂ computed from the density-corrected state.
@testset "PositivityLimiter: pressure step on intermediate state ($FT)" for FT in FTs
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    ρ_min = FT(1e-6)
    p_min = FT(1e-2)
    lim = Limiters.PositivityLimiter(
        FT;
        g_min = p_min,
        maxiter = 30,
        floors = (; ρ = ρ_min),
        g = zs_pressure,
    )
    (ρ, ρe, u1, u2, u3, _, aux) = admissible_state(space)
    # Same node: density undershoot and a momentum spike (p < p_min there).
    ClimaComms.allowscalar(device) do
        parent(ρ)[1, 1, 1, 1, 1] = -FT(1) / 10
        parent(u1)[1, 1, 1, 1, 1] = FT(4)
    end
    ints_before = map(element_integrals, (ρ, ρe, u1, u2, u3))
    Limiters.apply_limiter!((; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3), aux, lim)
    ints_after = map(element_integrals, (ρ, ρe, u1, u2, u3))
    for (b, a) in zip(ints_before, ints_after)
        @test all(isapprox.(b, a; rtol = ints_rtol(FT)))
    end
    @test minimum(parent(ρ)) >= ρ_min - eps(FT)
    p = @. ideal_gas_p(ρ, ρe, u1, u2, u3, aux)
    @test minimum(parent(p)) >= p_min - eps(FT)
end

# ForwardDiff derivatives match central differences when the pressure step is
# active (at node `a`) and at another node `b` of the same slab. Float64 only:
# central differences are too noisy in Float32.
@testset "PositivityLimiter: ForwardDiff derivatives (Float64)" begin
    FT = Float64
    space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
    # (v, i, j, f, h): same slab v = 2, h = 1.
    a, b = CartesianIndex(2, 2, 1, 1, 1), CartesianIndex(2, 3, 1, 1, 1)
    lim = Limiters.PositivityLimiter(
        FT;
        floors = (; ρ = FT(1e-6)),
        g = zs_pressure,
        g_min = FT(1e-2),
        maxiter = 50,
    )
    function limited_ρu1(ε, node)
        (ρ, ρe, u1, u2, u3) = map(
            val -> fill(FT(val) + zero(ε), space),
            (1, 10, 1 / 10, -1 / 5, 3 / 10),
        )
        aux = fill(FT(2), space)
        ClimaComms.allowscalar(device) do
            Fields.field_values(u1)[a] = 4 + ε  # p < p_min
            Fields.field_values(u1)[b] = FT(1) / 2
            Fields.field_values(ρe)[b] = 10 + 3ε
        end
        Limiters.apply_limiter!((; ρ, ρe, ρu1 = u1, ρu2 = u2, ρu3 = u3), aux, lim)
        return ClimaComms.allowscalar(() -> Fields.field_values(u1)[node], device)
    end
    δ = FT(1e-4)
    for node in (a, b)
        ad = ForwardDiff.derivative(ε -> limited_ρu1(ε, node), FT(0))
        fd = (limited_ρu1(δ, node) - limited_ρu1(-δ, node)) / 2δ
        @test ad ≈ fd rtol = 1e-6 atol = 1e-8
    end
end
