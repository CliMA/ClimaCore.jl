using ClimaCore
using ClimaCore.CommonSpaces
import ClimaComms
import ClimaCore: DataLayouts, Fields
using Test
ClimaComms.@import_required_backends

const ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
@assert !isnothing(ext) # cuda must be loaded to test this extension

# An A100's occupancy-relevant attributes, so the band arithmetic is checked
# against known hardware wherever these tests run.
const A100 = (;
    threads_per_warp = 32,
    max_threads_per_sm = 2048,
    max_regs_per_sm = 65536,
    regs_per_allocation = 256,
    max_regs_per_thread = 256,
    schedulers_per_sm = 4,
)

@testset "resident warps from a register count" begin
    warps(r) = ext.max_warps_per_sm_by_regs(r, A100)
    @test warps(0) == 64 # no registers, so every warp the hardware holds
    @test warps(1) == 64
    @test warps(256) == 8
    @test warps(257) == 0 # past the per-thread limit, nothing can be resident
    # Spending registers never buys warps
    @test all(warps(r + 1) <= warps(r) for r in 0:256)
    # Where an A100's bands end. A cap anywhere else spends registers and buys
    # no warps, so these are the only counts worth asking for.
    @test [r for r in 1:256 if warps(r + 1) < warps(r)] ==
          [32, 40, 48, 56, 64, 72, 80, 96, 128, 168, 256]
end

@testset "the band edge below a register count" begin
    warps(r) = ext.max_warps_per_sm_by_regs(r, A100)
    edge(r) = ext.band_edge_below(r, A100)
    # The counts the policy capped over a coupled AMIP run, and the edge each
    # was capped to: a microphysics cache kernel at 136 registers, two more at
    # 108, and two land surface kernels at 104 and 103.
    @test edge(136) == 128
    @test edge(108) == 96
    @test edge(104) == 96
    @test edge(103) == 96
    # Inside the widest band there is nothing below to ask for
    @test isnothing(edge(1))
    @test isnothing(edge(32))
    # An edge buys warps, and nothing between it and the count does
    for r in 33:256
        e = edge(r)
        @test !isnothing(e)
        @test warps(e) > warps(r)
        @test all(warps(k) == warps(r) for k in (e + 1):r)
    end
end

# The shape the policy fires on in the coupled model: a pointwise broadcast over
# the cell centers of an extruded cubed sphere, several fields in and one out.
function amip_shaped_fields(::Type{FT}) where {FT}
    space = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 4,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    ρ = fill(FT(1.2), space)
    return (; dest = similar(ρ), ρ, θ = fill(FT(290), space), q = fill(FT(0.01), space))
end

# Register pressure in the shape of a microphysics closure: N independent
# partial results, all live at once because they are consumed in reverse, so
# none can be folded away. N is what moves the kernel's register count.
@inline function heavy_closure(ρ, θ, q, ::Val{N}) where {N}
    terms = ntuple(i -> muladd(θ, ρ^i, q / (i + one(q))), Val(N))
    s = zero(θ)
    @inbounds for i in 1:N
        s = muladd(terms[i], sqrt(abs(terms[N - i + 1])), s)
    end
    return s
end

# One function per level rather than a Val argument, so each is its own kernel
# specialization and the policy decides about it separately.
heavy_8(ρ, θ, q) = heavy_closure(ρ, θ, q, Val(8))
heavy_16(ρ, θ, q) = heavy_closure(ρ, θ, q, Val(16))
heavy_24(ρ, θ, q) = heavy_closure(ρ, θ, q, Val(24))
heavy_32(ρ, θ, q) = heavy_closure(ρ, θ, q, Val(32))
const HEAVY = (heavy_8, heavy_16, heavy_24, heavy_32)

# The policy's settings are process-global and read on every launch, so a case
# that changes them restores them, and the decisions, on its way out.
const POLICY_REFS = (;
    on = ext.AUTO_REGISTER_CAP,
    max_shed = ext.AUTO_REGISTER_CAP_MAX_SHED,
    max_warps = ext.AUTO_REGISTER_CAP_MAX_WARPS,
    growth_pct = ext.AUTO_REGISTER_CAP_LOCAL_GROWTH_PCT,
    slack = ext.AUTO_REGISTER_CAP_LOCAL_SLACK,
    exempt = ext.AUTO_REGISTER_CAP_EXEMPT,
    named = ext.NAME_KERNELS_FROM_STACK_TRACE,
)

function with_policy(body; kwargs...)
    saved = map(ref -> ref[], POLICY_REFS)
    for (name, value) in pairs(kwargs)
        POLICY_REFS[name][] = value
    end
    empty!(ext.AUTO_CAP_DECISIONS)
    try
        return body()
    finally
        for (name, value) in pairs(saved)
            POLICY_REFS[name][] = value
        end
        empty!(ext.AUTO_CAP_DECISIONS)
    end
end

# Decide about one broadcast and nothing else: everything is warm after the
# first launch, so clearing the decisions and launching again re-decides the
# one kernel under test.
function decisions_for(launch!)
    launch!()
    empty!(ext.AUTO_CAP_DECISIONS)
    launch!()
    return collect(values(ext.auto_cap_decisions()))
end

# What a decision has to satisfy whatever it decided, so that a cap is never
# asked for on terms the policy does not claim to meet.
function check_decision(d)
    @test d.warps == ext.max_warps_per_sm_by_regs(d.regs)
    @test (d.reason === :capped) == !isnothing(d.cap)
    isnothing(d.cap) && return
    @test d.cap == d.edge
    @test d.capped_regs <= d.cap # the cap was honored
    @test d.capped_warps > d.warps # and bought warps
    @test d.capped_local <= d.local_budget # without spilling past the budget
    @test d.regs - d.edge <= POLICY_REFS.max_shed[]
    @test d.warps <= POLICY_REFS.max_warps[]
end

@testset "a register-heavy broadcast is capped at its band edge" begin
    fields = amip_shaped_fields(Float32)
    # Filters wide open, so every kernel with a band below it is a candidate and
    # the compiler decides the rest. The sweep is over register pressure: which
    # level lands past an edge depends on the toolchain, so the test asks that
    # one of them does, and that each decision is self-consistent either way.
    with_policy(
        on = true,
        max_warps = 64,
        max_shed = 256,
        growth_pct = 1000,
        slack = 4096,
    ) do
        capped = falses(length(HEAVY))
        for (i, heavy) in enumerate(HEAVY)
            ds = decisions_for() do
                @. fields.dest = heavy(fields.ρ, fields.θ, fields.q)
            end
            @test !isempty(ds)
            for d in ds
                check_decision(d)
                capped[i] |= d.reason === :capped
            end
        end
        @test any(capped)
    end
end

@testset "a capped kernel computes what an uncapped one does" begin
    fields = amip_shaped_fields(Float32)
    launch!() = @. fields.dest = heavy_32(fields.ρ, fields.θ, fields.q)
    uncapped = with_policy(on = false) do
        launch!()
        @test isempty(ext.auto_cap_decisions()) # nothing is decided when off
        Array(parent(fields.dest))
    end
    with_policy(
        on = true,
        max_warps = 64,
        max_shed = 256,
        growth_pct = 1000,
        slack = 4096,
    ) do
        launch!()
        # A register cap changes the schedule, not the arithmetic
        @test Array(parent(fields.dest)) == uncapped
    end
end

@testset "the filters decline the kernels they are for" begin
    fields = amip_shaped_fields(Float32)
    # A kernel already holding more warps than the threshold is not starved the
    # way the capped ones are, so it is left alone at its default threshold.
    with_policy(on = true, max_warps = 16) do
        ds = decisions_for() do
            @. fields.dest = fields.ρ + 1
        end
        @test length(ds) == 1
        d = only(ds)
        @test d.reason === :already_resident
        @test isnothing(d.cap)
        @test d.warps > 16
        check_decision(d)
    end
    # An edge further away than the shed is not asked for, however starved the
    # kernel is: a cap that far below costs more spill than it buys warps.
    with_policy(on = true, max_warps = 64, max_shed = 0) do
        for heavy in HEAVY
            ds = decisions_for() do
                @. fields.dest = heavy(fields.ρ, fields.θ, fields.q)
            end
            for d in ds
                @test d.reason in (:edge_too_far, :no_band_below)
                @test isnothing(d.cap)
                check_decision(d)
            end
        end
    end
    # A local frame that may not grow at all rejects every cap that spills
    with_policy(
        on = true,
        max_warps = 64,
        max_shed = 256,
        growth_pct = 0,
        slack = 0,
    ) do
        for heavy in HEAVY
            ds = decisions_for() do
                @. fields.dest = heavy(fields.ρ, fields.θ, fields.q)
            end
            for d in ds
                @test d.local_budget == d.local_bytes
                if d.reason === :capped
                    @test d.capped_local <= d.local_bytes
                end
                check_decision(d)
            end
        end
    end
end

@testset "a kernel can be exempted by name" begin
    fields = amip_shaped_fields(Float32)
    settings = (;
        on = true,
        max_warps = 64,
        max_shed = 256,
        growth_pct = 1000,
        slack = 4096,
        named = true,
    )
    # The exempt list matches on the kernel's name, so it needs the names on
    name = with_policy(; settings...) do
        ds = decisions_for() do
            @. fields.dest = heavy_32(fields.ρ, fields.θ, fields.q)
        end
        d = only(ds)
        @test !isnothing(d.name)
        # A kernel is named after the call site that launched it, which for a
        # broadcast is this file and the line it is written on
        @test occursin("auto_register_cap", d.name)
        d.name
    end
    with_policy(; settings..., exempt = [name]) do
        ds = decisions_for() do
            @. fields.dest = heavy_32(fields.ρ, fields.θ, fields.q)
        end
        d = only(ds)
        @test d.reason === :exempt
        @test isnothing(d.cap)
    end
end

@testset "a loop that names its own cap keeps it" begin
    fields = amip_shaped_fields(Float32)
    (dest, ρ) = map(Fields.field_values, (fields.dest, fields.ρ))
    with_policy(
        on = true,
        max_warps = 64,
        max_shed = 256,
        growth_pct = 1000,
        slack = 4096,
    ) do
        # A count, and :none, are both call sites that know their kernel, so
        # neither is a question for the policy
        for cap in (Val(128), Val(:none))
            empty!(ext.AUTO_CAP_DECISIONS)
            DataLayouts.foreach_point(dest, ρ; maxregs = cap) do dest_point, ρ_point
                @inbounds dest_point[] = ρ_point[] + 1
            end
            @test isempty(ext.auto_cap_decisions())
        end
        @test Array(parent(fields.dest)) == Array(parent(fields.ρ)) .+ 1
    end
end

@testset "a specialization is decided once" begin
    fields = amip_shaped_fields(Float32)
    with_policy(on = true, max_warps = 64, max_shed = 256) do
        launch!() = @. fields.dest = heavy_24(fields.ρ, fields.θ, fields.q)
        launch!()
        decided = ext.auto_cap_decisions()
        @test !isempty(decided)
        launch!()
        # The second launch reuses the decision rather than compiling twice again
        @test ext.auto_cap_decisions() == decided
    end
end
