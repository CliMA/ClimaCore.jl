using Test
import ClimaComms
import ClimaCore: DataLayouts
ClimaComms.@import_required_backends

# Regression tests for the broadcast argument traversal functions layout_args,
# equal_layout_shapes, and f_dim, which are called from GPU kernels. Their
# implementations must not trigger inference's recursion-widening heuristics,
# since widened code requires dynamic dispatch and heap allocation, neither of
# which can be compiled for GPUs. The heuristics are only triggered during GPU
# compilation of wide or complexly typed broadcast expressions, like the fused
# parameterization broadcasts in ClimaAtmos (CPU inference checks like JET do 
# not detect them, even for the argument tuples that break GPU compilation), so 
# the fused broadcasts below are materialized at the original failure's scale
# and this file is run in a 1-GPU Buildkite job. The implementation pattern
# itself is guarded by the "Kernel-reachable unrolled functions" test in
# test/aqua.jl.

device = ClimaComms.device()
FT = Float64
CT = Complex{FT}
A = ClimaComms.array_type(device){FT}
(Nv, Nij, Nh) = (4, 3, 5)

new_data(::Type{T} = FT) where {T} =
    DataLayouts.VIJFH{T, Nv, Nij, Nij, nothing}(A, Nh)
new_point_data(::Type{T} = FT) where {T} = DataLayouts.DataF{T}(A)

@testset "broadcast arg traversal" begin
    wide_args = (
        ntuple(_ -> new_data(), 8)...,
        new_point_data(),
        FT(1),
        Base.broadcasted(*, new_point_data(), new_data()),
        ntuple(_ -> new_data(), 8)...,
    )
    wide_bc = Base.broadcasted(+, wide_args...)
    nested_bc = Base.broadcasted(
        +,
        new_data(),
        Base.broadcasted(
            *,
            FT(2),
            Base.broadcasted(-, new_data(), new_point_data()),
        ),
    )
    fused_bc = DataLayouts.FusedMultiBroadcast((
        Pair(new_data(), wide_bc),
        Pair(new_data(), nested_bc),
        Pair(new_point_data(), Base.broadcasted(+, new_point_data(), FT(1))),
    ))

    @testset "layout_args" begin
        @test length(DataLayouts.layout_args(wide_bc)) == 18
        @test length(DataLayouts.layout_args(nested_bc)) == 2
        @test length(DataLayouts.layout_args(fused_bc)) == 6
    end

    @testset "equal_layout_shapes" begin
        same_shape_args =
            (ntuple(_ -> new_data(), 16)..., new_point_data(), FT(1))
        mixed_shape_args = (
            same_shape_args...,
            DataLayouts.VIJFH{FT, Nv + 1, Nij, Nij, nothing}(A, Nh),
        )
        point_args = (new_point_data(), FT(1))
        @test DataLayouts.equal_layout_shapes(same_shape_args)
        @test !DataLayouts.equal_layout_shapes(mixed_shape_args)
        @test DataLayouts.equal_layout_shapes(point_args)
        @test DataLayouts.equal_layout_shapes(())
    end

    @testset "f_dim" begin
        @test DataLayouts.f_dim(wide_bc) ==
              DataLayouts.f_dim(typeof(new_data()))
        @test DataLayouts.f_dim(nested_bc) ==
              DataLayouts.f_dim(typeof(new_data()))
    end
end

# Materialize a wide broadcast expression whose arguments mix 0-dimensional
# and 3-dimensional layouts. GPU compilation of expressions like this one
# fails with an InvalidIRError when the broadcast argument traversal functions
# trigger inference's widening heuristics.
@testset "materialize wide broadcast with point and non-point args" begin
    args = (
        ntuple(_ -> fill!(new_data(), FT(1)), 8)...,
        fill!(new_point_data(), FT(2)),
        FT(1),
        Base.broadcasted(
            *,
            fill!(new_point_data(), FT(2)),
            fill!(new_data(), FT(1)),
        ),
        ntuple(_ -> fill!(new_data(), FT(1)), 8)...,
    )
    dest = new_data()
    Base.materialize!(dest, Base.broadcasted(+, args...))
    @test all(==(FT(16 + 2 + 1 + 2)), parent(dest))
end

# Materialize fused broadcasts whose statements mix 0-dimensional and
# 3-dimensional layouts of heterogeneous element types, mirroring the fused
# EDMFX cloud fraction broadcasts in ClimaAtmos AMIP runs. Heterogeneous
# element types also guard against union-typed accumulators in the argument
# traversal functions, which only widen when the layout types differ, so each
# fused broadcast below mixes both element types. The statements are split
# into two fused broadcasts because a single one exceeds the 4 KiB kernel
# parameter limit of sm_60 GPUs (newer architectures allow 32 KiB). The fused
# results are compared against unfused reference results, so the expected
# values do not need to be hand-computed.
@testset "materialize wide heterogeneous fused broadcast" begin
    (a1, a2, a3, a4, a5, a6, a7, a8) =
        ntuple(i -> fill!(new_data(), FT(i)), 8)
    (c1, c2, c3, c4) = ntuple(i -> fill!(new_data(CT), CT(i, 2i)), 4)
    pf = fill!(new_point_data(), FT(2))
    pc = fill!(new_point_data(CT), CT(1, 1))
    (d1, d2, d3, d4, d5, d6, d7, d8) = ntuple(_ -> new_data(), 8)
    (dc1, dc2) = ntuple(_ -> new_data(CT), 2)
    (r1, r2, r3, r4, r5, r6, r7, r8) = ntuple(_ -> new_data(), 8)
    (rc1, rc2) = ntuple(_ -> new_data(CT), 2)

    DataLayouts.@fused_direct begin
        @. d1 = a1 + pf * a2 - a3 + a4 * a5 - a6 + a7 * a8
        @. d2 = a2 - pf + a3 * (a4 - a5) + a6 - a7
        @. d3 = pf * (a1 + a3) - a4 + a5 * a6 + a8
        @. d4 = a1 * a2 + a3 * a4 + a5 * a6 + a7 * a8 - pf
        @. dc1 = c1 + pc * c2 - c3 * c4
    end
    DataLayouts.@fused_direct begin
        @. d5 = (a1 - a2) * (a3 - a4) + (a5 - a6) * (a7 - a8)
        @. d6 = a1 - a4 + pf * (a2 + a6 - a8)
        @. d7 = a7 * pf - a5 + a3 * a1
        @. d8 = a8 + a6 * (a2 - pf) - a4 * a2
        @. dc2 = c2 * c3 - pc + c1 - c4
    end
    @. r1 = a1 + pf * a2 - a3 + a4 * a5 - a6 + a7 * a8
    @. r2 = a2 - pf + a3 * (a4 - a5) + a6 - a7
    @. r3 = pf * (a1 + a3) - a4 + a5 * a6 + a8
    @. r4 = a1 * a2 + a3 * a4 + a5 * a6 + a7 * a8 - pf
    @. r5 = (a1 - a2) * (a3 - a4) + (a5 - a6) * (a7 - a8)
    @. r6 = a1 - a4 + pf * (a2 + a6 - a8)
    @. r7 = a7 * pf - a5 + a3 * a1
    @. r8 = a8 + a6 * (a2 - pf) - a4 * a2
    @. rc1 = c1 + pc * c2 - c3 * c4
    @. rc2 = c2 * c3 - pc + c1 - c4

    fused_and_reference_pairs =
        ((d1, r1), (d2, r2), (d3, r3), (d4, r4), (d5, r5), (d6, r6), (d7, r7),
            (d8, r8), (dc1, rc1), (dc2, rc2))
    for (dest, reference_dest) in fused_and_reference_pairs
        @test Array(parent(dest)) == Array(parent(reference_dest))
    end
end

# The messages for these errors interpolate the scope types on the host; the
# GPU-compiled throw sites only reference singleton exception types.
@testset "scope error messages" begin
    scope1 = DataLayouts.ThisThread()
    scope2 = DataLayouts.ThisThreadPool()
    @test sprint(showerror, DataLayouts.InvalidSubscopeError(scope1, scope2)) ==
          "$(typeof(scope1)) is not a subset of $(typeof(scope2))"
    @test sprint(
        showerror,
        DataLayouts.NonOverlappingScopesError(scope1, scope2),
    ) ==
          "$(typeof(scope1)) and $(typeof(scope2)) do not overlap, so they \
           cannot be put in the same DataScope"
end
