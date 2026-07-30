#=
julia --project
using Revise; include(joinpath("test", "Operators", "spectralelement", "unit_tensor_product.jl"))
=#
using Test
using LinearAlgebra
import ClimaCore: DataLayouts, Operators, Quadratures

# `tensor_product!` writes into three different output layouts and is also used in
# place, but only its `IH1JH2` and `VIH1` outputs are reached from
# `matrix_interpolate`; the `VIJFH` output (ClimaCorePlots) and the two-argument
# in-place form (examples/bickleyjet) have no other coverage. Every case is
# checked against an explicit `M * x * M'`.
@testset "tensor_product!" begin
    FT = Float64
    Nij, Nu, Nh, Nv = 4, 5, 3, 2
    M = Quadratures.interpolation_matrix(
        FT,
        Quadratures.Uniform{Nu}(),
        Quadratures.GLL{Nij}(),
    )

    # two node axes: the same M is contracted along both
    indata = DataLayouts.VIJFH{FT, 1, Nij, Nij, Nh}(rand(FT, 1, Nij, Nij, 1, Nh))
    ref = map(1:Nh) do h
        M * reshape(parent(indata)[1, :, :, 1, h], Nij, Nij) * M'
    end

    out = DataLayouts.VIJFH{FT, 1, Nu, Nu, Nh}(zeros(FT, 1, Nu, Nu, 1, Nh))
    Operators.tensor_product!(out, indata, M)
    @test all(h -> parent(out)[1, :, :, 1, h] ≈ ref[h], 1:Nh)

    out = DataLayouts.IH1JH2{FT, Nu, Nu, nothing}(zeros(FT, Nu * Nh, Nu))
    Operators.tensor_product!(out, indata, M)
    @test all(h -> parent(out)[(Nu * (h - 1) + 1):(Nu * h), :] ≈ ref[h], 1:Nh)

    # one node axis, several levels
    in1d = DataLayouts.VIJFH{FT, Nv, Nij, 1, Nh}(rand(FT, Nv, Nij, 1, 1, Nh))
    ref1d = [M * parent(in1d)[v, :, 1, 1, h] for v in 1:Nv, h in 1:Nh]
    vhs = ((v, h) for v in 1:Nv, h in 1:Nh)

    out = DataLayouts.VIH1{FT, Nv, Nu, nothing}(zeros(FT, Nv, Nu * Nh))
    Operators.tensor_product!(out, in1d, M)
    @test all(
        ((v, h),) -> parent(out)[v, (Nu * (h - 1) + 1):(Nu * h)] ≈ ref1d[v, h],
        vhs,
    )

    out = DataLayouts.VIJFH{FT, Nv, Nu, 1, Nh}(zeros(FT, Nv, Nu, 1, 1, Nh))
    Operators.tensor_product!(out, in1d, M)
    @test all(((v, h),) -> parent(out)[v, :, 1, 1, h] ≈ ref1d[v, h], vhs)

    # in place, which needs a square M
    Msq = Quadratures.cutoff_filter_matrix(FT, Quadratures.GLL{Nij}(), 3)
    inout = DataLayouts.VIJFH{FT, 1, Nij, Nij, Nh}(copy(parent(indata)))
    refsq = map(1:Nh) do h
        Msq * reshape(parent(indata)[1, :, :, 1, h], Nij, Nij) * Msq'
    end
    Operators.tensor_product!(inout, Msq)
    @test all(h -> parent(inout)[1, :, :, 1, h] ≈ refsq[h], 1:Nh)
end
