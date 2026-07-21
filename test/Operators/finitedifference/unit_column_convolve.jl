# test/Operators/finitedifference/unit_column_convolve.jl

using Test
using ClimaComms
import ClimaCore: DataLayouts, Operators, Geometry
using ClimaCore.DataLayouts: VF

ClimaComms.@import_required_backends

device = ClimaComms.device()
ArrayType = ClimaComms.array_type(device)

const getidx_extrapolated = Operators.getidx_extrapolated
const ConvolutionKernel = Operators.ConvolutionKernel

@testset "apply_kernel! — scalar VF" begin
    FT = Float64
    Nv = 10

    kernel = ConvolutionKernel{-1}(FT(0.25), FT(0.5), FT(0.25))

    # Build input VF with known values
    in_data = VF{FT}(ArrayType{FT}, zeros; Nv)
    parent(in_data)[:, 1] .= FT.(1:Nv)
    out_data = similar(in_data)


    # Expected with zero-padding at boundaries:
    expected = ArrayType{FT}(undef, Nv)
    v = Float64[0; 1:Nv; 0]  # zero-padded values
    for i in 1:Nv
        expected[i] = 0.25 * v[i] + 0.5 * v[i + 1] + 0.25 * v[i + 2]
    end

    Operators.apply_kernel!(out_data, kernel, in_data)
    @test parent(out_data)[:, 1] ≈ expected
end

@testset "apply_kernel! — UVWVector VF" begin
    FT = Float64
    Nv = 10
    S = Geometry.UVWVector{FT}

    kernel = ConvolutionKernel{-1}(FT(0.25), FT(0.5), FT(0.25))

    # Build input VF with known UVWVector values
    # Each level i has u=i, v=2i, w=3i
    in_data = VF{S}(ArrayType{FT}, zeros; Nv)
    for i in 1:Nv
        in_data[i] = S(FT(i), FT(2i), FT(3i))
    end
    out_data = similar(in_data)

    # Expected with zero-padding at boundaries:
    # i=1: 0.25*zero(S) + 0.5*S(1,2,3) + 0.25*S(2,4,6) = S(1.25, 2.5, 3.75)
    # i=2: 0.25*S(1,2,3) + 0.5*S(2,4,6) + 0.25*S(3,6,9) = S(2.25, 4.5, 6.75)
    # i=Nv: 0.25*S(Nv-1,2(Nv-1),3(Nv-1)) + 0.5*S(Nv,2Nv,3Nv) + 0.25*zero(S)
    expected = Vector{S}(undef, Nv)
    v = Vector{S}(undef, Nv + 2)
    v[1] = zero(S)  # left padding
    for i in 1:Nv
        v[i + 1] = S(FT(i), FT(2i), FT(3i))
    end
    v[Nv + 2] = zero(S)  # right padding

    for i in 1:Nv
        expected[i] = FT(0.25) * v[i] + FT(0.5) * v[i + 1] + FT(0.25) * v[i + 2]
    end

    Operators.apply_kernel!(out_data, kernel, in_data)
    for i in 1:Nv
        @test out_data[i] ≈ expected[i]
    end
end
