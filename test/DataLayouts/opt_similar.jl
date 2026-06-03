using Test
using JET
using StaticArrays: SMatrix, MArray
import ClimaComms
import ClimaCore: DataLayouts, Geometry
ClimaComms.@import_required_backends

function test_similar!(data)
    if isnothing(DataLayouts.f_dim(data))
        new_data = similar(data)
        @test_opt similar(data)
    else
        FT = eltype(parent(data))
        LG = Geometry.LocalGeometryType(Geometry.ZPoint{FT}, FT, (3,))
        new_data = similar(data, LG)
        @test_opt similar(data, LG)
    end
    DataLayouts.DataScope(data) == DataLayouts.ThisThread() &&
        DataLayouts.has_inferred_size(data) &&
        @test parent(new_data) isa MArray
end

@testset "similar" begin
    device = ClimaComms.device()
    FT = Float64
    A = ClimaComms.array_type(device){FT}

    test_similar!(DataLayouts.DataF{FT}(A))

    (Nv, Nij, Nh) = (4, 3, 5)
    for Nh in (1, Nh)
        Nh_parameter = Nh == 1 ? 1 : missing
        test_similar!(DataLayouts.VIJFH{FT, Nv, Nij, Nij, Nh_parameter}(A, Nh))
        test_similar!(DataLayouts.VIJHF{FT, Nv, Nij, Nij, Nh_parameter}(A, Nh))
        test_similar!(DataLayouts.VIH1{FT, Nv, Nij, Nh_parameter}(A, Nh))
        test_similar!(DataLayouts.IH1JH2{FT, Nij, Nij, Nh_parameter}(A, Nh))
    end
end
