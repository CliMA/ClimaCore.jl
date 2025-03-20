#=
julia --project=test
using Revise; include(joinpath("test", "DataLayouts", "data2dx.jl"))
=#
using Test
using ClimaComms
ClimaComms.@import_required_backends
using ClimaCore.DataLayouts
import ClimaCore.DataLayouts:
    VF,
    IJFH,
    VIJFH,
    slab,
    column,
    slab_index,
    vindex,
    data2array_rrtmgp!,
    array2data_rrtmgp!

device = ClimaComms.device()
ArrayType = ClimaComms.array_type(device)

@testset "VIJFH data2array_rrtmgp!" begin
    Nv = 10 # number of vertical levels
    Nij = 4 # Nij × Nij nodal points per element
    Nh = 10 # number of elements

    nl = Nv # number of levels/layers in the array
    ncol = Nij * Nij * Nh # number of columns in the array

    for FT in (Float32, Float64)
        data1 = VIJFH{FT, Nv, Nij}(
            ArrayType(
                reshape(FT(1.0):(Nv * Nij * Nij * Nh), Nv, Nij, Nij, 1, Nh),
            ),
        )

        array1 = ArrayType{FT}(undef, nl, ncol)
        array1t = ArrayType{FT}(undef, ncol, nl)

        data2array_rrtmgp!(array1, data1, Val(false))
        data2array_rrtmgp!(array1t, data1, Val(true))

        @test array1 == ArrayType(transpose(array1t))

        array2 = ArrayType(reshape(FT(1.0):(Nv * Nij * Nij * Nh), Nv, :))
        array2t = ArrayType{FT}(undef, ncol, nl)

        data2 = VIJFH{FT, Nv, Nij}(ArrayType{FT}(undef, Nv, Nij, Nij, 1, Nh))

        array2data_rrtmgp!(data2, array2, Val(false))
        @test parent(data2) == parent(data1)

        array2t = ArrayType(transpose(Array(array2)))

        parent(data2) .= NaN
        array2data_rrtmgp!(data2, array2t, Val(true))
        @test parent(data2) == parent(data1)
    end
end
