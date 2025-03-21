#=
julia --project=test
using Revise; include(joinpath("test", "DataLayouts", "data2dx.jl"))
=#
using Test
using ClimaComms
ClimaComms.@import_required_backends
using ClimaCore.DataLayouts
using ClimaCore.Geometry
import ClimaCore.DataLayouts:
    VF,
    VIFH,
    VIJFH,
    VIJHF,
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

        data3 = VIJFH{WVector{FT}, Nv, Nij}(
            ArrayType(
                reshape(FT(1.0):(Nv * Nij * Nij * Nh), Nv, Nij, Nij, 1, Nh),
            ),
        )
        array3 = ArrayType{FT}(undef, nl, ncol)
        array3t = ArrayType{FT}(undef, ncol, nl)

        data2array_rrtmgp!(array3, data3, Val(false))
        data2array_rrtmgp!(array3t, data3, Val(true))

        @test array3 == ArrayType(transpose(array3t))


        array4 = ArrayType(reshape(FT(1.0):(Nv * Nij * Nij * Nh), Nv, :))
        array4t = ArrayType{FT}(undef, ncol, nl)

        data4 = VIJFH{FT, Nv, Nij}(ArrayType{FT}(undef, Nv, Nij, Nij, 1, Nh))

        array2data_rrtmgp!(data4, array4, Val(false))
        @test parent(data4) == parent(data4)
    end
end

@testset "VIFH data2array_rrtmgp!" begin
    Nv = 10 # number of vertical levels
    Ni = 4 # Nij × Nij nodal points per element
    Nh = 10 # number of elements

    nl = Nv # number of levels/layers in the array
    ncol = Ni * Nh # number of columns in the array

    for FT in (Float32, Float64)
        data1 = VIFH{FT, Nv, Ni}(
            ArrayType(reshape(FT(1.0):(Nv * Ni * Nh), Nv, Ni, 1, Nh)),
        )

        array1 = ArrayType{FT}(undef, nl, ncol)
        array1t = ArrayType{FT}(undef, ncol, nl)

        data2array_rrtmgp!(array1, data1, Val(false))
        data2array_rrtmgp!(array1t, data1, Val(true))

        @test array1 == ArrayType(transpose(array1t))

        array2 = ArrayType(reshape(FT(1.0):(Nv * Ni * Nh), Nv, :))
        array2t = ArrayType{FT}(undef, ncol, nl)

        data2 = VIFH{FT, Nv, Ni}(ArrayType{FT}(undef, Nv, Ni, 1, Nh))

        array2data_rrtmgp!(data2, array2, Val(false))
        @test parent(data2) == parent(data1)

        array2t = ArrayType(transpose(Array(array2)))

        parent(data2) .= NaN
        array2data_rrtmgp!(data2, array2t, Val(true))
        @test parent(data2) == parent(data1)

        data3 = VIFH{WVector{FT}, Nv, Ni}(
            ArrayType(reshape(FT(1.0):(Nv * Ni * Nh), Nv, Ni, 1, Nh)),
        )
        array3 = ArrayType{FT}(undef, nl, ncol)
        array3t = ArrayType{FT}(undef, ncol, nl)

        data2array_rrtmgp!(array3, data3, Val(false))
        data2array_rrtmgp!(array3t, data3, Val(true))

        @test array3 == ArrayType(transpose(array3t))


        array4 = ArrayType(reshape(FT(1.0):(Nv * Ni * Nh), Nv, :))
        array4t = ArrayType{FT}(undef, ncol, nl)

        data4 = VIFH{FT, Nv, Ni}(ArrayType{FT}(undef, Nv, Ni, 1, Nh))

        array2data_rrtmgp!(data4, array4, Val(false))
        @test parent(data4) == parent(data4)
    end
end

@testset "VF data2array_rrtmgp!" begin
    Nv = 10 # number of vertical levels

    nl = Nv # number of levels/layers in the array
    ncol = 1# number of columns in the array

    for FT in (Float32, Float64)
        data1 = VF{FT, Nv}(ArrayType(reshape(FT(1.0):(Nv), Nv, 1)))

        array1 = ArrayType{FT}(undef, nl, ncol)
        array1t = ArrayType{FT}(undef, ncol, nl)

        data2array_rrtmgp!(array1, data1, Val(false))
        data2array_rrtmgp!(array1t, data1, Val(true))

        @test array1 == ArrayType(transpose(array1t))

        array2 = ArrayType(reshape(FT(1.0):(Nv), Nv, :))
        array2t = ArrayType{FT}(undef, ncol, nl)

        data2 = VF{FT, Nv}(ArrayType{FT}(undef, Nv, 1))

        array2data_rrtmgp!(data2, array2, Val(false))
        @test parent(data2) == parent(data1)

        array2t = ArrayType(transpose(Array(array2)))

        parent(data2) .= NaN
        array2data_rrtmgp!(data2, array2t, Val(true))
        @test parent(data2) == parent(data1)

        data3 = VF{WVector{FT}, Nv}(ArrayType(reshape(FT(1.0):(Nv), Nv, 1)))
        array3 = ArrayType{FT}(undef, nl, ncol)
        array3t = ArrayType{FT}(undef, ncol, nl)

        data2array_rrtmgp!(array3, data3, Val(false))
        data2array_rrtmgp!(array3t, data3, Val(true))

        @test array3 == ArrayType(transpose(array3t))


        array4 = ArrayType(reshape(FT(1.0):(Nv), Nv, :))
        array4t = ArrayType{FT}(undef, ncol, nl)

        data4 = VF{FT, Nv}(ArrayType{FT}(undef, Nv, 1))

        array2data_rrtmgp!(data4, array4, Val(false))
        @test parent(data4) == parent(data4)
    end
end
