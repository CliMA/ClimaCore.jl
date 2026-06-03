using Test
import ClimaComms
import ClimaCore.DataLayouts: VIJFH
ClimaComms.@import_required_backends

function knl_copy!(dest, src)
    i = threadIdx().x
    j = threadIdx().y
    h = blockIdx().x
    p_dest = slab(dest, h)
    p_src = slab(src, h)
    @inbounds p_dest[1, i, j, 1] = p_src[1, i, j, 1]
    return nothing
end

@testset "data in GPU kernels" begin
    device = ClimaComms.device()
    FT = Float64
    A = ClimaComms.array_type(device){FT}
    T = Tuple{Complex{FT}, FT}
    (Nv, Nij, Nh) = (1, 4, 10)
    src = VIJFH{T, Nv, Nij, Nij, missing}(A, Nh)
    dest = VIJFH{T, Nv, Nij, Nij, missing}(A, Nh)
    CUDA.@cuda threads = (Nij, Nij) blocks = (Nh,) knl_copy!(dest, src)
    @test parent(dest) == parent(src)
end

@testset "broadcasting" begin
    device = ClimaComms.device()
    FT = Float64
    A = ClimaComms.array_type(device){FT}

    T = NamedTuple{(:a, :b), Tuple{Complex{FT}, FT}}
    f(a1, a2) = a1.a.re * a2 + a1.b
    for (Nv, Nij, Nh) in ((1, 2, 2), (33, 4, 2))
        data1 = VIJFH{T, Nv, Nij, Nij, missing}(A, Nh)
        data2 = VIJFH{FT, Nv, Nij, Nij, missing}(A, Nh)
        parent(data1) .= 1
        parent(data2) .= 1
        @test Array(parent(f.(data1, data2))) == repeat(FT[2], Nv, Nij, Nij, 1, Nh)
    end

    T = Complex{FT}
    for (Nv, Nij, Nh) in ((1, 2, 3), (33, 4, 3))
        data = VIJFH{T, Nv, Nij, Nij, missing}(A, Nh)
        data .= Complex(1, 2)
        @test Array(parent(data.re)) == repeat(FT[1], Nv, Nij, Nij, 1, Nh)
        @test Array(parent(data.im)) == repeat(FT[2], Nv, Nij, Nij, 1, Nh)
    end
end
