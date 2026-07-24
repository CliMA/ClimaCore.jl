using Test
import ClimaComms
import ClimaCore: slab
import ClimaCore.DataLayouts: VIJFH
ClimaComms.@import_required_backends

function knl_copy!(dest, src)
    i = CUDA.threadIdx().x
    j = CUDA.threadIdx().y
    h = CUDA.blockIdx().x
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
    src = VIJFH{T, Nv, Nij, Nij, nothing}(A, Nh)
    dest = VIJFH{T, Nv, Nij, Nij, nothing}(A, Nh)
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
        data1 = VIJFH{T, Nv, Nij, Nij, nothing}(A, Nh)
        data2 = VIJFH{FT, Nv, Nij, Nij, nothing}(A, Nh)
        parent(data1) .= 1
        parent(data2) .= 1
        @test Array(parent(f.(data1, data2))) == repeat(FT[2], Nv, Nij, Nij, 1, Nh)
    end

    T = Complex{FT}
    for (Nv, Nij, Nh) in ((1, 2, 3), (33, 4, 3))
        data = VIJFH{T, Nv, Nij, Nij, nothing}(A, Nh)
        data .= Complex(1, 2)
        @test Array(parent(data.re)) == repeat(FT[1], Nv, Nij, Nij, 1, Nh)
        @test Array(parent(data.im)) == repeat(FT[2], Nv, Nij, Nij, 1, Nh)
    end
end

@testset "kernel argument compaction" begin
    import ClimaCore
    import Adapt
    ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
    FT = Float32
    T = NamedTuple{(:a, :b, :c), NTuple{3, FT}}
    data = VIJFH{T, 10, 4, 4, nothing}(CUDA.CuArray{FT}, 20)
    to = CUDA.KernelAdaptor()

    # Full arrays pass through unchanged; every view compacts to a
    # CompactDeviceView whose type is statically inferred from the host types,
    # so that kernel launches never hit dynamic dispatch.
    full = @inferred Adapt.adapt(to, data)
    @test parent(full) isa CUDA.CuDeviceArray
    for view_data in (data.b, ClimaCore.level(data, 2))
        compact = @inferred Adapt.adapt(to, view_data)
        @test parent(compact) isa ext.CompactDeviceView
        @test size(parent(compact)) == size(parent(view_data))
        @test sizeof(typeof(parent(compact))) <
              sizeof(typeof(Adapt.adapt(to, parent(view_data))))
    end
end
