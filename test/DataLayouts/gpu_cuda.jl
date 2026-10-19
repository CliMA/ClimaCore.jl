using Test
import ClimaComms
using CUDA
import ClimaCore
import ClimaCore: slab, DataLayouts, Fields, Geometry
import ClimaCore.DataLayouts: VIJFH
include(joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"))
import .TestUtilities as TU
ClimaComms.@import_required_backends

# Cross-check every launch configuration computed by this test against CUDA's
# occupancy API. The check is disabled by default so that a tie-breaking
# difference on an untested device degrades performance instead of crashing.
ClimaCore.DataLayouts.VALIDATE_LAUNCH_CONFIGURATIONS[] = true

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
    import ClimaCore.Adapt: Adapt
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

# Point loops whose bodies broadcast over many single-point views
struct ScaledSum{FT}
    scale::FT
end
Base.broadcastable(f::ScaledSum) = tuple(f)
@inline (f::ScaledSum)(args...) = f.scale * +(args...)

@testset "point broadcasts nested in point loops" begin
    device = ClimaComms.device()
    FT = Float32
    A = ClimaComms.array_type(device){FT}
    (Nv, Nij, Nh) = (10, 4, 5)
    dest = VIJFH{FT, Nv, Nij, Nij, nothing}(A, Nh)
    args = ntuple(7) do n
        data = VIJFH{FT, Nv, Nij, Nij, nothing}(A, Nh)
        parent(data) .= n
        data
    end
    f = ScaledSum(FT(2))
    c = FT(1)
    for n in (1, 5, 7)
        ClimaCore.DataLayouts.foreach_point(dest, args[1:n]...) do dest_point, arg_points...
            @. dest_point = f(arg_points..., c)
        end
        @test all(==(2 * (sum(1:n) + 1)), Array(parent(dest)))
    end
end

# Points of CuArrays outside of kernels are views at Cartesian indices, which GPUArrays can
# broadcast over, including points of layouts with several components at linear indices.
@testset "host views of points in GPU layouts" begin
    data = VIJFH{Tuple{Float64, Float64}, 3, 2, 2, nothing}(CuArray{Float64}, 3)
    fill!(parent(data), 0)
    for index in (5, CartesianIndex(2, 1, 2, 3))
        point = view(data, index)
        @test parent(parent(point)) isa CuArray
        parent(point) .+= 1
    end
    @test count(==(1), Array(parent(data))) == 4
end

@testset "point views of compact device views" begin
    CUDA_EXT = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
    # Each component of a point in a view of a parent array must be found at the parent
    # index of the corresponding view element, including in views of single levels.
    for (parent_size, indices, F) in (
        ((3, 2, 2, 3, 2), (2:2, :, :, :, :), 5), # VIJHF level
        ((3, 2, 2, 2, 1), (2:2, :, :, :, :), 4), # VIJFH level with one element
        ((3, 2, 2, 2, 3), (2:2, :, :, :, :), 4), # VIJFH level
        ((3, 2, 2, 3, 2), (:, 2:2, 1:1, 2:2, :), 5), # VIJHF column
        ((3, 2, 2, 2, 3), (2:2, :, :, :, 3:3), 4), # VIJFH slab
    )
        array = reshape(collect(1.0:prod(parent_size)), parent_size)
        array_view = view(array, indices...)
        R = filter(d -> indices[d] isa UnitRange, ntuple(identity, length(parent_size)))
        compact_view = CUDA_EXT.CompactDeviceView{
            Float64, length(parent_size), size(array_view), R, length(R), 0,
            typeof(array),
        }(
            array,
            map(d -> UInt32(first(indices[d]) - 1), R),
            (),
        )
        point_indices = CartesianIndices(size(array_view)[filter(!=(F), 1:length(indices))])
        for point_index in point_indices
            index = (Tuple(point_index)[1:(F - 1)]..., 1, Tuple(point_index)[F:end]...)
            (parent_array, (offset, parent_stride)) = CUDA_EXT.device_offset_and_stride(
                compact_view,
                LinearIndices(array_view)[index...],
                prod(size(array_view)[1:(F - 1)]),
            )
            @test [
                parent_array[offset + (f - 1) * parent_stride] for
                f in 1:size(array_view, F)
            ] ==
                  array_view[Base.setindex(index, :, F)...]
        end
    end

    # Writing to a level of a field with multiple components.
    FT = Float64
    for VIJH in (DataLayouts.VIJHF, DataLayouts.VIJFH), helem in (1, 2)
        space = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem = 2, helem, VIJH)
        cpu_space = TU.CenterExtrudedFiniteDifferenceSpace(
            FT;
            zelem = 2,
            helem,
            VIJH,
            context = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
        )
        value = Geometry.Covariant12Vector(FT(1), FT(2))
        uₕ = fill(zero(value), space)
        cpu_uₕ = fill(zero(value), cpu_space)
        fill!(Fields.level(uₕ, 1), value)
        fill!(Fields.level(cpu_uₕ, 1), value)
        @test Array(parent(uₕ)) == parent(cpu_uₕ)
    end
end
