# Layout-level guards for slice indexing and setindex!:
#  - `slab`/`column`/`level` return correctly shaped views and throw on
#    out-of-range indices;
#  - `setindex!` converts compatible values and throws on incompatible ones
#    instead of silently corrupting the parent array.
using Test
import ClimaCore.DataLayouts
import ClimaCore.DataLayouts: VIJFH, slab, column, level

@testset "slab/column/level shapes and bounds [$FT]" for FT in
                                                         (Float32, Float64)
    (Nv, Nij, Nh) = (4, 3, 5)
    data = VIJFH{FT, Nv, Nij, Nij, nothing}(Array{FT}, Nh)
    parent(data) .= FT(0)

    @test size(column(data, 1, 2, 3)) == (Nv, 1, 1, 1)
    @test size(slab(data, 2, 4)) == (1, Nij, Nij, 1)
    @test size(level(data, 3)) == (1, Nij, Nij, Nh)

    @test_throws BoundsError column(data, Nij + 1, 1, 1)
    @test_throws BoundsError column(data, 1, Nij + 1, 1)
    @test_throws BoundsError column(data, 1, 1, Nh + 1)
    @test_throws BoundsError slab(data, Nv + 1, 1)
    @test_throws BoundsError slab(data, 1, Nh + 1)
    @test_throws BoundsError level(data, Nv + 1)

    # Views alias the parent data.
    col = column(data, 1, 1, 1)
    col[1, 1, 1, 1] = FT(7)
    @test data[1, 1, 1, 1] == FT(7)

    # A layout with a single slab and column still checks its indices.
    single = VIJFH{FT, 1, 1, 1, nothing}(Array{FT}, 1)
    parent(single) .= FT(0)
    @test size(slab(single, 1, 1)) == (1, 1, 1, 1)
    @test size(column(single, 1, 1, 1)) == (1, 1, 1, 1)
    @test_throws BoundsError slab(single, 2, 1)
    @test_throws BoundsError column(single, 2, 1, 1)
end

@testset "setindex! type safety" begin
    data = VIJFH{Tuple{Int64, Float64}, 2, 2, 2, nothing}(Array{Float64}, 2)
    parent(data) .= 0.0

    # Compatible values are stored via convert.
    data[1, 1, 1, 1] = (1, 2.5)
    @test data[1, 1, 1, 1] === (Int64(1), 2.5)
    data[2, 1, 1, 1] = (Int32(2), 3)
    @test data[2, 1, 1, 1] === (Int64(2), 3.0)

    # Incompatible values throw instead of corrupting the parent array.
    @test_throws MethodError data[1, 1, 1, 1] = (1, 2.5, 3.0)
    @test_throws MethodError data[1, 1, 1, 1] = "invalid"
    @test data[1, 1, 1, 1] === (Int64(1), 2.5) # unchanged by failed writes

    scalar_data = VIJFH{Float64, 2, 2, 2, nothing}(Array{Float64}, 2)
    parent(scalar_data) .= 0.0
    scalar_data[1, 1, 1, 1] = 1 // 3
    @test scalar_data[1, 1, 1, 1] == 1 / 3
    @test_throws InexactError scalar_data[1, 1, 1, 1] = 1.5 + 2.0im
end
