using LinearAlgebra: I
using ClimaCore.Utilities: enable_auto_broadcasting

include("matrix_field_test_utils.jl")

@testset "BandMatrixRow Unit Tests" begin
    @test_all DiagonalMatrixRow(1) ==
              DiagonalMatrixRow(0.5) + DiagonalMatrixRow(1 // 2) ==
              DiagonalMatrixRow(1.5) - DiagonalMatrixRow(1 // 2) ==
              DiagonalMatrixRow(0.5) * 2 ==
              0.5 * DiagonalMatrixRow(2) ==
              DiagonalMatrixRow(2) / 2 ==
              I

    @test_all DiagonalMatrixRow(1 // 2) + 0.5 * I === DiagonalMatrixRow(1.0)
    @test_all BidiagonalMatrixRow(1 // 2, 0.5) === BidiagonalMatrixRow(1, 1) / 2

    @test_all convert(TridiagonalMatrixRow{Int}, DiagonalMatrixRow(1)) ===
              convert(TridiagonalMatrixRow{Int}, I) ===
              TridiagonalMatrixRow(0, 1, 0)

    @test_all QuaddiagonalMatrixRow(0.5, 1, 1, 1 // 2) +
              BidiagonalMatrixRow(-0.5, -1 // 2) ==
              QuaddiagonalMatrixRow(1, 1, 1, 1) / 2
    @test_all PentadiagonalMatrixRow(0, 0.5, 1, 1 // 2, 0) -
              TridiagonalMatrixRow(1, 0, 1) / 2 - 0.5 * DiagonalMatrixRow(2) ==
              PentadiagonalMatrixRow(0, 0, 0, 0, 0)

    @test_all PentadiagonalMatrixRow(0, 0.5, 1, 1 // 2, 0) -
              TridiagonalMatrixRow(1, 0, 1) / 2 - I ==
              zero(PentadiagonalMatrixRow{Int})

    NT = enable_auto_broadcasting ∘ nested_type
    @test_all QuaddiagonalMatrixRow(NT(0.5), NT(1), NT(1), NT(1 // 2)) +
              BidiagonalMatrixRow(NT(-0.5), NT(-1 // 2)) ==
              QuaddiagonalMatrixRow(NT(1), NT(1), NT(1), NT(1)) / 2
    @test_all PentadiagonalMatrixRow(NT(0), NT(0.5), NT(1), NT(1 // 2), NT(0)) -
              TridiagonalMatrixRow(NT(1), NT(0), NT(1)) / 2 -
              0.5 * DiagonalMatrixRow(NT(2)) ==
              PentadiagonalMatrixRow(NT(0), NT(0), NT(0), NT(0), NT(0))

    @test_throws "Cannot promote" BidiagonalMatrixRow(1, 1) + I
    @test_throws "Cannot promote" BidiagonalMatrixRow(1, 1) +
                                  DiagonalMatrixRow(1)

    @test_throws "Cannot convert" convert(BidiagonalMatrixRow{Int}, I)
    @test_throws "Cannot convert" convert(
        BidiagonalMatrixRow{Int},
        DiagonalMatrixRow(1),
    )
    @test_throws "Cannot convert" convert(
        TridiagonalMatrixRow{Int},
        PentadiagonalMatrixRow(0, 0, 1, 0, 0),
    )

    @test map(tuple, DiagonalMatrixRow(1), DiagonalMatrixRow(1.0)) ==
          DiagonalMatrixRow{Tuple{Int, Float64}}(((1, 1.0),))
    @test map(tuple, DiagonalMatrixRow(1)) == DiagonalMatrixRow(tuple(1))
    @test zero(typeof(DiagonalMatrixRow(1))) == DiagonalMatrixRow(0)
    @test eltype(typeof(DiagonalMatrixRow(1))) == Int
    @test inv(DiagonalMatrixRow(1)) == DiagonalMatrixRow(float(1))
end
