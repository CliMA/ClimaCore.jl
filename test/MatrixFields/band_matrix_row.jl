using Test
using JET
using LinearAlgebra: I

using ClimaCore.MatrixFields
import ClimaCore: Geometry

macro test_all(expression)
    return quote
        local test_func() = $(esc(expression))
        @test test_func()                   # correctness
        @test (@allocated test_func()) == 0 # allocations
        @test_opt test_func()               # type instabilities
    end
end

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

    T(value) = (; a = (), b = value, c = (value, (; d = (value,)), (;)))
    @test_all QuaddiagonalMatrixRow(T(0.5), T(1), T(1), T(1 // 2)) +
              BidiagonalMatrixRow(T(-0.5), T(-1 // 2)) ==
              QuaddiagonalMatrixRow(T(1), T(1), T(1), T(1)) / 2
    @test_all PentadiagonalMatrixRow(T(0), T(0.5), T(1), T(1 // 2), T(0)) -
              TridiagonalMatrixRow(T(1), T(0), T(1)) / 2 -
              0.5 * DiagonalMatrixRow(T(2)) ==
              PentadiagonalMatrixRow(T(0), T(0), T(0), T(0), T(0))

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
end
