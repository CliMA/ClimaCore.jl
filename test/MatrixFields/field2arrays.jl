using Test
using JET

import ClimaCore: Geometry, Domains, Meshes, Spaces, Fields, MatrixFields

@testset "field2arrays Unit Tests" begin
    FT = Float64
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(1)),
        Geometry.ZPoint(FT(4));
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain, nelems = 3)
    center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)
    ᶜz = Fields.coordinate_field(center_space).z
    ᶠz = Fields.coordinate_field(face_space).z

    ᶜᶜmat = map(z -> MatrixFields.TridiagonalMatrixRow(2 * z, 4 * z, 8 * z), ᶜz)
    ᶜᶠmat = map(z -> MatrixFields.BidiagonalMatrixRow(2 * z, 4 * z), ᶜz)
    ᶠᶠmat = map(z -> MatrixFields.TridiagonalMatrixRow(2 * z, 4 * z, 8 * z), ᶠz)
    ᶠᶜmat = map(z -> MatrixFields.BidiagonalMatrixRow(2 * z, 4 * z), ᶠz)

    @test MatrixFields.column_field2array(ᶜz) ==
          MatrixFields.column_field2array_view(ᶜz) ==
          [1.5, 2.5, 3.5]

    @test MatrixFields.column_field2array(ᶠz) ==
          MatrixFields.column_field2array_view(ᶠz) ==
          [1, 2, 3, 4]

    @test MatrixFields.column_field2array(ᶜᶜmat) ==
          MatrixFields.column_field2array_view(ᶜᶜmat) ==
          [
              6 12 0
              5 10 20
              0 7 14
          ]

    @test MatrixFields.column_field2array(ᶜᶠmat) ==
          MatrixFields.column_field2array_view(ᶜᶠmat) ==
          [
              3 6 0 0
              0 5 10 0
              0 0 7 14
          ]

    @test MatrixFields.column_field2array(ᶠᶠmat) ==
          MatrixFields.column_field2array_view(ᶠᶠmat) ==
          [
              4 8 0 0
              4 8 16 0
              0 6 12 24
              0 0 8 16
          ]

    @test MatrixFields.column_field2array(ᶠᶜmat) ==
          MatrixFields.column_field2array_view(ᶠᶜmat) ==
          [
              4 0 0
              4 8 0
              0 6 12
              0 0 8
          ]

    ᶜᶜmat_array_not_view = MatrixFields.column_field2array(ᶜᶜmat)
    ᶜᶜmat_array_view = MatrixFields.column_field2array_view(ᶜᶜmat)
    ᶜᶜmat .*= 2
    @test ᶜᶜmat_array_not_view == MatrixFields.column_field2array(ᶜᶜmat) ./ 2
    @test ᶜᶜmat_array_view == MatrixFields.column_field2array(ᶜᶜmat)

    @test MatrixFields.field2arrays(ᶜᶜmat) ==
          [MatrixFields.column_field2array(ᶜᶜmat)]

    # Check for type instabilities.
    @test_opt broken = true MatrixFields.column_field2array(ᶜᶜmat)
    @test_opt MatrixFields.column_field2array_view(ᶜᶜmat)
    @test_opt broken = true MatrixFields.field2arrays(ᶜᶜmat)

    # Because this test is broken, printing matrix fields allocates some memory.
    @test_broken (@allocated MatrixFields.column_field2array_view(ᶜᶜmat)) == 0
end
