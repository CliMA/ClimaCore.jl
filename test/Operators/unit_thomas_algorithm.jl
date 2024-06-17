#=
julia --project
using Revise; include(joinpath("test", "Operators", "unit_thomas_algorithm.jl"))
=#
using Test
import Random: seed!
import LinearAlgebra: Tridiagonal, norm
import ClimaCore
import ClimaCore: Geometry, Spaces, Fields, Operators
import ClimaComms
ClimaComms.@import_required_backends

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

function test_thomas_algorithm(space)
    coords = Fields.coordinate_field(space)

    # Set the seed to ensure reproducibility.
    seed!(1)

    # Set A to a random diagonally dominant tri-diagonal matrix.
    A = map(coords) do coord
        FT = Geometry.float_type(coord)
        Operators.StencilCoefs{-1, 1}((rand(FT), 10 + rand(FT), rand(FT)))
    end

    # Set b to a random vector.
    b = map(coord -> rand(Geometry.float_type(coord)), coords)

    # Copy A and b, since they will be overwritten by column_thomas_solve!.
    A_copy = copy(A)
    b_copy = copy(b)

    Operators.column_thomas_solve!(A, b)

    # Verify that column_thomas_solve! correctly replaced b with A \ b.
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    for i in 1:Ni, j in 1:Nj, h in 1:Nh
        A_column_data = Array(parent(Spaces.column(A_copy, i, j, h)))
        A_column_array = Tridiagonal(
            A_column_data[2:end, 1],
            A_column_data[:, 2],
            A_column_data[1:(end - 1), 3],
        )
        b_column_array = Array(parent(Spaces.column(b_copy, i, j, h)))[:]
        x_column_array = Array(parent(Spaces.column(b, i, j, h)))[:]
        x_column_array_ref = A_column_array \ b_column_array
        FT = Spaces.undertype(space)
        @test all(@. abs(x_column_array - x_column_array_ref) < eps(FT))
    end
end

@testset "Thomas Algorithm unit tests" begin
    for FT in (Float32, Float64),
        space in (
            TU.ColumnCenterFiniteDifferenceSpace(FT),
            TU.ColumnFaceFiniteDifferenceSpace(FT),
            TU.CenterExtrudedFiniteDifferenceSpace(FT),
            TU.FaceExtrudedFiniteDifferenceSpace(FT),
        )

        test_thomas_algorithm(space)
    end
end
