using StaticArrays: @SMatrix

import ClimaCore.MatrixFields: rmul_with_projection

include("matrix_field_test_utils.jl")

function test_rmul_with_projection(
    x::X,
    y::Y,
    lg::LG,
    expected_result,
) where {X, Y, LG}
    result = rmul_with_projection(x, y, lg)
    result_type = Base.promote_op(rmul_with_projection, X, Y, LG)

    # Compute the maximum error as an integer multiple of machine epsilon.
    FT = Geometry.undertype(LG)
    object2tuple(obj) =
        reinterpret(NTuple{sizeof(obj) ÷ sizeof(FT), FT}, [obj])[1]
    max_error = maximum(
        ((value, expected_value),) ->
            Int(abs(value - expected_value) / eps(expected_value)),
        zip(object2tuple(result), object2tuple(expected_result)),
    )

    @test max_error <= 1                                                     # correctness
    @test (@allocated rmul_with_projection(x, y, lg)) == 0                   # allocations
    @test_opt rmul_with_projection(x, y, lg)                                 # type instabilities

    @test result_type == typeof(result)                                      # correctness
    @test (@allocated Base.promote_op(rmul_with_projection, X, Y, LG)) == 0  # allocations
    @test_opt Base.promote_op(rmul_with_projection, X, Y, LG)                # type instabilities
end

@testset "rmul_with_projection Unit Tests" begin
    seed!(1) # ensures reproducibility

    FT = Float64
    coord = Geometry.LatLongZPoint(rand(FT), rand(FT), rand(FT))
    ∂x∂ξ = Geometry.AxisTensor(
        (Geometry.LocalAxis{(1, 2, 3)}(), Geometry.CovariantAxis{(1, 2, 3)}()),
        (@SMatrix rand(FT, 3, 3)),
    )
    lg = Geometry.LocalGeometry(coord, rand(FT), rand(FT), ∂x∂ξ)

    number = rand(FT)
    vector = Geometry.Covariant123Vector(rand(FT), rand(FT), rand(FT))
    covector = Geometry.Covariant12Vector(rand(FT), rand(FT))'
    tensor = vector * covector
    cotensor =
        (covector' * Geometry.Contravariant12Vector(rand(FT), rand(FT))')'

    dual_axis = Geometry.Contravariant12Axis()
    projected_vector = Geometry.project(dual_axis, vector, lg)
    projected_tensor = Geometry.project(dual_axis, tensor, lg)

    # Test all valid combinations of single values.
    test_rmul_with_projection(number, number, lg, number * number)
    test_rmul_with_projection(number, vector, lg, number * vector)
    test_rmul_with_projection(number, tensor, lg, number * tensor)
    test_rmul_with_projection(number, covector, lg, number * covector)
    test_rmul_with_projection(number, cotensor, lg, number * cotensor)
    test_rmul_with_projection(vector, number, lg, vector * number)
    test_rmul_with_projection(vector, covector, lg, vector * covector)
    test_rmul_with_projection(tensor, number, lg, tensor * number)
    test_rmul_with_projection(tensor, vector, lg, tensor * projected_vector)
    test_rmul_with_projection(tensor, tensor, lg, tensor * projected_tensor)
    test_rmul_with_projection(tensor, cotensor, lg, tensor * cotensor)
    test_rmul_with_projection(covector, number, lg, covector * number)
    test_rmul_with_projection(covector, vector, lg, covector * projected_vector)
    test_rmul_with_projection(covector, tensor, lg, covector * projected_tensor)
    test_rmul_with_projection(covector, cotensor, lg, covector * cotensor)
    test_rmul_with_projection(cotensor, number, lg, cotensor * number)
    test_rmul_with_projection(cotensor, vector, lg, cotensor * projected_vector)
    test_rmul_with_projection(cotensor, tensor, lg, cotensor * projected_tensor)
    test_rmul_with_projection(cotensor, cotensor, lg, cotensor * cotensor)

    # Test some combinations of complicated nested values.
    T = nested_type
    test_rmul_with_projection(
        number,
        T(covector, vector, tensor),
        lg,
        T(number * covector, number * vector, number * tensor),
    )
    test_rmul_with_projection(
        T(covector, vector, tensor),
        number,
        lg,
        T(covector * number, vector * number, tensor * number),
    )
    test_rmul_with_projection(
        vector,
        T(number, number, number),
        lg,
        T(vector * number, vector * number, vector * number),
    )
    test_rmul_with_projection(
        T(number, number, number),
        covector,
        lg,
        T(number * covector, number * covector, number * covector),
    )
    test_rmul_with_projection(
        T(number, vector, number),
        T(covector, number, tensor),
        lg,
        T(number * covector, vector * number, number * tensor),
    )
    test_rmul_with_projection(
        T(covector, number, tensor),
        T(number, vector, number),
        lg,
        T(covector * number, number * vector, tensor * number),
    )
    test_rmul_with_projection(
        covector,
        T(vector, number, tensor),
        lg,
        T(
            covector * projected_vector,
            covector * number,
            covector * projected_tensor,
        ),
    )
    test_rmul_with_projection(
        T(covector, number, covector),
        vector,
        lg,
        T(
            covector * projected_vector,
            number * vector,
            covector * projected_vector,
        ),
    )
    test_rmul_with_projection(
        T(covector, number, covector),
        T(number, vector, tensor),
        lg,
        T(covector * number, number * vector, covector * projected_tensor),
    )
end
