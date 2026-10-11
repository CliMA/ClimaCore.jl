using Test
using JET
import Random
using StaticArrays: @SMatrix

import ClimaCore.Geometry
import ClimaCore.Utilities: add_auto_broadcasters, return_type

nested_type(value) = nested_type(value, value, value)
nested_type(value1, value2, value3) =
    (; a = (), b = value1, c = (value2, (; d = (value3,)), (;)))

# The product of x and y with y projected for it, as in the rows of band matrix
# products, using either all of the local geometry or only the metric that the
# projection reads from it (as in MatrixFields.projected_operand).
function mul_with_projection(x, y, lg)
    axes = Geometry._dual_axes_for_projection(typeof(x))
    return x * Geometry.project_for_mul(axes, y, lg)
end
function mul_with_projection_metric(x, y, lg)
    axes = Geometry._dual_axes_for_projection(typeof(x))
    metric = Geometry.projection_metric(axes, typeof(y), lg)
    return x * Geometry.project_for_mul(axes, y, metric)
end

function test_mul_with_projection(x::X, y::Y, lg, expected_result) where {X, Y}
    for f in (mul_with_projection, mul_with_projection_metric)
        result = f(x, y, lg)
        result_type = return_type(f, Tuple{X, Y, typeof(lg)})

        # Compute the maximum error as an integer multiple of machine epsilon.
        FT = Geometry.undertype(typeof(lg))
        object2tuple(obj) =
            reinterpret(NTuple{sizeof(obj) ÷ sizeof(FT), FT}, [obj])[1]
        max_error = maximum(
            ((value, expected_value),) ->
                Int(abs(value - expected_value) / eps(expected_value)),
            zip(object2tuple(result), object2tuple(expected_result)),
        )

        @test max_error <= 1                     # correctness
        @test (@allocated f(x, y, lg)) == 0      # allocations
        @test_opt f(x, y, lg)                    # type instabilities

        @test result_type == typeof(result)      # inferred type
    end
end

@testset "mul_with_projection Unit Tests" begin
    Random.seed!(1) # ensures reproducibility

    FT = Float64
    coord = Geometry.LatLongZPoint(rand(FT), rand(FT), rand(FT))
    ∂x∂ξ = Geometry.Tensor(
        (@SMatrix rand(FT, 3, 3)),
        (
            Geometry.Components{Geometry.Orthonormal, (1, 2, 3)}(),
            Geometry.Components{Geometry.Covariant, (1, 2, 3)}(),
        ),
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
    test_mul_with_projection(number, number, lg, number * number)
    test_mul_with_projection(number, vector, lg, number * vector)
    test_mul_with_projection(number, tensor, lg, number * tensor)
    test_mul_with_projection(number, covector, lg, number * covector)
    test_mul_with_projection(number, cotensor, lg, number * cotensor)
    test_mul_with_projection(vector, number, lg, vector * number)
    test_mul_with_projection(vector, covector, lg, vector * covector)
    test_mul_with_projection(tensor, number, lg, tensor * number)
    test_mul_with_projection(tensor, vector, lg, tensor * projected_vector)
    test_mul_with_projection(tensor, tensor, lg, tensor * projected_tensor)
    test_mul_with_projection(tensor, cotensor, lg, tensor * cotensor)
    test_mul_with_projection(covector, number, lg, covector * number)
    test_mul_with_projection(covector, vector, lg, covector * projected_vector)
    test_mul_with_projection(covector, tensor, lg, covector * projected_tensor)
    test_mul_with_projection(covector, cotensor, lg, covector * cotensor)
    test_mul_with_projection(cotensor, number, lg, cotensor * number)
    test_mul_with_projection(cotensor, vector, lg, cotensor * projected_vector)
    test_mul_with_projection(cotensor, tensor, lg, cotensor * projected_tensor)
    test_mul_with_projection(cotensor, cotensor, lg, cotensor * cotensor)

    # Test some combinations of complicated nested values.
    T = add_auto_broadcasters ∘ nested_type
    test_mul_with_projection(
        number,
        T(covector, vector, tensor),
        lg,
        T(number * covector, number * vector, number * tensor),
    )
    test_mul_with_projection(
        T(covector, vector, tensor),
        number,
        lg,
        T(covector * number, vector * number, tensor * number),
    )
    test_mul_with_projection(
        vector,
        T(number, number, number),
        lg,
        T(vector * number, vector * number, vector * number),
    )
    test_mul_with_projection(
        T(number, number, number),
        covector,
        lg,
        T(number * covector, number * covector, number * covector),
    )
    test_mul_with_projection(
        T(number, vector, number),
        T(covector, number, tensor),
        lg,
        T(number * covector, vector * number, number * tensor),
    )
    test_mul_with_projection(
        T(covector, number, tensor),
        T(number, vector, number),
        lg,
        T(covector * number, number * vector, tensor * number),
    )
    test_mul_with_projection(
        covector,
        T(vector, number, tensor),
        lg,
        T(
            covector * projected_vector,
            covector * number,
            covector * projected_tensor,
        ),
    )
    test_mul_with_projection(
        T(covector, number, covector),
        vector,
        lg,
        T(
            covector * projected_vector,
            number * vector,
            covector * projected_vector,
        ),
    )
    test_mul_with_projection(
        T(covector, number, covector),
        T(number, vector, tensor),
        lg,
        T(covector * number, number * vector, covector * projected_tensor),
    )
end

@testset "Products with Dual numbers promote the storage type" begin
    import ForwardDiff
    for FT in (Float32, Float64)
        D = ForwardDiff.Dual{Nothing, FT, 2}
        d = ForwardDiff.Dual{Nothing}(FT(2), FT(1), FT(0))
        tensors = (
            Geometry.Covariant3Vector(FT(1)),
            Geometry.Covariant12Vector(FT(1), FT(2)),
            Geometry.UVWVector(FT(1), FT(2), FT(3)),
            Geometry.Covariant3Vector(FT(1))',                     # covector
            Geometry.UVWVector(FT(1), FT(2), FT(3)) *
            Geometry.Covariant123Vector(FT(1), FT(2), FT(3))',     # 2-tensor
        )
        for x in tensors
            X = typeof(x)
            # A `Dual` scalar times a `Float` tensor is a `Dual` tensor whose storage is
            # `Dual` as well, and the inferred type is the type `*` actually produces.
            @test return_type(*, Tuple{D, X}) == typeof(d * x)
            @test return_type(*, Tuple{X, D}) == typeof(x * d)
            @test eltype(parent(d * x)) == D
            # No promotion, no change: the existing result types are untouched.
            @test return_type(*, Tuple{FT, X}) == X
            @test return_type(*, Tuple{X, FT}) == X
        end
    end
end
