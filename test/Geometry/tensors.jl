using Test, JET
using ClimaCore.Geometry, ClimaCore.DataLayouts
using LinearAlgebra, StaticArrays
import ClimaCore

@testset "Tensors" begin
    x = Geometry.Covariant12Vector(1.0, 2.0)
    y = Geometry.Contravariant12Vector(1.0, 4.0)

    @test x.u₁ === 1.0
    @test x.u₂ === 2.0
    @test x.u₃ === 0.0

    f(x) = x.u₁ + x.u₂ + x.u₃
    @test_opt f(x)

    ref = Ref(zero(x))
    ref[] = Geometry.Covariant12Vector(1, 2) # Int components instead of Float64
    @test ref[] == x

    M = Geometry.Tensor(
        [1.0 0.0; 0.5 2.0],
        (Geometry.UVAxis(), Geometry.Covariant12Axis()),
    )

    @test dot(x, y) == x' * y == 9.0
    @test dot(y, x) == y' * x == 9.0

    @test x == x
    @test x != parent(x)

    @test x[1] == 1.0
    @test y[2] == 4.0
    @test M[2] == 0.5
    @test M[2, 1] == 0.5
    @test M[:, 1] == Geometry.UVVector(1.0, 0.5)
    @test M[1, :] == Geometry.Covariant12Vector(1.0, 0.0)

    @test x + zero(x) == x
    @test x' + zero(x') == x'

    @test -x + x * 2 - x / 2 == -x + 2 * x - 2 \ x == x / 2
    @test -x' + x' * 2 - x' / 2 == -x' + 2 * x' - 2 \ x' == (x / 2)'

    @test x * 3 == x ⊗ 3 == Geometry.Covariant12Vector(3.0, 6.0)
    @test x * y' ==
          x ⊗ y ==
          Geometry.Tensor(
              parent(x) * parent(y)',
              (axes(x, 1), axes(y, 1)),
          )

    @test parent(M * inv(M)) == @SMatrix [1.0 0.0; 0.0 1.0]
    @test parent(inv(M) * M) == @SMatrix [1.0 0.0; 0.0 1.0]

    @test M * y == Geometry.UVVector(1.0, 8.5)
    @test M \ Geometry.UVVector(1.0, 8.5) == y

    @test_throws Geometry.NoMetricError dot(x, x)
    @test_throws Geometry.NoMetricError M * x
    @test_throws Geometry.NoMetricError M \ x
    @test sprint(showerror, Geometry.NoMetricError{Int, Float64}()) ==
          "Metric is needed for change of basis: Int64 vs Float64"

    @test DataLayouts.num_basetypes(Float64, typeof(x)) == 2
end

@testset "Printing" begin
    # https://github.com/CliMA/ClimaCore.jl/issues/768
    T = Geometry.Tensor{
        2,
        Float64,
        Tuple{
            Geometry.Components{Geometry.Orthonormal, (1, 2)},
            Geometry.Components{Geometry.Covariant, (1, 2)},
        },
        SMatrix{2, 2, Float64, 4},
    }
    components = SMatrix{2, 2, Float64, 4}([4.0 0.0; 0.0 5.0])
    bases = (
        Geometry.Components{Geometry.Orthonormal, (1, 2)}(),
        Geometry.Components{Geometry.Covariant, (1, 2)}(),
    )
    ats = T(components, bases)
    s = sprint(show, ats)
    s = replace(s, "StaticArraysCore." => "")
    s = replace(s, "ClimaCore.Geometry." => "")
    if !Sys.iswindows()
        @test occursin("Tensor(", s)
        @test occursin("Orthonormal", s)
        @test occursin("Covariant", s)
    end
end

@testset "transform" begin
    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant1Vector(2.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)
    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)

    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant1Vector(2.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'
    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'
end

@testset "project" begin
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant1Vector(2.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)

    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant12Vector(2.0, 2.0),
    ) == Geometry.Covariant12Vector(2.0, 2.0)

    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant123Vector(2.0, 2.0, 0.0),
    ) == Geometry.Covariant12Vector(2.0, 2.0)

    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant123Vector(2.0, 2.0, 1.0),
    ) == Geometry.Covariant12Vector(2.0, 2.0)


    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 2.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)

    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant1Vector(2.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 2.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'

    # Test projection over rightmost axis
    x_C12 = Geometry.Covariant12Vector(2.0, 2.0)
    x_Cart123 = Geometry.UVWVector(1.0, 1.0, 1.0)
    @test Geometry.project(x_C12 * x_Cart123', Geometry.WAxis()) ==
          x_C12 * Geometry.WVector(1.0)'
    @test Geometry.project(x_C12 * x_Cart123', Geometry.VWAxis()) ==
          x_C12 * Geometry.VWVector(1.0, 1.0)'

    # Test projection over both axes
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        x_C12 * x_Cart123',
        Geometry.UVWAxis(),
    ) == x_C12 * x_Cart123'
    @test Geometry.project(
        Geometry.Covariant2Axis(),
        x_C12 * x_Cart123',
        Geometry.UWAxis(),
    ) == Geometry.Covariant2Vector(2.0) * Geometry.UWVector(1.0, 1.0)'
end


@testset "cross product" begin
    M = @SMatrix [
        4.0 1.0
        0.5 2.0
    ]
    J = det(M)
    local_geom = Geometry.LocalGeometry(
        Geometry.XYPoint(0.0, 0.0),
        J,
        J,
        Geometry.Tensor(M, (Geometry.UVAxis(), Geometry.Covariant12Axis())),
    )

    u = Geometry.UVVector(1.0, 2.0)
    v = Geometry.WVector(3.0)
    @test u × v == -v × u == Geometry.UVVector(6.0, -3.0)
    uⁱ = Geometry.ContravariantVector(u, local_geom)
    vⁱ = Geometry.ContravariantVector(v, local_geom)
    @test Geometry.UVVector(Geometry._cross(uⁱ, vⁱ, local_geom), local_geom) ≈
          Geometry.UVVector(6.0, -3.0)
end


@testset "project" begin
    M = @SMatrix [
        2.0 0.0
        0.0 1.0
    ]
    J = det(M)

    local_geom = Geometry.LocalGeometry(
        Geometry.XYPoint(0.0, 0.0),
        J,
        J,
        Geometry.Tensor(M, (Geometry.UVAxis(), Geometry.Covariant12Axis())),
    )

    @test Geometry.project(
        Geometry.Contravariant12Axis(),
        Covariant12Vector(1.0, 1.0),
        local_geom,
    ) == Contravariant12Vector(0.25, 1.0)
    @test Geometry.project(
        Geometry.Contravariant1Axis(),
        Covariant12Vector(1.0, 1.0),
        local_geom,
    ) == Contravariant1Vector(0.25)
    @test Geometry.project(
        Geometry.Contravariant2Axis(),
        Covariant12Vector(1.0, 1.0),
        local_geom,
    ) == Contravariant2Vector(1.0)
    @test Geometry.project(
        Geometry.Contravariant123Axis(),
        Covariant12Vector(1.0, 1.0),
        local_geom,
    ) == Contravariant123Vector(0.25, 1.0, 0.0)
    @test Geometry.project(
        Geometry.Contravariant123Axis(),
        Covariant123Vector(1.0, 1.0, 1.0),
        local_geom,
    ) == Contravariant123Vector(0.25, 1.0, 1.0)


    @test Geometry.project(
        Geometry.Contravariant12Axis(),
        Covariant12Vector(1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant12Vector(0.25, 1.0) ⊗ Covariant12Vector(2.0, 8.0)
    @test Geometry.project(
        Geometry.Contravariant1Axis(),
        Covariant12Vector(1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant1Vector(0.25) ⊗ Covariant12Vector(2.0, 8.0)
    @test Geometry.project(
        Geometry.Contravariant2Axis(),
        Covariant12Vector(1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant2Vector(1.0) ⊗ Covariant12Vector(2.0, 8.0)
    @test Geometry.project(
        Geometry.Contravariant123Axis(),
        Covariant12Vector(1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant123Vector(0.25, 1.0, 0.0) ⊗ Covariant12Vector(2.0, 8.0)
    @test Geometry.project(
        Geometry.Contravariant123Axis(),
        Covariant123Vector(1.0, 1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant123Vector(0.25, 1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0)
end

# Components carries its component names as a type parameter, and
# unrolled_allunique, unrolled_unique, unrolled_filter, and unrolled_findfirst
# all determine the *values* of those names, so the types of their results
# depend on the values of their inputs. Basis computations are compiled for
# GPUs, where nothing can be inferred from a run-time value, so these calls are
# only valid because the names reach them through the type domain and fold. A
# folding failure would leave the number of selected names unknown, which shows
# up as an abstract return type and a heap allocation here, and as an
# InvalidIRError when a kernel that computes a basis is compiled.
const COVARIANT = Geometry.Covariant()
const B12 = Geometry.Components(COVARIANT, (1, 2))
const B23 = Geometry.Components(COVARIANT, (2, 3))
const B123 = Geometry.Components(COVARIANT, (1, 2, 3))
const B2 = Geometry.Components(COVARIANT, (2,))
const V12 = Geometry.Covariant12Vector(1.0, 2.0)
const V23 = Geometry.Covariant23Vector(2.0, 3.0)

# Every function below takes its bases as arguments, as a kernel would, so that
# the names are only available to inference through the argument types.
combine_two(b1, b2) = Geometry.combine_components(b1, b2)
combine_three(b1, b2, b3) = Geometry.combine_components(b1, b2, b3)
overlap_two(b1, b2) = Geometry.overlap_components(b1, b2)
matching_indices(b1, b2) = Geometry.matching_component_indices(b1, b2)
components_of(b) =
    Geometry.Components(Geometry.components_type(b), Geometry.component_names(b))
components_from_names(type, names) = Geometry.Components(type, names)
widen_basis(v) = reshape(v, (Geometry.Covariant123Axis(),))
# The result is discarded because returning a tensor that does not fit in the
# argument registers allocates a boxed copy of it however it is computed.
discard_result(f::F, args...) where {F} = (f(args...); nothing)

@testset "component names fold into Components type parameters" begin
    @testset "unrolled_unique through combine_components" begin
        @test @inferred(combine_two(B12, B23)) === B123
        @test @inferred(combine_three(B12, B23, B2)) === B123
        @test_opt combine_two(B12, B23)
        @test (@allocated combine_two(B12, B23)) == 0
    end

    @testset "unrolled_filter and unrolled_in through overlap_components" begin
        @test @inferred(overlap_two(B12, B23)) === B2
        @test @inferred(overlap_two(B123, B23)) === B23
        @test_opt overlap_two(B12, B23)
        @test (@allocated overlap_two(B12, B23)) == 0
    end

    @testset "unrolled_findfirst through matching_component_indices" begin
        # The element types depend on which names match, so an unfolded search
        # cannot produce a concrete Tuple type.
        @test @inferred(matching_indices(B123, B12)) === (1, 2, nothing)
        @test isconcretetype(Base.promote_op(matching_indices, typeof(B123), typeof(B12)))
        @test_opt matching_indices(B123, B12)
        @test (@allocated matching_indices(B123, B12)) == 0
    end

    @testset "unrolled_allunique through the Components constructor" begin
        @test @inferred(components_of(B123)) === B123
        @test_throws Geometry.DuplicateComponentNamesError Geometry.Components(
            COVARIANT,
            (1, 2, 2),
        )
        # Names that are only a run-time value cannot fold, because they have to
        # become a type parameter. This is a property of Components itself, and
        # it is asserted so that a caller passing non-constant names is caught
        # here rather than by a GPU compilation failure.
        @test !isconcretetype(
            Base.promote_op(components_from_names, typeof(COVARIANT), Tuple{Int, Int}),
        )
    end

    @testset "tensor operations that combine and widen bases" begin
        # Adding vectors with different bases combines their names, and
        # reshaping to a wider basis zero-fills the names it is missing.
        @test @inferred(V12 + V23) === Geometry.Covariant123Vector(1.0, 4.0, 3.0)
        @test @inferred(widen_basis(V12)) === Geometry.Covariant123Vector(1.0, 2.0, 0.0)
        @test_opt V12 + V23
        @test_opt widen_basis(V12)
        discard_result(+, V12, V23)
        @test (@allocated discard_result(+, V12, V23)) == 0
        discard_result(widen_basis, V12)
        @test (@allocated discard_result(widen_basis, V12)) == 0
    end
end
