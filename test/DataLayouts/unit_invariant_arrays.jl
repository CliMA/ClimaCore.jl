#=
julia --project=.buildkite
using Revise; include(joinpath("test", "DataLayouts", "unit_invariant_arrays.jl"))
=#
using Test
import Adapt
import ClimaComms
ClimaComms.@import_required_backends
import LinearAlgebra: I, norm
import ClimaCore: CommonSpaces, DataLayouts, Fields, MatrixFields, Spaces
import ClimaCore.MatrixFields: @name, DiagonalMatrixRow

test_space() = CommonSpaces.ExtrudedCubedSphereSpace(
    Float32;
    z_elem = 4,
    z_min = 0,
    z_max = 1,
    radius = 10,
    h_elem = 2,
    n_quad_points = 4,
    staggering = CommonSpaces.CellCenter(),
)
is_invariant(field) = parent(field) isa DataLayouts.AnyInvariantArray

@testset "InvariantArray" begin
    array = rand(Float32, 3, 4)
    invariant = DataLayouts.InvariantArray(array)
    @test invariant == array
    @test invariant[2, 3] === array[2, 3]
    @test_throws ArgumentError("Cannot modify InvariantArray") (invariant[2, 3] = 0)
    @test_throws ArgumentError (view(invariant, :, 1) .= 0)
    @test similar(invariant) isa Matrix{Float32}
    @test copy(invariant) isa Matrix{Float32}
    @test Adapt.adapt(Array, invariant) isa DataLayouts.InvariantArray
    @test DataLayouts.DataScope(invariant) == DataLayouts.DataScope(array)
end

@testset "copies, comparisons, reductions, and broadcasts use the wrapped arrays" begin
    array = rand(Float32, 4, 4, 3, 2, 5)
    invariant = DataLayouts.InvariantArray(array)
    wrappers = (
        (invariant, array),
        (view(invariant, :, :, :, 1:1, :), view(array, :, :, :, 1:1, :)),
        (
            reshape(view(invariant, :, :, :, 1:1, :), :, 5),
            reshape(view(array, :, :, :, 1:1, :), :, 5),
        ),
    )
    for (wrapped, unwrapped) in wrappers
        @test DataLayouts.unwrap_invariant(wrapped) == unwrapped
        @test Array(wrapped) isa Array && Array(wrapped) == unwrapped
        @test collect(wrapped) == unwrapped
        @test !(copy(wrapped) isa DataLayouts.InvariantArray) &&
              copy(wrapped) == unwrapped
        dest = similar(Array(unwrapped))
        @test copyto!(dest, wrapped) == unwrapped
        dest_view = view(similar(dest), axes(dest)...)
        @test copyto!(dest_view, wrapped) == unwrapped
        @test which(copyto!, Tuple{typeof(dest_view), typeof(wrapped)}).module ===
              DataLayouts
        @test wrapped == wrapped && isequal(wrapped, wrapped)
        @test wrapped == unwrapped && isequal(wrapped, unwrapped)
        @test unwrapped == wrapped && isequal(unwrapped, wrapped)
        for types in (
            (typeof(wrapped), typeof(wrapped)),
            (typeof(wrapped), typeof(unwrapped)),
            (typeof(unwrapped), typeof(wrapped)),
        )
            @test which(==, types).module === DataLayouts
            @test which(isequal, types).module === DataLayouts
        end
        @test maximum(wrapped) == maximum(unwrapped)
        @test norm(wrapped) == norm(unwrapped) && norm(wrapped, 1) == norm(unwrapped, 1)
        @test isapprox(wrapped, wrapped) && isapprox(wrapped, unwrapped)
        @test any(>(0.5), wrapped) == any(>(0.5), unwrapped)
        @test all(>(0.5), wrapped) == all(>(0.5), unwrapped)
        @test count(>(0.5), wrapped) == count(>(0.5), unwrapped)
        for (f, args) in
            ((norm, (wrapped,)), (any, (isnan, wrapped)), (all, (isnan, wrapped)),
            (count, (isnan, wrapped)))
            @test which(f, typeof(args)).module === DataLayouts
        end
        @test Base.Broadcast.BroadcastStyle(typeof(wrapped)) ==
              Base.Broadcast.BroadcastStyle(typeof(unwrapped))
        @test wrapped .+ 1 == unwrapped .+ 1
    end
end

@testset "grids store their local geometry in InvariantArrays" begin
    space = test_space()
    @test parent(Spaces.local_geometry_data(space)) isa DataLayouts.InvariantArray
    local_geometry = Fields.local_geometry_field(space)
    z = Fields.coordinate_field(space).z
    # The geometry and its deep copies cannot be modified, but copies of it, similar
    # fields, and the results of broadcasts over it can.
    @test_throws ArgumentError (parent(z) .= 0)
    @test is_invariant(deepcopy(z))
    @test_throws ArgumentError (parent(deepcopy(z)) .= 0)
    for field in (local_geometry, local_geometry.J, z)
        @test is_invariant(field)
        for new_field in (copy(field), similar(field), identity.(field))
            @test !is_invariant(new_field)
            @test axes(new_field) === axes(field)
        end
        @test copy(field) == field
    end
    z_copy = copy(z)
    z_copy .= 0
    @test maximum(abs, z_copy) == 0
end

@testset "invariant_data of fields, field vectors, and field matrices" begin
    space = test_space()
    field = Fields.coordinate_field(space).z .+ 1
    @test !is_invariant(field)
    invariant_field = DataLayouts.invariant_data(field)
    @test is_invariant(invariant_field)
    @test axes(invariant_field) === axes(field)
    @test invariant_field == field
    @test_throws ArgumentError (parent(invariant_field) .= 0)
    # InvariantArrays are not wrapped again
    @test parent(DataLayouts.invariant_data(invariant_field)) === parent(invariant_field)

    vector = Fields.FieldVector(; a = field, b = (; c = 2 .* field))
    invariant_vector = DataLayouts.invariant_data(vector)
    @test is_invariant(invariant_vector.a) && is_invariant(invariant_vector.b.c)
    @test invariant_vector == vector
    @test !is_invariant(copy(invariant_vector).a)

    matrix = MatrixFields.FieldMatrix(
        (@name(a), @name(a)) => I,
        (@name(a), @name(b.c)) => DiagonalMatrixRow.(field),
    )
    invariant_matrix = DataLayouts.invariant_data(matrix)
    @test keys(invariant_matrix) == keys(matrix)
    @test invariant_matrix[(@name(a), @name(a))] === I
    @test is_invariant(invariant_matrix[(@name(a), @name(b.c))])
    @test invariant_matrix[(@name(a), @name(b.c))] == matrix[(@name(a), @name(b.c))]
end
