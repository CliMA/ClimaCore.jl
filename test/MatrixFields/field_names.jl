import LinearAlgebra: I
import ClimaCore.DataLayouts: replace_basetype
import ClimaCore.MatrixFields: @name, is_subset_that_covers_set

include("matrix_field_test_utils.jl")

struct FooFieldName{T}
    _value::T
end
Base.propertynames(::FooFieldName) = (:value,)
Base.getproperty(foo::FooFieldName, s::Symbol) =
    s == :value ? getfield(foo, :_value) : error("Invalid property name")
Base.convert(::Type{FooFieldName{T}}, foo::FooFieldName) where {T} =
    FooFieldName{T}(foo.value)
Base.zero(::FooFieldName{T}) where {T} = FooFieldName(zero(T))

get_x() =
    (; foo = FooFieldName(0), a = (; b = 1, c = ((; d = 2), (;), (3, ()))))

@testset "FieldName Unit Tests" begin
    x = get_x()
    @test_all @name() == MatrixFields.FieldName()
    @test_all @name(a.c.:(1).d) == MatrixFields.FieldName(:a, :c, 1, :d)
    @test_all @name(a.c.:(3).:(1)) == MatrixFields.FieldName(:a, :c, 3, 1)

    i = 1
    @test @name(a.c.:($i).d) == @name(a.c.:(1).d)
    @test @name(a.c.:($(i + 2)).:($i)) == @name(a.c.:(3).:(1))

    @test_throws "invalid syntax" @macroexpand @name("a")
    @test_throws "invalid syntax" @macroexpand @name([a])
    @test_throws "invalid syntax" @macroexpand @name((a.c.:3.0):1)
    @test_throws "invalid syntax" @macroexpand @name(a.c.:(3).(1))

    @test string(@name()) == "@name()"
    @test string(@name(a.c.:(1).d)) == "@name(a.c.:(1).d)"
    @test string(@name(a.c.:(3).:(1))) == "@name(a.c.:(3).:(1))"

    @test_all MatrixFields.has_field(x, @name())
    @test_all MatrixFields.has_field(x, @name(foo.value))
    @test_all MatrixFields.has_field(x, @name(a.b))
    @test_all MatrixFields.has_field(x, @name(a.c.:(1).d))
    @test_all !MatrixFields.has_field(x, @name(foo.invalid_name))

    @test_all MatrixFields.get_field(x, @name()) == x
    @test_all MatrixFields.get_field(x, @name(foo.value)) == 0
    @test_all MatrixFields.get_field(x, @name(a.b)) == 1
    @test_all MatrixFields.get_field(x, @name(a.c.:(1).d)) == 2

    @test_all MatrixFields.broadcasted_has_field(typeof(x), @name())
    @test_all MatrixFields.broadcasted_has_field(typeof(x), @name(foo._value))
    @test_all MatrixFields.broadcasted_has_field(typeof(x), @name(a.b))
    @test_all MatrixFields.broadcasted_has_field(typeof(x), @name(a.c.:(1).d))
    @test_all !MatrixFields.broadcasted_has_field(
        typeof(x),
        @name(foo.invalid_name)
    )

    @test_all MatrixFields.broadcasted_get_field(x, @name()) == x
    @test_all MatrixFields.broadcasted_get_field(x, @name(foo._value)) == 0
    @test_all MatrixFields.broadcasted_get_field(x, @name(a.b)) == 1
    @test_all MatrixFields.broadcasted_get_field(x, @name(a.c.:(1).d)) == 2

    @test_all MatrixFields.is_child_name(@name(a.c.:(1).d), @name(a))
    @test_all !MatrixFields.is_child_name(@name(a.c.:(1).d), @name(foo))

    @test_all MatrixFields.is_overlapping_name(@name(a), @name(a.c.:(1).d))
    @test_all MatrixFields.is_overlapping_name(@name(a.c.:(1).d), @name(a))
    @test_all !MatrixFields.is_overlapping_name(@name(foo), @name(a.c.:(1).d))
    @test_all !MatrixFields.is_overlapping_name(@name(a.c.:(1).d), @name(foo))

    @test_all MatrixFields.extract_internal_name(@name(a.c.:(1).d), @name(a)) ==
              @name(c.:(1).d)
    @test_throws "is not a child name" MatrixFields.extract_internal_name(
        @name(a.c.:(1).d),
        @name(foo)
    )

    @test_all MatrixFields.append_internal_name(@name(a), @name(c.:(1).d)) ==
              @name(a.c.:(1).d)

    @test_all MatrixFields.top_level_names(x) == (@name(foo), @name(a))
    @test_all MatrixFields.top_level_names(x.foo) == (@name(value),)
    @test_all MatrixFields.top_level_names(x.a) == (@name(b), @name(c))
    @test_all MatrixFields.top_level_names(x.a.c) ==
              (@name(1), @name(2), @name(3))
end

@testset "FieldNameTree Unit Tests" begin
    x = get_x()
    name_tree = MatrixFields.FieldNameTree(x)

    @test_all MatrixFields.FieldNameTree(x) == name_tree

    @test_all MatrixFields.is_valid_name(@name(), name_tree)
    @test_all MatrixFields.is_valid_name(@name(foo.value), name_tree)
    @test_all MatrixFields.is_valid_name(@name(a.b), name_tree)
    @test_all MatrixFields.is_valid_name(@name(a.c.:(1).d), name_tree)
    @test_all !MatrixFields.is_valid_name(@name(foo.invalid_name), name_tree)

    @test_all MatrixFields.child_names(@name(), name_tree) ==
              (@name(foo), @name(a))
    @test_all MatrixFields.child_names(@name(foo), name_tree) ==
              (@name(foo.value),)
    @test_all MatrixFields.child_names(@name(a), name_tree) ==
              (@name(a.b), @name(a.c))
    @test_all MatrixFields.child_names(@name(a.c), name_tree) ==
              (@name(a.c.:(1)), @name(a.c.:(2)), @name(a.c.:(3)))
    @test_throws "does not have child names" MatrixFields.child_names(
        @name(a.c.:(2)),
        name_tree,
    )
    @test_throws "is not a valid name" MatrixFields.child_names(
        @name(foo.invalid_name),
        name_tree,
    )
end

@testset "FieldNameSet Unit Tests" begin
    x = get_x()
    name_tree = MatrixFields.FieldNameTree(x)

    vector_keys(names...) = MatrixFields.FieldVectorKeys(names, name_tree)
    matrix_keys(name_pairs...) =
        MatrixFields.FieldMatrixKeys(name_pairs, name_tree)

    vector_keys_no_tree(names...) = MatrixFields.FieldVectorKeys(names)
    matrix_keys_no_tree(name_pairs...) =
        MatrixFields.FieldMatrixKeys(name_pairs)

    drop_tree(set) =
        set isa MatrixFields.FieldVectorKeys ?
        MatrixFields.FieldVectorKeys(set.values) :
        MatrixFields.FieldMatrixKeys(set.values)

    @testset "FieldNameSet Constructors" begin
        @test_throws "Invalid FieldNameSet value" vector_keys(
            @name(invalid_name)
        )
        @test_throws "Invalid FieldNameSet value" matrix_keys((
            @name(invalid_name),
            @name(a.c)
        ),)

        for constructor in (vector_keys, vector_keys_no_tree)
            @test_throws "Duplicate FieldNameSet values" constructor(
                @name(foo),
                @name(foo)
            )
            @test_throws "Overlapping FieldNameSet values" constructor(
                @name(foo),
                @name(foo.value)
            )
        end
        for constructor in (matrix_keys, matrix_keys_no_tree)
            @test_throws "Duplicate FieldNameSet values" constructor(
                (@name(foo.value), @name(a.c)),
                (@name(foo.value), @name(a.c)),
            )
            @test_throws "Overlapping FieldNameSet values" constructor(
                (@name(foo), @name(a.c)),
                (@name(foo.value), @name(a.c)),
            )
        end
    end

    v_set1 = vector_keys(@name(foo), @name(a.c))
    m_set1 = matrix_keys((@name(foo), @name(a.c)), (@name(a.b), @name(foo)))

    # Proper subsets of v_set1 and m_set1.
    v_set2 = vector_keys(@name(foo))
    m_set2 = matrix_keys((@name(foo), @name(a.c)))

    # Subsets of v_set1 and m_set1 that cover those sets.
    v_set3 = vector_keys(
        @name(foo.value),
        @name(a.c.:(1)),
        @name(a.c.:(2)),
        @name(a.c.:(3))
    )
    m_set3 = matrix_keys(
        (@name(foo.value), @name(a.c.:(1))),
        (@name(foo), @name(a.c.:(2))),
        (@name(foo), @name(a.c.:(3))),
        (@name(a.b), @name(foo.value)),
    )

    # Sets that overlap with v_set1 and m_set1, but are neither subsets nor
    # supersets of those sets. Some of the values in m_set4 overlap with values
    # in m_set1, but they are neither children nor parents of those values (this
    # is only possible with matrix keys).
    v_set4 = vector_keys(@name(a.b), @name(a.c.:(1)), @name(a.c.:(2)))
    m_set4 = matrix_keys(
        (@name(), @name(a.c.:(1))),
        (@name(foo.value), @name(foo)),
        (@name(foo.value), @name(a.c.:(2))),
        (@name(a), @name(foo.value)),
        (@name(a.c.:(3)), @name(a.c.:(3))),
    )

    @testset "FieldNameSet Basic Operations" begin
        # We need to use endswith instead of == in the following tests to
        # account for module qualifiers that may or may not get printed,
        # depending on how these tests are run.

        @test endswith(
            string(v_set1),
            "FieldVectorKeys(@name(foo), @name(a.c); <FieldNameTree>)",
        )
        @test endswith(
            string(drop_tree(v_set1)),
            "FieldVectorKeys(@name(foo), @name(a.c))",
        )
        @test endswith(
            string(m_set1),
            "FieldMatrixKeys((@name(foo), @name(a.c)), \
             (@name(a.b), @name(foo)); <FieldNameTree>)",
        )
        @test endswith(
            string(drop_tree(m_set1)),
            "FieldMatrixKeys((@name(foo), @name(a.c)), \
             (@name(a.b), @name(foo)))",
        )

        @test_all map(name -> (name, name), v_set1) ==
                  ((@name(foo), @name(foo)), (@name(a.c), @name(a.c)))
        @test_all map(name_pair -> name_pair[1], m_set1) ==
                  (@name(foo), @name(a.b))

        @test_all isnothing(foreach(name -> (name, name), v_set1))
        @test_all isnothing(foreach(name_pair -> name_pair[1], m_set1))

        for set1 in (v_set1, drop_tree(v_set1))
            @test_all @name(foo) in set1
            @test_all @name(foo.value) in set1
            @test_all !(@name(a.b) in set1)
            @test_all !(@name(invalid_name) in set1)
        end
        for set1 in (m_set1, drop_tree(m_set1))
            @test_all (@name(foo), @name(a.c)) in set1
            @test_all (@name(foo.value), @name(a.c)) in set1
            @test_all !((@name(foo), @name(a.b)) in set1)
            @test_all !((@name(foo), @name(invalid_name)) in set1)
        end

        @test_all !(@name(foo.invalid_name) in v_set1)
        @test_all @name(foo.invalid_name) in drop_tree(v_set1)
        @test_all !((@name(foo.invalid_name), @name(a.c)) in m_set1)
        @test_all (@name(foo.invalid_name), @name(a.c)) in drop_tree(m_set1)
    end

    @testset "FieldNameSet Complement Sets" begin
        @test_all MatrixFields.set_complement(v_set1) ==
                  vector_keys_no_tree(@name(a.b))
        @test_all MatrixFields.set_complement(v_set2) ==
                  vector_keys_no_tree(@name(a))
        @test_all MatrixFields.set_complement(v_set3) ==
                  vector_keys_no_tree(@name(a.b))
        @test_all MatrixFields.set_complement(v_set4) ==
                  vector_keys_no_tree(@name(foo), @name(a.c.:(3)))
        @test_throws "FieldNameTree" MatrixFields.set_complement(
            drop_tree(v_set1),
        )

        @test_all MatrixFields.set_complement(m_set1) == matrix_keys_no_tree(
            (@name(foo), @name(foo)),
            (@name(foo), @name(a.b)),
            (@name(a), @name(a)),
            (@name(a.c), @name(foo)),
        )
        @test_all MatrixFields.set_complement(m_set2) == matrix_keys_no_tree(
            (@name(foo), @name(foo)),
            (@name(foo), @name(a.b)),
            (@name(a), @name(foo)),
            (@name(a), @name(a)),
        )
        @test_all MatrixFields.set_complement(m_set3) == matrix_keys_no_tree(
            (@name(foo), @name(foo)),
            (@name(foo), @name(a.b)),
            (@name(a), @name(a)),
            (@name(a.c), @name(foo)),
        )
        @test_all MatrixFields.set_complement(m_set4) == matrix_keys_no_tree(
            (@name(foo), @name(a.b)),
            (@name(foo), @name(a.c.:(3))),
            (@name(a), @name(a.b)),
            (@name(a), @name(a.c.:(2))),
            (@name(a.b), @name(a.c.:(3))),
            (@name(a.c.:(1)), @name(a.c.:(3))),
            (@name(a.c.:(2)), @name(a.c.:(3))),
        )
        @test_throws "FieldNameTree" MatrixFields.set_complement(
            drop_tree(m_set1),
        )
    end

    @testset "FieldNameSet Binary Set Operations" begin
        for set1 in (v_set1, drop_tree(v_set1), m_set1, drop_tree(m_set1))
            @test_all set1 == set1
            @test_all issubset(set1, set1)
            @test_all is_subset_that_covers_set(set1, set1)
            @test_all intersect(set1, set1) == set1
            @test_all union(set1, set1) == set1
            @test_all isempty(setdiff(set1, set1))
        end

        for (set1, set2) in (
            (v_set1, v_set2),
            (v_set1, drop_tree(v_set2)),
            (drop_tree(v_set1), v_set2),
            (drop_tree(v_set1), drop_tree(v_set2)),
            (m_set1, m_set2),
            (m_set1, drop_tree(m_set2)),
            (drop_tree(m_set1), m_set2),
            (drop_tree(m_set1), drop_tree(m_set2)),
        )
            @test_all set1 != set2 && set2 != set1
            @test_all !issubset(set1, set2) && issubset(set2, set1)
            @test_all !is_subset_that_covers_set(set1, set2) &&
                      !is_subset_that_covers_set(set2, set1)
            @test_all intersect(set1, set2) == intersect(set2, set1) == set2
            @test_all union(set1, set2) == union(set2, set1) == set1
            if set1 isa MatrixFields.FieldVectorKeys
                @test_all setdiff(set1, set2) == vector_keys(@name(a.c))
            else
                @test_all setdiff(set1, set2) ==
                          matrix_keys((@name(a.b), @name(foo)))
            end
            @test_all isempty(setdiff(set2, set1))
        end

        for (set1, set3) in (
            (v_set1, v_set3),
            (v_set1, drop_tree(v_set3)),
            (drop_tree(v_set1), v_set3),
            (m_set1, m_set3),
            (m_set1, drop_tree(m_set3)),
            (drop_tree(m_set1), m_set3),
        )
            @test_all set1 != set3 && set3 != set1
            @test_all !issubset(set1, set3) && issubset(set3, set1)
            @test_all !is_subset_that_covers_set(set1, set3) &&
                      is_subset_that_covers_set(set3, set1)
            @test_all intersect(set1, set3) == intersect(set3, set1) == set3
            @test_all union(set1, set3) == union(set3, set1) == set3
            @test_all isempty(setdiff(set1, set3)) &&
                      isempty(setdiff(set3, set1))
        end

        for (set1, set3) in (
            (drop_tree(v_set1), drop_tree(v_set3)),
            (drop_tree(m_set1), drop_tree(m_set3)),
        )
            @test_all set1 != set3 && set3 != set1
            @test_all !issubset(set1, set3) && issubset(set3, set1)
            @test_all !is_subset_that_covers_set(set1, set3)
            @test_throws "FieldNameTree" is_subset_that_covers_set(set3, set1)
            @test_throws "FieldNameTree" intersect(set1, set3)
            @test_throws "FieldNameTree" intersect(set3, set1)
            @test_throws "FieldNameTree" union(set1, set3)
            @test_throws "FieldNameTree" union(set3, set1)
            @test_throws "FieldNameTree" setdiff(set1, set3)
            @test_throws "FieldNameTree" setdiff(set3, set1)
        end

        for (set1, set4) in (
            (v_set1, v_set4),
            (v_set1, drop_tree(v_set4)),
            (drop_tree(v_set1), v_set4),
            (m_set1, m_set4),
            (m_set1, drop_tree(m_set4)),
            (drop_tree(m_set1), m_set4),
        )
            @test_all set1 != set4 && set4 != set1
            @test_all !issubset(set1, set4) && !issubset(set4, set1)
            @test_all !is_subset_that_covers_set(set1, set4) &&
                      !is_subset_that_covers_set(set4, set1)
            if set1 isa MatrixFields.FieldVectorKeys
                @test_all intersect(set1, set4) ==
                          intersect(set4, set1) ==
                          vector_keys_no_tree(@name(a.c.:(1)), @name(a.c.:(2)))
                @test_all union(set1, set4) ==
                          union(set4, set1) ==
                          vector_keys_no_tree(
                              @name(foo),
                              @name(a.b),
                              @name(a.c.:(1)),
                              @name(a.c.:(2)),
                              @name(a.c.:(3))
                          )
                @test_all setdiff(set1, set4) ==
                          vector_keys_no_tree(@name(foo), @name(a.c.:(3)))
                @test_all setdiff(set4, set1) == vector_keys_no_tree(@name(a.b))
            else
                @test_all intersect(set1, set4) ==
                          intersect(set4, set1) ==
                          matrix_keys_no_tree(
                              (@name(foo), @name(a.c.:(1))),
                              (@name(foo.value), @name(a.c.:(2))),
                              (@name(a.b), @name(foo.value)),
                          )
                @test_all union(set1, set4) ==
                          union(set4, set1) ==
                          matrix_keys_no_tree(
                              (@name(foo), @name(a.c.:(1))),
                              (@name(foo), @name(a.c.:(3))),
                              (@name(foo.value), @name(foo)),
                              (@name(foo.value), @name(a.c.:(2))),
                              (@name(a), @name(a.c.:(1))),
                              (@name(a.b), @name(foo.value)),
                              (@name(a.c), @name(foo.value)),
                              (@name(a.c.:(3)), @name(a.c.:(3))),
                          )
                @test_all setdiff(set1, set4) ==
                          matrix_keys_no_tree((@name(foo), @name(a.c.:(3))))
                @test_all setdiff(set4, set1) == matrix_keys_no_tree(
                    (@name(foo.value), @name(foo)),
                    (@name(a), @name(a.c.:(1))),
                    (@name(a.c), @name(foo.value)),
                    (@name(a.c.:(3)), @name(a.c.:(3))),
                )
            end
        end

        for (set1, set4) in (
            (drop_tree(v_set1), drop_tree(v_set4)),
            (drop_tree(m_set1), drop_tree(m_set4)),
        )
            @test_all set1 != set4 && set4 != set1
            @test_all !issubset(set1, set4) && !issubset(set4, set1)
            @test_all !is_subset_that_covers_set(set1, set4) &&
                      !is_subset_that_covers_set(set4, set1)
            @test_throws "FieldNameTree" intersect(set1, set4)
            @test_throws "FieldNameTree" intersect(set4, set1)
            @test_throws "FieldNameTree" union(set1, set4)
            @test_throws "FieldNameTree" union(set4, set1)
            @test_throws "FieldNameTree" setdiff(set1, set4)
            @test_throws "FieldNameTree" setdiff(set4, set1)
        end
    end

    @testset "FieldNameSet Operations for Matrix Multiplication" begin
        # With two exceptions, none of the following operations require a
        # FieldNameTree.

        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree((@name(foo), @name(a.c))),
            vector_keys_no_tree(@name(a.c)),
        ) == vector_keys_no_tree(@name(foo))
        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree((@name(foo), @name(a.c))),
            matrix_keys_no_tree((@name(a.c), @name(a.b))),
        ) == matrix_keys_no_tree((@name(foo), @name(a.b)))

        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree((@name(foo), @name(a.c.:(1)))),
            vector_keys_no_tree(@name(a.c)),
        ) == vector_keys_no_tree(@name(foo))
        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree((@name(foo), @name(a.c.:(1)))),
            matrix_keys_no_tree((@name(a.c), @name(a.b))),
        ) == matrix_keys_no_tree((@name(foo), @name(a.b)))

        @test_throws "extract internal column" MatrixFields.matrix_product_keys(
            matrix_keys_no_tree((@name(foo), @name(a.c))),
            vector_keys_no_tree(@name(a.c.:(1))),
        )
        @test_throws "extract internal column" MatrixFields.matrix_product_keys(
            matrix_keys_no_tree((@name(foo), @name(a.c))),
            matrix_keys_no_tree((@name(a.c.:(1)), @name(a.b))),
        )

        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree((@name(a.c), @name(a.c))),
            vector_keys_no_tree(@name(a.c.:(1))),
        ) == vector_keys_no_tree(@name(a.c.:(1)))
        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree((@name(a.c), @name(a.c))),
            matrix_keys_no_tree((@name(a.c.:(1)), @name(a.b))),
        ) == matrix_keys_no_tree((@name(a.c.:(1)), @name(a.b)))

        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree(
                (@name(foo), @name(a.c.:(1))),
                (@name(foo.value), @name(foo.value)),
            ),
            vector_keys(@name(foo), @name(a.c)),
        ) == vector_keys_no_tree(@name(foo.value))
        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree(
                (@name(foo), @name(a.c.:(1))),
                (@name(foo.value), @name(foo.value)),
            ),
            matrix_keys((@name(foo), @name(a.b)), (@name(a.c), @name(a.b))),
        ) == matrix_keys_no_tree((@name(foo.value), @name(a.b)))

        @test_throws "FieldNameTree" MatrixFields.matrix_product_keys(
            matrix_keys_no_tree(
                (@name(foo), @name(a.c.:(1))),
                (@name(foo.value), @name(foo.value)),
            ),
            vector_keys_no_tree(@name(foo), @name(a.c)),
        )
        @test_throws "FieldNameTree" MatrixFields.matrix_product_keys(
            matrix_keys_no_tree(
                (@name(foo), @name(a.c.:(1))),
                (@name(foo.value), @name(foo.value)),
            ),
            matrix_keys_no_tree(
                (@name(foo), @name(a.b)),
                (@name(a.c), @name(a.b)),
            ),
        )

        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree(
                (@name(a.c.:(1)), @name(foo)),
                (@name(a.c), @name(a.c)),
            ),
            vector_keys(@name(foo), @name(a.c)),
        ) == vector_keys_no_tree(
            @name(a.c.:(1)),
            @name(a.c.:(2)),
            @name(a.c.:(3))
        )
        @test_all MatrixFields.matrix_product_keys(
            matrix_keys_no_tree(
                (@name(a.c.:(1)), @name(foo)),
                (@name(a.c), @name(a.c)),
            ),
            matrix_keys((@name(foo), @name(a.b)), (@name(a.c), @name(a.b))),
        ) == matrix_keys_no_tree(
            (@name(a.c.:(1)), @name(a.b)),
            (@name(a.c.:(2)), @name(a.b)),
            (@name(a.c.:(3)), @name(a.b)),
        )

        @test_throws "FieldNameTree" MatrixFields.matrix_product_keys(
            matrix_keys_no_tree(
                (@name(a.c.:(1)), @name(foo)),
                (@name(a.c), @name(a.c)),
            ),
            vector_keys_no_tree(@name(foo), @name(a.c)),
        )
        @test_throws "FieldNameTree" MatrixFields.matrix_product_keys(
            matrix_keys_no_tree(
                (@name(a.c.:(1)), @name(foo)),
                (@name(a.c), @name(a.c)),
            ),
            matrix_keys_no_tree(
                (@name(foo), @name(a.b)),
                (@name(a.c), @name(a.b)),
            ),
        )

        @test_all MatrixFields.summand_names_for_matrix_product(
            @name(foo),
            matrix_keys_no_tree((@name(foo), @name(a.c))),
            vector_keys_no_tree(@name(a.c)),
        ) == vector_keys_no_tree(@name(a.c))
        @test_all MatrixFields.summand_names_for_matrix_product(
            (@name(foo), @name(a.b)),
            matrix_keys_no_tree((@name(foo), @name(a.c))),
            matrix_keys_no_tree((@name(a.c), @name(a.b))),
        ) == vector_keys_no_tree(@name(a.c))

        @test_all MatrixFields.summand_names_for_matrix_product(
            @name(foo),
            matrix_keys_no_tree((@name(foo), @name(a.c.:(1)))),
            vector_keys_no_tree(@name(a.c)),
        ) == vector_keys_no_tree(@name(a.c.:(1)))
        @test_all MatrixFields.summand_names_for_matrix_product(
            (@name(foo), @name(a.b)),
            matrix_keys_no_tree((@name(foo), @name(a.c.:(1)))),
            matrix_keys_no_tree((@name(a.c), @name(a.b))),
        ) == vector_keys_no_tree(@name(a.c.:(1)))

        @test_all MatrixFields.summand_names_for_matrix_product(
            @name(a.c.:(1)),
            matrix_keys_no_tree((@name(a.c), @name(a.c))),
            vector_keys_no_tree(@name(a.c.:(1))),
        ) == vector_keys_no_tree(@name(a.c.:(1)))
        @test_all MatrixFields.summand_names_for_matrix_product(
            (@name(a.c.:(1)), @name(a.b)),
            matrix_keys_no_tree((@name(a.c), @name(a.c))),
            matrix_keys_no_tree((@name(a.c.:(1)), @name(a.b))),
        ) == vector_keys_no_tree(@name(a.c.:(1)))

        @test_all MatrixFields.summand_names_for_matrix_product(
            @name(foo.value),
            matrix_keys_no_tree(
                (@name(foo), @name(a.c.:(1))),
                (@name(foo.value), @name(foo.value)),
            ),
            vector_keys_no_tree(@name(foo), @name(a.c)),
        ) == vector_keys_no_tree(@name(foo.value), @name(a.c.:(1)))
        @test_all MatrixFields.summand_names_for_matrix_product(
            (@name(foo.value), @name(a.b)),
            matrix_keys_no_tree(
                (@name(foo), @name(a.c.:(1))),
                (@name(foo.value), @name(foo.value)),
            ),
            matrix_keys_no_tree(
                (@name(foo), @name(a.b)),
                (@name(a.c), @name(a.b)),
            ),
        ) == vector_keys_no_tree(@name(foo.value), @name(a.c.:(1)))

        @test_all MatrixFields.summand_names_for_matrix_product(
            @name(a.c.:(1)),
            matrix_keys_no_tree(
                (@name(a.c.:(1)), @name(foo)),
                (@name(a.c), @name(a.c)),
            ),
            vector_keys_no_tree(@name(foo), @name(a.c)),
        ) == vector_keys_no_tree(@name(foo), @name(a.c.:(1)))
        @test_all MatrixFields.summand_names_for_matrix_product(
            (@name(a.c.:(1)), @name(a.b)),
            matrix_keys_no_tree(
                (@name(a.c.:(1)), @name(foo)),
                (@name(a.c), @name(a.c)),
            ),
            matrix_keys_no_tree(
                (@name(foo), @name(a.b)),
                (@name(a.c), @name(a.b)),
            ),
        ) == vector_keys_no_tree(@name(foo), @name(a.c.:(1)))
    end

    @testset "Other FieldNameSet Operations" begin
        # With three exceptions, none of the following operations require a
        # FieldNameTree.

        @test_all MatrixFields.corresponding_matrix_keys(drop_tree(v_set1)) ==
                  matrix_keys_no_tree(
            (@name(foo), @name(foo)),
            (@name(a.c), @name(a.c)),
        )

        @test_all MatrixFields.cartesian_product(
            drop_tree(v_set1),
            drop_tree(v_set4),
        ) == matrix_keys_no_tree(
            (@name(foo), @name(a.b)),
            (@name(foo), @name(a.c.:(1))),
            (@name(foo), @name(a.c.:(2))),
            (@name(a.c), @name(a.b)),
            (@name(a.c), @name(a.c.:(1))),
            (@name(a.c), @name(a.c.:(2))),
        )

        @test_all MatrixFields.matrix_row_keys(drop_tree(m_set1)) ==
                  vector_keys_no_tree(@name(foo), @name(a.b))
        @test_all MatrixFields.matrix_col_keys(drop_tree(m_set1)) ==
                  vector_keys_no_tree(@name(foo), @name(a.c))

        @test_all MatrixFields.matrix_row_keys(m_set4) == vector_keys_no_tree(
            @name(foo.value),
            @name(a.b),
            @name(a.c.:(1)),
            @name(a.c.:(2)),
            @name(a.c.:(3))
        )
        @test_all MatrixFields.matrix_col_keys(m_set4) == vector_keys_no_tree(
            @name(foo.value),
            @name(a.c.:(1)),
            @name(a.c.:(2)),
            @name(a.c.:(3))
        )

        @test_throws "FieldNameTree" MatrixFields.matrix_row_keys(
            drop_tree(m_set4),
        )
        @test_throws "FieldNameTree" MatrixFields.matrix_col_keys(
            drop_tree(m_set4),
        )

        @test_all MatrixFields.matrix_off_diagonal_keys(drop_tree(m_set4)) ==
                  matrix_keys_no_tree(
            (@name(), @name(a.c.:(1))),
            (@name(foo.value), @name(foo)),
            (@name(foo.value), @name(a.c.:(2))),
            (@name(a), @name(foo.value)),
        )

        @test_all MatrixFields.matrix_diagonal_keys(drop_tree(m_set4)) ==
                  matrix_keys_no_tree(
            (@name(foo.value), @name(foo.value)),
            (@name(a.c.:(1)), @name(a.c.:(1))),
            (@name(a.c.:(3)), @name(a.c.:(3))),
        )

        @test_all MatrixFields.matrix_inferred_diagonal_keys(
            drop_tree(m_set1),
        ) == matrix_keys_no_tree(
            (@name(foo), @name(foo)),
            (@name(a.b), @name(a.b)),
            (@name(a.c), @name(a.c)),
        )

        @test_all MatrixFields.matrix_inferred_diagonal_keys(m_set4) ==
                  matrix_keys_no_tree(
            (@name(foo.value), @name(foo.value)),
            (@name(a.b), @name(a.b)),
            (@name(a.c.:(1)), @name(a.c.:(1))),
            (@name(a.c.:(2)), @name(a.c.:(2))),
            (@name(a.c.:(3)), @name(a.c.:(3))),
        )

        @test_throws "FieldNameTree" MatrixFields.matrix_inferred_diagonal_keys(
            drop_tree(m_set4),
        )
    end
end

@testset "FieldNameDict Unit Tests" begin
    x = get_x()
    FT = Float64
    x_FT = convert(replace_basetype(Int, FT, typeof(x)), x)

    C3 = Geometry.Covariant3Vector{FT}
    C12 = Geometry.Covariant12Vector{FT}
    CT3 = Geometry.Contravariant3Vector{FT}
    CT12 = Geometry.Contravariant12Vector{FT}
    C12XC3 = typeof(zero(C12) * zero(C3)')
    CT3XC3 = typeof(zero(CT3) * zero(C3)')
    C12XCT12 = typeof(zero(C12) * zero(CT12)')
    CT3XCT12 = typeof(zero(CT3) * zero(CT12)')
    x_C12 = nested_zero(replace_basetype(Int, C12, typeof(x)))
    x_CT3 = nested_zero(replace_basetype(Int, CT3, typeof(x)))
    x_C12XC3 = nested_zero(replace_basetype(Int, C12XC3, typeof(x)))
    x_CT3XCT12 = nested_zero(replace_basetype(Int, CT3XCT12, typeof(x)))
    I_CT3XC3 = DiagonalMatrixRow(Geometry.AxisTensor(axes(CT3XC3), I))
    I_C12XCT12 = DiagonalMatrixRow(Geometry.AxisTensor(axes(C12XCT12), I))

    center_space, face_space = test_spaces(FT)

    seed!(1) # ensures reproducibility

    vector_of_scalars = Fields.FieldVector(;
        foo = random_field(typeof(x_FT.foo), center_space),
        a = random_field(typeof(x_FT.a), face_space),
    )

    matrix_of_scalars = MatrixFields.FieldMatrix(
        (@name(foo), @name(foo)) =>
            random_field(DiagonalMatrixRow{FT}, center_space),
        (@name(foo), @name(a.b)) => random_field(
            BidiagonalMatrixRow{typeof(x_FT.foo)},
            center_space,
        ),
        (@name(a), @name(foo._value)) =>
            random_field(QuaddiagonalMatrixRow{typeof(x_FT.a)}, face_space),
        (@name(a), @name(a)) => -I,
    )

    vector_of_vectors = Fields.FieldVector(;
        foo = random_field(typeof(x_C12.foo), center_space),
        a = random_field(typeof(x_CT3.a), face_space),
    )

    matrix_of_tensors = MatrixFields.FieldMatrix(
        (@name(foo), @name(foo)) =>
            random_field(DiagonalMatrixRow{C12XCT12}, center_space),
        (@name(foo), @name(a.b)) => random_field(
            BidiagonalMatrixRow{typeof(x_C12XC3.foo)},
            center_space,
        ),
        (@name(a), @name(foo._value)) => random_field(
            QuaddiagonalMatrixRow{typeof(x_CT3XCT12.a)},
            face_space,
        ),
        (@name(a), @name(a)) => -I_CT3XC3,
    )

    for (vector, matrix, I_foo, I_a, is_scalar_test) in (
        (vector_of_scalars, matrix_of_scalars, I, I, true),
        (vector_of_vectors, matrix_of_tensors, I_C12XCT12, I_CT3XC3, false),
    )
        @test_all MatrixFields.field_vector_view(vector) ==
                  MatrixFields.FieldVectorView(
            @name(foo) => vector.foo,
            @name(a) => vector.a,
        )

        vector_view = MatrixFields.field_vector_view(vector)

        matrix_with_tree = MatrixFields.replace_name_tree(
            matrix,
            MatrixFields.FieldNameTree(vector),
        )

        # Some of the `.*`s in the following RegEx strings are needed to account
        # for module qualifiers that may or may not get printed, depending on
        # how these tests are run.

        @test startswith(
            string(vector_view),
            r"""
            .*FieldVectorView with 2 entries:
              @name\(foo\) => .*-valued Field:
                _value: (.|\n)*
              @name\(a\) => .*-valued Field:
                (.|\n)*""",
        )

        @test startswith(
            string(matrix),
            r"""
            .*FieldMatrix with 4 entries:
              \(@name\(foo\), @name\(foo\)\) => .*DiagonalMatrixRow{.*}-valued Field:
                entries: \
                  1: (.|\n)*
              \(@name\(foo\), @name\(a.b\)\) => .*BidiagonalMatrixRow{.*}-valued Field:
                entries: \
                  1: \
                    _value: (.|\n)*
                  2: \
                    _value: (.|\n)*
              \(@name\(a\), @name\(foo._value\)\) => .*QuaddiagonalMatrixRow{.*}-valued Field:
                entries: (.|\n)*
              \(@name\(a\), @name\(a\)\) => .*I""",
        ) broken = Sys.iswindows()

        @test_throws KeyError vector_view[@name(invalid_name)]
        @test_throws KeyError vector_view[@name(a.invalid_name)]

        @test_all vector_view[@name(a)] == vector.a
        @test_all vector_view[@name(a.c)] == vector.a.c
        @test_all vector_view[@name(foo._value)] == vector.foo._value

        @test_throws KeyError matrix[@name(invalid_name), @name(invalid_name)]
        @test_throws KeyError matrix[@name(invalid_name), @name(a)]
        @test_throws KeyError matrix[@name(a), @name(invalid_name)]
        @test_throws KeyError matrix[@name(a), @name(a.invalid_name)]
        @test_throws KeyError matrix[@name(a.invalid_name), @name(a)]

        @test_throws KeyError matrix[@name(a), @name(a.c)]
        @test_throws KeyError matrix[@name(a.c), @name(a)]
        @test_throws KeyError matrix[@name(foo), @name(foo._value)]
        @test_throws KeyError matrix[@name(foo._value), @name(foo)]

        @test_all matrix[@name(a), @name(a)] == -I_a
        @test_all matrix[@name(a.c), @name(a.c)] == -I_a
        @test_all matrix[@name(a.c), @name(a.b)] == zero(I_a)
        @test_all matrix[@name(foo._value), @name(foo._value)] ==
                  matrix[@name(foo), @name(foo)]
        entry = matrix[@name(foo._value), @name(a.b)]
        @test_all entry isa (
            is_scalar_test ? MatrixFields.ColumnwiseBandMatrixField :
            Base.AbstractBroadcasted
        )
        entry = is_scalar_test ? entry : Base.materialize(entry)
        @test entry == map(
            row -> map(foo -> foo.value, row),
            matrix[@name(foo), @name(a.b)],
        )

        @test_all matrix[@name(a.c), @name(foo._value)] isa
                  Base.AbstractBroadcasted
        @test Base.materialize(matrix[@name(a.c), @name(foo._value)]) == map(
            row -> map(a -> a.c, row),
            matrix[@name(a), @name(foo._value)],
        )

        vector_keys = MatrixFields.FieldVectorKeys((@name(foo), @name(a.c)))
        @test_all vector_view[vector_keys] == MatrixFields.FieldVectorView(
            @name(foo) => vector_view[@name(foo)],
            @name(a.c) => vector_view[@name(a.c)],
        )

        matrix_keys1 = MatrixFields.FieldMatrixKeys((
            (@name(foo), @name(foo)),
            (@name(foo), @name(a.b)),
        ),)
        @test_all matrix[matrix_keys1] == MatrixFields.FieldMatrix(
            (@name(foo), @name(foo)) => matrix[@name(foo), @name(foo)],
            (@name(foo), @name(a.b)) => matrix[@name(foo), @name(a.b)],
        )

        matrix_keys2 = MatrixFields.FieldMatrixKeys((
            (@name(foo), @name(foo)),
            (@name(a.c), @name(a.c)), # child key of (@name(a), @name(a))
        ),)
        @test_throws "FieldNameTree" matrix[matrix_keys2]
        @test_all matrix_with_tree[matrix_keys2] == MatrixFields.FieldMatrix(
            (@name(foo), @name(foo)) => matrix[@name(foo), @name(foo)],
            (@name(a.c), @name(a.c)) => matrix[@name(a.c), @name(a.c)],
        )

        partial_identity_matrix1 = MatrixFields.FieldMatrix(
            (@name(foo), @name(foo)) => I_foo,
            (@name(a.b), @name(a.b)) => I, # default for inferred diagonal key
        )
        @test_all one(matrix[matrix_keys1]) == partial_identity_matrix1

        partial_identity_matrix2 = MatrixFields.FieldMatrix(
            (@name(foo), @name(foo)) => I_foo,
            (@name(a.c), @name(a.c)) => I_a,
        )
        @test_all one(matrix_with_tree[matrix_keys2]) ==
                  partial_identity_matrix2

        identity_matrix1 = MatrixFields.FieldMatrix(
            (@name(foo), @name(foo)) => I_foo,
            (@name(a), @name(a)) => I_a,
        )
        @test_throws "FieldNameTree" one(matrix)
        @test_all one(matrix_with_tree) == identity_matrix1

        identity_matrix2 = MatrixFields.FieldMatrix(
            (@name(foo._value), @name(foo._value)) => I_foo,
            (@name(a.b), @name(a.b)) => I_a,
            (@name(a.c.:(1).d), @name(a.c.:(1).d)) => I_a,
            (@name(a.c.:(3).:(1)), @name(a.c.:(3).:(1))) => I_a,
        )
        @test_all MatrixFields.identity_field_matrix(vector) == identity_matrix2

        # TODO: Modify == so that identity_matrix1 == identity_matrix2 is true.
    end

    # FieldNameDict broadcast operations are tested in field_matrix_solvers.jl.
end
