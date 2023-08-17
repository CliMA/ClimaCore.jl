import ClimaCore.MatrixFields: @name

include("matrix_field_test_utils.jl")

struct Foo{T}
    _value::T
end
Base.propertynames(::Foo) = (:value,)
Base.getproperty(foo::Foo, s::Symbol) =
    s == :value ? getfield(foo, :_value) : nothing

const x = (; foo = Foo(0), a = (; b = 1, c = ((; d = 2), (;), ((), nothing))))

@testset "FieldName Unit Tests" begin
    @test_all @name() == MatrixFields.FieldName()
    @test_all @name(a.c.:(1).d) == MatrixFields.FieldName(:a, :c, 1, :d)
    @test_all @name(a.c.:(3).:(1)) == MatrixFields.FieldName(:a, :c, 3, 1)

    @test_throws "not a valid property name" @macroexpand @name("a")
    @test_throws "not a valid property name" @macroexpand @name([a])
    @test_throws "not a valid property name" @macroexpand @name((a.c.:(3)):(1))
    @test_throws "not a valid property name" @macroexpand @name(a.c.:(3).(1))

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
        @name(foo.invalid_name),
    )

    @test_all MatrixFields.broadcasted_get_field(x, @name()) == x
    @test_all MatrixFields.broadcasted_get_field(x, @name(foo._value)) == 0
    @test_all MatrixFields.broadcasted_get_field(x, @name(a.b)) == 1
    @test_all MatrixFields.broadcasted_get_field(x, @name(a.c.:(1).d)) == 2

    @test_all MatrixFields.is_child_name(@name(a.c.:(1).d), @name(a))
    @test_all !MatrixFields.is_child_name(@name(a.c.:(1).d), @name(foo))

    @test_all MatrixFields.names_are_overlapping(@name(a), @name(a.c.:(1).d))
    @test_all MatrixFields.names_are_overlapping(@name(a.c.:(1).d), @name(a))
    @test_all !MatrixFields.names_are_overlapping(@name(foo), @name(a.c.:(1).d))
    @test_all !MatrixFields.names_are_overlapping(@name(a.c.:(1).d), @name(foo))

    @test_all MatrixFields.extract_internal_name(@name(a.c.:(1).d), @name(a)) ==
              @name(c.:(1).d)
    @test_throws "is not a child name" MatrixFields.extract_internal_name(
        @name(a.c.:(1).d),
        @name(foo),
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
    @test_throws "does not contain any child names" MatrixFields.child_names(
        @name(a.c.:(2)),
        name_tree,
    )
    @test_throws "does not contain the name" MatrixFields.child_names(
        @name(foo.invalid_name),
        name_tree,
    )
end

@testset "FieldNameSet Unit Tests" begin
    name_tree = MatrixFields.FieldNameTree(x)
    vector_keys(names...) = MatrixFields.FieldVectorKeys(names, name_tree)
    matrix_keys(name_pairs...) =
        MatrixFields.FieldMatrixKeys(name_pairs, name_tree)

    vector_keys_no_tree(names...) = MatrixFields.FieldVectorKeys(names)
    matrix_keys_no_tree(name_pairs...) =
        MatrixFields.FieldMatrixKeys(name_pairs)

    @testset "FieldNameSet Construction" begin
        @test_throws "Invalid FieldNameSet value" vector_keys(
            @name(foo.invalid_name),
        )
        @test_throws "Invalid FieldNameSet value" matrix_keys((
            @name(foo.invalid_name),
            @name(a.c),
        ),)

        for constructor in (vector_keys, vector_keys_no_tree)
            @test_throws "Duplicate FieldNameSet values" constructor(
                @name(foo),
                @name(foo),
            )
            @test_throws "Overlapping FieldNameSet values" constructor(
                @name(foo),
                @name(foo.value),
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

    @testset "FieldNameSet Iteration" begin
        v_set1 = vector_keys(@name(foo), @name(a.c))
        v_set1_no_tree = vector_keys_no_tree(@name(foo), @name(a.c))
        m_set1 = matrix_keys((@name(foo), @name(a.c)), (@name(a.b), @name(foo)))
        m_set1_no_tree = matrix_keys_no_tree(
            (@name(foo), @name(a.c)),
            (@name(a.b), @name(foo)),
        )

        @test_all map(name -> (name, name), v_set1) ==
                  ((@name(foo), @name(foo)), (@name(a.c), @name(a.c)))
        @test_all map(name_pair -> name_pair[1], m_set1) ==
                  (@name(foo), @name(a.b))

        @test_all isnothing(foreach(name -> (name, name), v_set1))
        @test_all isnothing(foreach(name_pair -> name_pair[1], m_set1))

        @test string(v_set1) ==
              "FieldVectorKeys(@name(foo), @name(a.c); <FieldNameTree>)"
        @test string(v_set1_no_tree) ==
              "FieldVectorKeys(@name(foo), @name(a.c))"
        @test string(m_set1) == "FieldMatrixKeys((@name(foo), @name(a.c)), \
                                 (@name(a.b), @name(foo)); <FieldNameTree>)"
        @test string(m_set1_no_tree) == "FieldMatrixKeys((@name(foo), \
                                         @name(a.c)), (@name(a.b), @name(foo)))"

        for set in (v_set1, v_set1_no_tree)
            @test_all @name(foo) in set
            @test_all !(@name(a.b) in set)
            @test_all !(@name(invalid_name) in set)
        end
        for set in (m_set1, m_set1_no_tree)
            @test_all (@name(foo), @name(a.c)) in set
            @test_all !((@name(foo), @name(a.b)) in set)
            @test_all !((@name(foo), @name(invalid_name)) in set)
        end

        @test_all @name(foo.value) in v_set1
        @test_all !(@name(foo.invalid_name) in v_set1)
        @test_throws "FieldNameTree" @name(foo.value) in v_set1_no_tree
        @test_throws "FieldNameTree" @name(foo.invalid_name) in v_set1_no_tree

        @test_all (@name(foo.value), @name(a.c)) in m_set1
        @test_all !((@name(foo.invalid_name), @name(a.c)) in m_set1)
        @test_throws "FieldNameTree" (@name(foo.value), @name(a.c)) in
                                     m_set1_no_tree
        @test_throws "FieldNameTree" (@name(foo.invalid_name), @name(a.c)) in
                                     m_set1_no_tree
    end

    @testset "FieldNameSet Operations for Addition/Subtraction" begin
        v_set1 = vector_keys(@name(foo), @name(a.c))
        v_set1_no_tree = vector_keys_no_tree(@name(foo), @name(a.c))
        m_set1 = matrix_keys((@name(foo), @name(a.c)), (@name(a.b), @name(foo)))
        m_set1_no_tree = matrix_keys_no_tree(
            (@name(foo), @name(a.c)),
            (@name(a.b), @name(foo)),
        )

        v_set2 = vector_keys(@name(foo))
        v_set2_no_tree = vector_keys_no_tree(@name(foo))
        m_set2 = matrix_keys((@name(foo), @name(a.c)))
        m_set2_no_tree = matrix_keys_no_tree((@name(foo), @name(a.c)))

        v_set3 = vector_keys(
            @name(foo.value),
            @name(a.c.:(1)),
            @name(a.c.:(2)),
            @name(a.c.:(3)),
        )
        v_set3_no_tree = vector_keys_no_tree(
            @name(foo.value),
            @name(a.c.:(1)),
            @name(a.c.:(2)),
            @name(a.c.:(3)),
        )
        m_set3 = matrix_keys(
            (@name(foo), @name(a.c.:(1))),
            (@name(foo), @name(a.c.:(2))),
            (@name(foo.value), @name(a.c.:(3))),
            (@name(a.b), @name(foo)),
        )
        m_set3_no_tree = matrix_keys_no_tree(
            (@name(foo), @name(a.c.:(1))),
            (@name(foo), @name(a.c.:(2))),
            (@name(foo.value), @name(a.c.:(3))),
            (@name(a.b), @name(foo)),
        )
        m_set3_no_tree′ = matrix_keys_no_tree(
            (@name(foo.value), @name(a.c.:(1))),
            (@name(foo.value), @name(a.c.:(2))),
            (@name(foo.value), @name(a.c.:(3))),
            (@name(a.b), @name(foo)),
        )

        for (set1, set2) in (
            (v_set1, v_set2),
            (v_set1, v_set2_no_tree),
            (v_set1_no_tree, v_set2),
            (m_set1, m_set2),
            (m_set1, m_set2_no_tree),
            (m_set1_no_tree, m_set2),
        )
            @test_all set1 != set2
            @test_all !issubset(set1, set2)
            @test_all issubset(set2, set1)
            @test_all intersect(set1, set2) == set2
            @test_all union(set1, set2) == set1
            @test_all !MatrixFields.is_subset_that_covers_set(set1, set2)
            @test_all !MatrixFields.is_subset_that_covers_set(set2, set1)
        end

        for (set1, set2) in
            ((v_set1_no_tree, v_set2_no_tree), (m_set1_no_tree, m_set2_no_tree))
            @test_all set1 != set2
            @test_all !issubset(set1, set2)
            @test_all issubset(set2, set1)
            @test_all intersect(set1, set2) == set2
            @test_all union(set1, set2) == set1
            @test_all !MatrixFields.is_subset_that_covers_set(set1, set2)
            @test_throws "FieldNameTree" MatrixFields.is_subset_that_covers_set(
                set2,
                set1,
            )
        end

        for (set1, set3) in (
            (v_set1, v_set3),
            (v_set1, v_set3_no_tree),
            (v_set1_no_tree, v_set3),
        )
            @test_all set1 != set3
            @test_all !issubset(set1, set3)
            @test_all issubset(set3, set1)
            @test_all intersect(set1, set3) == set3
            @test_all union(set1, set3) == set3
            @test_all !MatrixFields.is_subset_that_covers_set(set1, set3)
            @test_all MatrixFields.is_subset_that_covers_set(set3, set1)
        end

        for (set1, set3) in (
            (m_set1, m_set3),
            (m_set1, m_set3_no_tree),
            (m_set1_no_tree, m_set3),
        )
            @test_all set1 != set3
            @test_all !issubset(set1, set3)
            @test_all issubset(set3, set1)
            @test_all intersect(set1, set3) == m_set3_no_tree′
            @test_all union(set1, set3) == m_set3_no_tree′
            @test_all !MatrixFields.is_subset_that_covers_set(set1, set3)
            @test_all MatrixFields.is_subset_that_covers_set(set3, set1)
        end

        for (set1, set3) in
            ((v_set1_no_tree, v_set3_no_tree), (m_set1_no_tree, m_set3_no_tree))
            @test_all set1 != set3
            @test_all !issubset(set1, set3)
            @test_throws "FieldNameTree" issubset(set3, set1)
            @test_throws "FieldNameTree" intersect(set1, set3) == set3
            @test_throws "FieldNameTree" union(set1, set3) == set3
            @test_all !MatrixFields.is_subset_that_covers_set(set1, set3)
            @test_throws "FieldNameTree" MatrixFields.is_subset_that_covers_set(
                set3,
                set1,
            )
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
            @name(a.c.:(3)),
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
        v_set1 = vector_keys(@name(foo), @name(a.c))
        v_set1_no_tree = vector_keys_no_tree(@name(foo), @name(a.c))
        m_set1 = matrix_keys((@name(foo), @name(a.c)), (@name(a.b), @name(foo)))
        m_set1_no_tree = matrix_keys_no_tree(
            (@name(foo), @name(a.c)),
            (@name(a.b), @name(foo)),
        )

        v_set2 = vector_keys(@name(foo.value), @name(a.c.:(1)), @name(a.c.:(3)))
        v_set2_no_tree = vector_keys_no_tree(
            @name(foo.value),
            @name(a.c.:(1)),
            @name(a.c.:(3))
        )
        m_set2 = matrix_keys(
            (@name(foo), @name(foo)),
            (@name(foo), @name(a.c.:(1))),
            (@name(foo.value), @name(a.c.:(3))),
            (@name(a.b), @name(foo.value)),
            (@name(a), @name(a.c)),
        )
        m_set2_no_tree = matrix_keys_no_tree(
            (@name(foo), @name(foo)),
            (@name(foo), @name(a.c.:(1))),
            (@name(foo.value), @name(a.c.:(3))),
            (@name(a.b), @name(foo.value)),
            (@name(a), @name(a.c)),
        )

        @test_all MatrixFields.set_complement(v_set2) ==
                  vector_keys(@name(a.b), @name(a.c.:(2)))
        @test_throws "FieldNameTree" MatrixFields.set_complement(v_set2_no_tree)

        @test_all MatrixFields.set_complement(m_set2) == matrix_keys(
            (@name(foo.value), @name(a.b)),
            (@name(foo.value), @name(a.c.:(2))),
            (@name(a.c), @name(foo.value)),
            (@name(a), @name(a.b)),
        )
        @test_throws "FieldNameTree" MatrixFields.set_complement(m_set2_no_tree)

        for (set1, set2) in (
            (v_set1, v_set2),
            (v_set1, v_set2_no_tree),
            (v_set1_no_tree, v_set2),
        )
            @test_all setdiff(set1, set2) == vector_keys(@name(a.c.:(2)))
        end

        for (set1, set2) in (
            (m_set1, m_set2),
            (m_set1, m_set2_no_tree),
            (m_set1_no_tree, m_set2),
        )
            @test_all setdiff(set1, set2) ==
                      matrix_keys((@name(foo.value), @name(a.c.:(2))))
        end

        for (set1, set2) in
            ((v_set1_no_tree, v_set2_no_tree), (m_set1_no_tree, m_set2_no_tree))
            @test_throws "FieldNameTree" setdiff(set1, set2)
        end

        # With one exception, none of the following operations require a
        # FieldNameTree.

        @test_all MatrixFields.corresponding_matrix_keys(v_set1_no_tree) ==
                  matrix_keys(
            (@name(foo), @name(foo)),
            (@name(a.c), @name(a.c)),
        )

        @test_all MatrixFields.cartesian_product(
            v_set1_no_tree,
            v_set2_no_tree,
        ) == matrix_keys(
            (@name(foo), @name(foo.value)),
            (@name(foo), @name(a.c.:(1))),
            (@name(foo), @name(a.c.:(3))),
            (@name(a.c), @name(foo.value)),
            (@name(a.c), @name(a.c.:(1))),
            (@name(a.c), @name(a.c.:(3))),
        )

        @test_all MatrixFields.matrix_row_keys(m_set1_no_tree) ==
                  vector_keys(@name(foo), @name(a.b))

        @test_all MatrixFields.matrix_row_keys(m_set2) ==
                  vector_keys(@name(foo.value), @name(a.b), @name(a.c))
        @test_throws "FieldNameTree" MatrixFields.matrix_row_keys(
            m_set2_no_tree,
        )

        @test_all MatrixFields.matrix_off_diagonal_keys(m_set2_no_tree) ==
                  matrix_keys(
            (@name(foo), @name(a.c.:(1))),
            (@name(foo.value), @name(a.c.:(3))),
            (@name(a.b), @name(foo.value)),
            (@name(a), @name(a.c)),
        )

        @test_all MatrixFields.matrix_diagonal_keys(m_set2_no_tree) ==
                  matrix_keys(
            (@name(foo), @name(foo)),
            (@name(a.c), @name(a.c)),
        )
    end
end
