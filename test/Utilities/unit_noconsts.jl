using Test
import ClimaCore.Utilities: @noconsts

# Each form of @noconsts used in ClimaCore disables constant propagation, so a return type
# that depends on the value of an argument is no longer inferred from a constant value.

with_consts(x) = x == 1 ? 1 : 1.0
@noconsts short_form(x) = x == 1 ? 1 : 1.0
@noconsts function long_form(x)
    return x == 1 ? 1 : 1.0
end
@inline @noconsts inlined_form(x) = x == 1 ? 1 : 1.0
@noconsts @generated generated_form(x) = :(x == 1 ? 1 : 1.0)
@noconsts keyword_form(x; y = 1) = x == y ? 1 : 1.0
statement_form = x -> begin
    @noconsts
    x == 1 ? 1 : 1.0
end

called_with_one(f) = f(1)
inferred_type(f) = only(Base.return_types(called_with_one, (typeof(f),)))

@testset "@noconsts" begin
    @test inferred_type(with_consts) == Int
    @test inferred_type(short_form) == Union{Int, Float64}
    @test inferred_type(long_form) == Union{Int, Float64}
    @test inferred_type(inlined_form) == Union{Int, Float64}
    @test inferred_type(generated_form) == Union{Int, Float64}
    @test inferred_type(keyword_form) == Union{Int, Float64}
    @test inferred_type(statement_form) == Union{Int, Float64}
    @test called_with_one(statement_form) === 1
end
