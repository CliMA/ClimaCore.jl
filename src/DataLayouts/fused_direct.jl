"""
    FusedMultiBroadcast(pairs)

A collection of broadcasts that are materialized in a single fused loop, where
`pairs` contains one `destination => broadcasted` pair per broadcast. This is
usually constructed with [`@fused_direct`](@ref).
"""
struct FusedMultiBroadcast{T <: Union{Tuple, AbstractArray}}
    pairs::T
end

"""
    @fused_direct begin
        @. dest1 = rhs1
        @. dest2 = rhs2
        ...
    end

Materializes several broadcasts over the same space in a single fused loop,
instead of launching one loop per broadcast. Every statement in the block must
be a broadcast assignment written with `@.`, including updating assignments
like `@. dest += rhs`. The broadcasts are materialized one after another at
each point, so a broadcast can read the results of earlier broadcasts. However,
non-broadcast subexpressions like `x[1:2]` or `\$(f(x))` are evaluated before
any broadcast is materialized, so they cannot read those results.
"""
macro fused_direct(expr)
    pairs = fused_direct_pairs(__module__, expr)
    return esc(:($(Base.copyto!)($FusedMultiBroadcast(($(pairs...),)))))
end

# Each `@.` statement is expanded in the caller's module and then converted into
# a `destination => broadcasted` pair, following the rules that Julia's lowering
# uses for dot syntax. Statements are not lowered directly because lowered code
# depends on the Julia version (e.g., 1.12 qualifies free variables with their
# module), whereas `@.` and the dot syntax it produces are stable.
function fused_direct_pairs(mod, expr)
    expr isa Expr && expr.head == :block ||
        error("@fused_direct expects a `begin ... end` block of `@.` statements")
    statements = filter(statement -> !(statement isa LineNumberNode), expr.args)
    return map(statements) do statement
        is_dot_macrocall(statement) || error(
            "Only `@.` statements are allowed inside @fused_direct blocks, \
             but found `$statement`",
        )
        fused_direct_pair(macroexpand(mod, statement; recursive = false))
    end
end

is_dot_macrocall(expr) =
    expr isa Expr &&
    expr.head == :macrocall &&
    expr.args[1] in (Symbol("@__dot__"), Symbol("@."))

function fused_direct_pair(expr)
    head_string = expr isa Expr ? string(expr.head) : ""
    is_assignment = startswith(head_string, '.') && endswith(head_string, '=')
    is_assignment && length(expr.args) == 2 ||
        error("@fused_direct statements must be broadcast assignments, but found `$expr`")
    dest = broadcast_destination(expr.args[1])
    bc = fused_broadcasted(expr.args[2])
    if head_string != ".="
        op = Symbol(head_string[2:(end - 1)]) # .+= becomes +, .*= becomes *, etc.
        bc = :($(Base.broadcasted)($op, $dest, $bc))
    elseif !is_broadcasted_call(bc)
        bc = :($(Base.broadcasted)($identity, $bc))
    end
    return :($Pair($dest, $bc))
end

# Destinations of property and index expressions are views, like in lowering.
broadcast_destination(expr) =
    if expr isa Expr && expr.head == :. && expr.args[2] isa QuoteNode
        :($(Base.dotgetproperty)($(eager_expr(expr.args[1])), $(expr.args[2])))
    elseif expr isa Expr && expr.head == :ref
        :($(Base.dotview)($(map(eager_expr, expr.args)...)))
    else
        eager_expr(expr)
    end

# Converts a dotted call into a lazy broadcasted object, fusing nested dotted
# calls into it, and evaluates everything else eagerly.
function fused_broadcasted(expr)
    is_dotted_call(expr) || return eager_expr(expr)
    check_dotted_call(expr)
    f, args = if expr.head == :.
        (eager_expr(expr.args[1]), expr.args[2].args) # f.(args...)
    else
        (Symbol(string(expr.args[1])[2:end]), expr.args[2:end]) # args[1] .op args[2]
    end
    if f == :^ && length(args) == 2 && args[2] isa Integer
        return :($(Base.broadcasted)(
            $(Base.literal_pow),
            ^,
            $(fused_broadcasted(args[1])),
            $(Val(args[2])),
        ))
    end
    return :($(Base.broadcasted)($f, $(map(fused_broadcasted, args)...)))
end

# Dotted calls that are not part of a fused expression, like the call to h in
# `f.(g(h.(x)))`, are materialized before they are used.
eager_expr(expr) =
    if is_dotted_call(expr)
        :($(Base.materialize)($(fused_broadcasted(expr))))
    elseif expr isa Expr
        Expr(expr.head, map(eager_expr, expr.args)...)
    else
        expr
    end

is_dotted_call(expr) =
    expr isa Expr && (
        (expr.head == :. && length(expr.args) == 2 && Meta.isexpr(expr.args[2], :tuple)) ||
        (expr.head == :call && is_dotted_operator(expr.args[1]))
    )

is_dotted_operator(op) =
    op isa Symbol &&
    op != :.. &&
    startswith(string(op), '.') &&
    Base.isoperator(Symbol(string(op)[2:end]))

check_dotted_call(expr) =
    expr.head == :. &&
    any(arg -> Meta.isexpr(arg, (:parameters, :kw, :...)), expr.args[2].args) &&
    error("Keyword arguments and splatting are not supported in @fused_direct blocks")

is_broadcasted_call(expr) =
    Meta.isexpr(expr, :call) && expr.args[1] === Base.broadcasted
