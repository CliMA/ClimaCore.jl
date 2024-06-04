# https://github.com/JuliaArrays/StaticArrays.jl/pull/1186

if VERSION >= v"1.11.0-DEV.103"
    const broadcast_flatten = Broadcast.flatten
else
    using Base: tail
    using Base.Broadcast: isflat, Broadcasted

    maybeconstructor(f) = f
    maybeconstructor(::Type{F}) where {F} =
        (args...; kwargs...) -> F(args...; kwargs...)

    broadcast_flatten(bc) = bc
    function broadcast_flatten(bc::Broadcasted{Style}) where {Style}
        isflat(bc) && return bc
        args = cat_nested(bc)
        len = Val{length(args)}()
        makeargs = make_makeargs(bc.args, len, ntuple(_ -> true, len))
        f = maybeconstructor(bc.f)
        @inline newf(args...) = f(prepare_args(makeargs, args)...)
        return Broadcasted{Style}(newf, args, bc.axes)
    end

    cat_nested(bc::Broadcasted) = cat_nested_args(bc.args)
    cat_nested_args(::Tuple{}) = ()
    cat_nested_args(t::Tuple) =
        (cat_nested(t[1])..., cat_nested_args(tail(t))...)
    cat_nested(@nospecialize(a)) = (a,)

    function make_makeargs(args::Tuple, len, flags)
        makeargs, r = _make_makeargs(args, len, flags)
        r isa Tuple{} || error("Internal error. Please file a bug")
        return makeargs
    end

    # We build `makeargs` by traversing the broadcast nodes recursively.
    # note: `len` isa `Val` indicates the length of whole flattened argument list.
    #       `flags` is a tuple of `Bool` with the same length of the rest arguments.
    @inline function _make_makeargs(args::Tuple, len::Val, flags::Tuple)
        head, flags′ = _make_makeargs1(args[1], len, flags)
        rest, flags″ = _make_makeargs(tail(args), len, flags′)
        (head, rest...), flags″
    end
    _make_makeargs(::Tuple{}, ::Val, x::Tuple) = (), x

    # For flat nodes:
    # 1. we just consume one argument, and return the "pick" function
    @inline function _make_makeargs1(
        @nospecialize(a),
        ::Val{N},
        flags::Tuple,
    ) where {N}
        pickargs(::Val{N}) where {N} = (@nospecialize(x::Tuple)) -> x[N]
        return pickargs(Val{N - length(flags) + 1}()), tail(flags)
    end

    # For nested nodes, we form the `makeargs1` based on the child `makeargs` (n += length(cat_nested(bc)))
    @inline function _make_makeargs1(bc::Broadcasted, len::Val, flags::Tuple)
        makeargs, flags′ = _make_makeargs(bc.args, len, flags)
        f = maybeconstructor(bc.f)
        @inline makeargs1(@nospecialize(args::Tuple)) =
            f(prepare_args(makeargs, args)...)
        makeargs1, flags′
    end

    prepare_args(::Tuple{}, @nospecialize(::Tuple)) = ()
    @inline prepare_args(makeargs::Tuple, @nospecialize(x::Tuple)) =
        (makeargs[1](x), prepare_args(tail(makeargs), x)...)
end
