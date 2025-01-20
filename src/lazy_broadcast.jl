# Temporary helpers to work around https://github.com/CliMA/ClimaCore.jl/issues/2146
using Base.Broadcast: materialize, instantiate
import Base.Broadcast: broadcasted
function _lazy_broadcast end
struct LazyBroadcasted{T}
    value::T
end
Base.Broadcast.broadcasted(::typeof(_lazy_broadcast), x) = LazyBroadcasted(x)
# Cannot return instantiated object here, due to https://github.com/CliMA/ClimaCore.jl/issues/2146
Base.materialize(x::LazyBroadcasted) = x.value
macro _lazy_broadcast(expr)
    return quote
        _lazy_broadcast.($(esc(expr)))
    end
end
macro lazy(expr)
    return quote
        _lazy_broadcast.($(esc(expr)))
    end
end
