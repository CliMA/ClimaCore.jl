# This module is for backwards compatibility with previous versions of ClimaCore
module RecursiveApply

using ..Utilities: add_auto_broadcasters, drop_auto_broadcasters

struct WithAutoBroadcasters{F}
    f::F
end

# Call f with all arguments wrapped in AutoBroadcasters, then unwrap the result
((; f)::WithAutoBroadcasters)(x) =
    drop_auto_broadcasters(f(add_auto_broadcasters(x)))
((; f)::WithAutoBroadcasters)(x, y) =
    drop_auto_broadcasters(f(add_auto_broadcasters(x), add_auto_broadcasters(y)))

for (f, rf) in ((:+, :radd), (:-, :rsub), (:*, :rmul), (:/, :rdiv))
    @eval const $rf = WithAutoBroadcasters($f)
end
for f in (:zero, :min, :max, :promote_type)
    @eval const $(Symbol(:r, f)) = WithAutoBroadcasters($f)
end

const ⊞ = radd
const ⊟ = rsub
const ⊠ = rmul

end
