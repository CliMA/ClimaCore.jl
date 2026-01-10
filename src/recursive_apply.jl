# This module is for backwards compatibility with previous versions of ClimaCore
module RecursiveApply

using ..Utilities: EnableAutoBroadcasting, new

const radd = ⊞ = EnableAutoBroadcasting(+)
const rsub = ⊟ = EnableAutoBroadcasting(-)
const rmul = ⊠ = EnableAutoBroadcasting(*)
const rdiv = EnableAutoBroadcasting(/)
const rmin = EnableAutoBroadcasting(min)
const rmax = EnableAutoBroadcasting(max)
const rzero = EnableAutoBroadcasting(zero)
const rpromote_type = EnableAutoBroadcasting(promote_type)

end
