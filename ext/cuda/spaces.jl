import ClimaCore: Spaces, AbstractSpace, bycolumn
import ClimaComms
function bycolumn(fn, space::AbstractSpace, ::ClimaComms.CUDADevice)
    fn(:)
    return nothing
end
