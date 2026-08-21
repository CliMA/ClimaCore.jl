module ClimaCoreCUDAExt

import NVTX
import ClimaCore.Limiters
import ClimaComms
import ClimaCore: DataLayouts, Fields, Geometry, Utilities
import ClimaCore.Geometry: AbstractTensor
import CUDA
using CUDA
using CUDA: threadIdx, blockIdx, blockDim
import StaticArrays: SVector, SMatrix, SArray
import ClimaCore.DebugOnly: call_post_op_callback, post_op_callback
import ClimaCore.DataLayouts: NoMask, IJHMask
import ClimaCore.DataLayouts: slab, column
import ClimaCore.Utilities: half, new, return_type

include(joinpath("cuda", "adapt.jl"))
include(joinpath("cuda", "cuda_utils.jl"))
include(joinpath("cuda", "data_layouts.jl"))
include(joinpath("cuda", "fields.jl"))
include(joinpath("cuda", "topologies_dss.jl"))
include(joinpath("cuda", "operators_finite_difference.jl"))
include(joinpath("cuda", "remapping_distributed.jl"))
include(joinpath("cuda", "operators_integral.jl"))
include(joinpath("cuda", "remapping_interpolate_array.jl"))
include(joinpath("cuda", "limiters.jl"))
include(joinpath("cuda", "matrix_fields_single_field_solve.jl"))
include(joinpath("cuda", "matrix_fields_multiple_field_solve.jl"))
include(joinpath("cuda", "operators_dg.jl"))

# Lift the recursion limit for the device-side reduce_points, whose recursion
# over sub-blocks forwards kwargs and looks unbounded to the compiler (causing
# it to widen the argument types and use dynamic dispatch), and for the
# host-side foreach_slice, which is re-entered when the function passed to an
# unfused slice loop launches its own kernels. The limit must also be lifted on
# the keyword-argument body functions, since that is where the recursion occurs.
@static if hasfield(Method, :recursion_relation)
    for f in (DataLayouts.reduce_points, DataLayouts.foreach_slice), method in methods(f)
        method.module === (@__MODULE__) || continue
        method.recursion_relation = Returns(true)
        body_function = Base.bodyfunction(method)
        isnothing(body_function) && continue
        for body_method in methods(body_function)
            body_method.recursion_relation = Returns(true)
        end
    end
end

end
