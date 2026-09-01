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
import ClimaCore.Utilities: @drop_recursion_limits

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

# reduce_points recurses over sub-blocks and foreach_slice is re-entered when
# an unfused slice loop's function launches its own kernels; the DataLayouts
# exemption does not cover the kwcall methods that belong to this module.
@drop_recursion_limits Core.kwcall

end
