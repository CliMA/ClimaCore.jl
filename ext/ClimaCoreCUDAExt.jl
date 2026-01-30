module ClimaCoreCUDAExt

import NVTX
import ClimaCore.Limiters
import ClimaComms
import ClimaCore: DataLayouts, Grids, Spaces, Fields
import ClimaCore: Geometry
import ClimaCore.Geometry: AxisTensor
import CUDA
using CUDA
using CUDA: threadIdx, blockIdx, blockDim
import StaticArrays: SVector, SMatrix, SArray
import ClimaCore.DebugOnly: call_post_op_callback, post_op_callback
import ClimaCore.DataLayouts: mapreduce_cuda
import ClimaCore.DataLayouts: ToCUDA
import ClimaCore.DataLayouts: NoMask, IJHMask
import ClimaCore.DataLayouts: slab, column
import ClimaCore.Utilities: half
import ClimaCore.Utilities: cart_ind, linear_ind
import ClimaCore.DataLayouts: get_N, get_Nv, get_Nij, get_Nij, get_Nh
import ClimaCore.DataLayouts: UniversalSize

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
include(joinpath("cuda", "operators_sem_shmem.jl"))
include(joinpath("cuda", "operators_fd_shmem_common.jl"))
include(joinpath("cuda", "operators_fd_shmem.jl"))
include(joinpath("cuda", "operators_columnwise.jl"))
include(joinpath("cuda", "matrix_fields_single_field_solve.jl"))
include(joinpath("cuda", "matrix_fields_multiple_field_solve.jl"))
include(joinpath("cuda", "operators_spectral_element.jl"))

end
