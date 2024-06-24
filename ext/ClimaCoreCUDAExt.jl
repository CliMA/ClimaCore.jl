module ClimaCoreCUDAExt

import NVTX
import ClimaComms
import ClimaCore: DataLayouts, Grids, Spaces, Fields
import ClimaCore: Geometry
import ClimaCore.Geometry: AxisTensor
import CUDA
using CUDA
using CUDA: threadIdx, blockIdx, blockDim
import StaticArrays: SVector, SMatrix, SArray
import ClimaCore.DataLayouts: slab, column
import ClimaCore.Utilities: half
import ClimaCore.Utilities: cart_ind, linear_ind
import ClimaCore.RecursiveApply:
    ⊠, ⊞, ⊟, radd, rmul, rsub, rdiv, rmap, rzero, rmin, rmax

const CuArrayBackedTypes =
    Union{CUDA.CuArray, SubArray{<:Any, <:Any, <:CUDA.CuArray}}

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
include(joinpath("cuda", "operators_thomas_algorithm.jl"))
include(joinpath("cuda", "matrix_fields_single_field_solve.jl"))
include(joinpath("cuda", "matrix_fields_multiple_field_solve.jl"))
include(joinpath("cuda", "operators_spectral_element.jl"))

end
