import ClimaComms
import ClimaCore: Fields, Topologies, Spaces, Operators
import ClimaCore.Operators:
    column_thomas_solve!, thomas_algorithm_kernel!, thomas_algorithm!
import CUDA
using CUDA: @cuda
function column_thomas_solve!(::ClimaComms.CUDADevice, A, b)
    us = UniversalSize(Fields.field_values(A))
    args = (A, b, us)
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    threads = threads_via_occupancy(thomas_algorithm_kernel!, args)
    nitems = Ni * Nj * Nh
    n_max_threads = min(threads, nitems)
    p = linear_partition(nitems, n_max_threads)
    auto_launch!(
        thomas_algorithm_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
end

function thomas_algorithm_kernel!(
    A::Fields.ExtrudedFiniteDifferenceField,
    b::Fields.ExtrudedFiniteDifferenceField,
    us::DataLayouts.UniversalSize,
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    if idx <= Ni * Nj * Nh
        i, j, h = cart_ind((Ni, Nj, Nh), idx).I
        thomas_algorithm!(Spaces.column(A, i, j, h), Spaces.column(b, i, j, h))
    end
    return nothing
end
