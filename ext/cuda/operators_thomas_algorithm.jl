import ClimaComms
import ClimaCore: Fields, Topologies, Spaces, Operators
import ClimaCore.Operators:
    column_thomas_solve!, thomas_algorithm_kernel!, thomas_algorithm!
import CUDA
using CUDA: @cuda
function column_thomas_solve!(::ClimaComms.CUDADevice, A, b)
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    nthreads, nblocks = _configure_threadblock(Ni * Nj * Nh)
    args = (A, b)
    auto_launch!(
        thomas_algorithm_kernel!,
        args,
        size(Fields.field_values(A));
        threads_s = nthreads,
        blocks_s = nblocks,
    )
end

function thomas_algorithm_kernel!(
    A::Fields.ExtrudedFiniteDifferenceField,
    b::Fields.ExtrudedFiniteDifferenceField,
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    if idx <= Ni * Nj * Nh
        i, j, h = cart_ind((Ni, Nj, Nh), idx).I
        thomas_algorithm!(Spaces.column(A, i, j, h), Spaces.column(b, i, j, h))
    end
    return nothing
end
