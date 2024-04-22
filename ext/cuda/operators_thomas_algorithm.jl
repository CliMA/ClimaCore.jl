import ClimaComms
import ClimaCore: Fields, Topologies, Spaces, Operators
import ClimaCore.Operators: column_thomas_solve!
import CUDA
using CUDA: @cuda
function column_thomas_solve!(::ClimaComms.CUDADevice, A, b)
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    nthreads, nblocks = Topologies._configure_threadblock(Ni * Nj * Nh)
    @cuda always_inline = true threads = nthreads blocks = nblocks thomas_algorithm_kernel!(
        A,
        b,
    )
end

function thomas_algorithm_kernel!(
    A::Fields.ExtrudedFiniteDifferenceField,
    b::Fields.ExtrudedFiniteDifferenceField,
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(A))
    if idx <= Ni * Nj * Nh
        i, j, h = Topologies._get_idx((Ni, Nj, Nh), idx)
        thomas_algorithm!(Spaces.column(A, i, j, h), Spaces.column(b, i, j, h))
    end
    return nothing
end
