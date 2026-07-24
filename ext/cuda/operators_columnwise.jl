import ClimaCore: Fields, Spaces

import ClimaCore.Operators:
    columnwise!,
    device_sync_threads,
    columnwise_kernel!,
    universal_index_columnwise,
    local_mem

device_sync_threads(device::ClimaComms.CUDADevice) = CUDA.sync_threads()

local_mem(
    device::ClimaComms.CUDADevice,
    ::Type{T},
    ::Val{dims},
) where {T, dims} = CUDA.CuStaticSharedArray(T, dims)

function columnwise!(
    device::ClimaComms.CUDADevice,
    ᶜf::ᶜF,
    ᶠf::ᶠF,
    ᶜYₜ::Fields.Field,
    ᶠYₜ::Fields.Field,
    ᶜY::Fields.Field,
    ᶠY::Fields.Field,
    p,
    t,
    ::Val{localmem_lg} = Val(true),
    ::Val{localmem_state} = Val(true),
) where {ᶜF, ᶠF, localmem_lg, localmem_state}
    ᶜspace = axes(ᶜY)
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠNv = Spaces.nlevels(ᶠspace)
    ᶜcf = Fields.coordinate_field(ᶜspace)
    (_, Ni, Nj, Nh) = size(Fields.field_values(ᶜcf))
    kernel = CUDA.@cuda(
        always_inline = true,
        launch = false,
        columnwise_kernel!(
            device,
            ᶜf,
            ᶠf,
            ᶜYₜ,
            ᶠYₜ,
            ᶜY,
            ᶠY,
            p,
            t,
            nothing,
            Val(localmem_lg),
            Val(localmem_state),
        )
    )
    threads = (ᶠNv,)
    blocks = (Nh, Ni * Nj)
    kernel(
        device,
        ᶜf,
        ᶠf,
        ᶜYₜ,
        ᶠYₜ,
        ᶜY,
        ᶠY,
        p,
        t,
        nothing,
        Val(localmem_lg),
        Val(localmem_state);
        threads,
        blocks,
    )
end

@inline function universal_index_columnwise(
    device::ClimaComms.CUDADevice,
    UI,
    data,
)
    (v,) = CUDA.threadIdx()
    (h, ij) = CUDA.blockIdx()
    (_, Ni, Nj, _) = size(data)
    Ni * Nj < ij && return CartesianIndex((-1, -1, -1, -1))
    @inbounds (i, j) = CartesianIndices((Ni, Nj))[ij].I
    return CartesianIndex((v, i, j, h))
end
