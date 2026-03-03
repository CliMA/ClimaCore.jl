import CUDA
import ClimaComms
import ClimaCore: Spaces, Quadratures, DataLayouts, Fields
import ClimaCore.Topologies: interior_faces, face_node_index
import ClimaCore.Operators:
    RoeNumericalFluxKernel,
    RusanovNumericalFluxKernel,
    KineticEnergyPreservingNumericalFlux,
    compute_roe_flux,
    compute_rusanov_flux
import ClimaCore.RecursiveApply: ⊞, ⊟, ⊠, rdiv, rmap
import ClimaCore.DataLayouts: slab, DataSlab2D, slab_index
import ClimaCoreCUDAExt: auto_launch!, thread_index, kernel_indexes

# Same topology hooks as CPU path and Topologies.dss_local_faces!: interior_faces(topology),
# face_node_index(face, Nq, q, reversed). Kernels apply numerical flux instead of DSS sum.

function add_numerical_flux_internal_kernel!(
    ::ClimaComms.CUDADevice,
    kernel::RoeNumericalFluxKernel,
    dydt_data,
    y_data,
    parameters_data,
    internal_surface_geometry,
    topology,
    space,
)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    (fluxfn, roe_average_fn, wavespeed_fn) =
        (kernel.fluxfn, kernel.roe_average_fn, kernel.wavespeed_fn)

    interior_faces_array = Array(interior_faces(topology))
    nfaces = length(interior_faces_array)
    nitems = nfaces * Nq

    nitems == 0 && return nothing

    args = (
        dydt_data,
        y_data,
        parameters_data,
        internal_surface_geometry,
        interior_faces_array,
        Nq,
        fluxfn,
        roe_average_fn,
        wavespeed_fn,
    )

    threads = min(nitems, CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
    blocks = cld(nitems, threads)

    auto_launch!(
        add_numerical_flux_internal_roe_kernel!,
        args,
        nitems;
        threads_s = (threads,),
        blocks_s = (blocks,),
    )

    return nothing
end

function add_numerical_flux_internal_roe_kernel!(
    dydt_data,
    y_data,
    parameters_data,
    internal_surface_geometry,
    interior_faces_array,
    Nq,
    fluxfn,
    roe_average_fn,
    wavespeed_fn,
)
    gidx = thread_index()
    total = length(interior_faces_array) * Nq
    gidx > total && return

    face_idx, q = kernel_indexes(gidx, (length(interior_faces_array), Nq))

    elem⁻, face⁻, elem⁺, face⁺, reversed =
        interior_faces_array[face_idx]

    internal_surface_geometry_slab = slab(internal_surface_geometry, face_idx)

    y_slab⁻ = slab(y_data, elem⁻)
    y_slab⁺ = slab(y_data, elem⁺)
    dydt_slab⁻ = slab(dydt_data, elem⁻)
    dydt_slab⁺ = slab(dydt_data, elem⁺)

    sgeom⁻ = internal_surface_geometry_slab[slab_index(q)]

    i⁻, j⁻ = face_node_index(face⁻, Nq, q, false)
    i⁺, j⁺ = face_node_index(face⁺, Nq, q, reversed)

    y⁻ = y_slab⁻[slab_index(i⁻, j⁻)]
    y⁺ = y_slab⁺[slab_index(i⁺, j⁺)]

    parameters⁻ =
        parameters_data isa DataSlab2D ?
        parameters_data[slab_index(i⁻, j⁻)] : parameters_data
    parameters⁺ =
        parameters_data isa DataSlab2D ?
        parameters_data[slab_index(i⁺, j⁺)] : parameters_data

    numflux⁻ = compute_roe_flux(
        sgeom⁻.normal,
        y⁻,
        y⁺,
        parameters⁻,
        parameters⁺,
        fluxfn,
        roe_average_fn,
        wavespeed_fn,
    )

    dydt_slab⁻[slab_index(i⁻, j⁻)] =
        dydt_slab⁻[slab_index(i⁻, j⁻)] ⊟ (sgeom⁻.sWJ ⊠ numflux⁻)
    dydt_slab⁺[slab_index(i⁺, j⁺)] =
        dydt_slab⁺[slab_index(i⁺, j⁺)] ⊞ (sgeom⁻.sWJ ⊠ numflux⁻)

    return nothing
end

function add_numerical_flux_internal_kernel!(
    ::ClimaComms.CUDADevice,
    kernel::RusanovNumericalFluxKernel,
    dydt_data,
    y_data,
    parameters_data,
    internal_surface_geometry,
    topology,
    space,
)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    (fluxfn, wavespeed_fn) = (kernel.fluxfn, kernel.wavespeed_fn)

    interior_faces_array = Array(interior_faces(topology))
    nfaces = length(interior_faces_array)
    nitems = nfaces * Nq

    nitems == 0 && return nothing

    args = (
        dydt_data,
        y_data,
        parameters_data,
        internal_surface_geometry,
        interior_faces_array,
        Nq,
        fluxfn,
        wavespeed_fn,
    )

    threads = min(nitems, CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
    blocks = cld(nitems, threads)

    auto_launch!(
        add_numerical_flux_internal_rusanov_kernel!,
        args,
        nitems;
        threads_s = (threads,),
        blocks_s = (blocks,),
    )

    return nothing
end

function add_numerical_flux_internal_rusanov_kernel!(
    dydt_data,
    y_data,
    parameters_data,
    internal_surface_geometry,
    interior_faces_array,
    Nq,
    fluxfn,
    wavespeed_fn,
)
    gidx = thread_index()
    total = length(interior_faces_array) * Nq
    gidx > total && return

    face_idx, q = kernel_indexes(gidx, (length(interior_faces_array), Nq))

    elem⁻, face⁻, elem⁺, face⁺, reversed =
        interior_faces_array[face_idx]

    internal_surface_geometry_slab = slab(internal_surface_geometry, face_idx)

    y_slab⁻ = slab(y_data, elem⁻)
    y_slab⁺ = slab(y_data, elem⁺)
    dydt_slab⁻ = slab(dydt_data, elem⁻)
    dydt_slab⁺ = slab(dydt_data, elem⁺)

    sgeom⁻ = internal_surface_geometry_slab[slab_index(q)]

    i⁻, j⁻ = face_node_index(face⁻, Nq, q, false)
    i⁺, j⁺ = face_node_index(face⁺, Nq, q, reversed)

    y⁻ = y_slab⁻[slab_index(i⁻, j⁻)]
    y⁺ = y_slab⁺[slab_index(i⁺, j⁺)]

    parameters⁻ =
        parameters_data isa DataSlab2D ?
        parameters_data[slab_index(i⁻, j⁻)] : parameters_data
    parameters⁺ =
        parameters_data isa DataSlab2D ?
        parameters_data[slab_index(i⁺, j⁺)] : parameters_data

    numflux⁻ = compute_rusanov_flux(
        sgeom⁻.normal,
        y⁻,
        y⁺,
        parameters⁻,
        parameters⁺,
        fluxfn,
        wavespeed_fn,
    )

    dydt_slab⁻[slab_index(i⁻, j⁻)] =
        dydt_slab⁻[slab_index(i⁻, j⁻)] ⊟ (sgeom⁻.sWJ ⊠ numflux⁻)
    dydt_slab⁺[slab_index(i⁺, j⁺)] =
        dydt_slab⁺[slab_index(i⁺, j⁺)] ⊞ (sgeom⁻.sWJ ⊠ numflux⁻)

    return nothing
end

# --- Kinetic-energy-preserving numerical flux (GPU) ---

function add_numerical_flux_internal_kernel!(
    ::ClimaComms.CUDADevice,
    ::KineticEnergyPreservingNumericalFlux,
    dydt_data,
    y_data,
    parameters_data,
    internal_surface_geometry,
    topology,
    space,
)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    interior_faces_array = Array(interior_faces(topology))
    nfaces = length(interior_faces_array)
    nitems = nfaces * Nq

    nitems == 0 && return nothing

    args = (
        dydt_data,
        y_data,
        parameters_data,
        internal_surface_geometry,
        interior_faces_array,
        Nq,
    )

    threads = min(
        nitems,
        CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
    )
    blocks = cld(nitems, threads)

    auto_launch!(
        add_numerical_flux_internal_kep_kernel!,
        args,
        nitems;
        threads_s = (threads,),
        blocks_s = (blocks,),
    )

    return nothing
end

# Device-safe KEP flux (inlined EOS to avoid host callbacks)
@inline function _pressure_from_state_kep(state, parameters)
    return parameters.g * state.ρ^2 / 2
end
@inline function _sound_speed_from_state_kep(state, parameters)
    p = _pressure_from_state_kep(state, parameters)
    ρ = state.ρ
    T = real(eltype(ρ))
    return sqrt(max(eps(T), (2 * p) / ρ))
end
@inline function _compute_kep_flux(normal, y⁻, y⁺, p⁻, p⁺)
    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ
    u⁻ = ρu⁻ / ρ⁻
    u⁺ = ρu⁺ / ρ⁺
    θ⁻ = ρθ⁻ / ρ⁻
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁻ = u⁻' * normal
    uₙ⁺ = u⁺' * normal
    m̂ₙ = (ρ⁻ * uₙ⁻ + ρ⁺ * uₙ⁺) / 2
    û = (u⁻ + u⁺) / 2
    pL = _pressure_from_state_kep(y⁻, p⁻)
    pR = _pressure_from_state_kep(y⁺, p⁺)
    p̄ = (pL + pR) / 2
    θ̂ = (θ⁻ + θ⁺) / 2
    flux_ρ = m̂ₙ
    flux_ρu = m̂ₙ * û + p̄ * normal
    flux_ρθ = m̂ₙ * θ̂
    F_core = (ρ = flux_ρ, ρu = flux_ρu, ρθ = flux_ρθ)
    cL = _sound_speed_from_state_kep(y⁻, p⁻)
    cR = _sound_speed_from_state_kep(y⁺, p⁺)
    λL = abs(uₙ⁻) + cL
    λR = abs(uₙ⁺) + cR
    λ = max(λL, λR)
    diss = (λ / 2) ⊠ (y⁻ ⊟ y⁺)
    return F_core ⊞ diss
end

function add_numerical_flux_internal_kep_kernel!(
    dydt_data,
    y_data,
    parameters_data,
    internal_surface_geometry,
    interior_faces_array,
    Nq,
)
    gidx = thread_index()
    total = length(interior_faces_array) * Nq
    gidx > total && return

    face_idx, q = kernel_indexes(gidx, (length(interior_faces_array), Nq))

    elem⁻, face⁻, elem⁺, face⁺, reversed =
        interior_faces_array[face_idx]

    internal_surface_geometry_slab = slab(internal_surface_geometry, face_idx)

    y_slab⁻ = slab(y_data, elem⁻)
    y_slab⁺ = slab(y_data, elem⁺)
    dydt_slab⁻ = slab(dydt_data, elem⁻)
    dydt_slab⁺ = slab(dydt_data, elem⁺)

    sgeom⁻ = internal_surface_geometry_slab[slab_index(q)]

    i⁻, j⁻ = face_node_index(face⁻, Nq, q, false)
    i⁺, j⁺ = face_node_index(face⁺, Nq, q, reversed)

    y⁻ = y_slab⁻[slab_index(i⁻, j⁻)]
    y⁺ = y_slab⁺[slab_index(i⁺, j⁺)]

    parameters⁻ =
        parameters_data isa DataSlab2D ?
        parameters_data[slab_index(i⁻, j⁻)] : parameters_data
    parameters⁺ =
        parameters_data isa DataSlab2D ?
        parameters_data[slab_index(i⁺, j⁺)] : parameters_data

    numflux⁻ = _compute_kep_flux(
        sgeom⁻.normal,
        y⁻,
        y⁺,
        parameters⁻,
        parameters⁺,
    )

    dydt_slab⁻[slab_index(i⁻, j⁻)] =
        dydt_slab⁻[slab_index(i⁻, j⁻)] ⊟ (sgeom⁻.sWJ ⊠ numflux⁻)
    dydt_slab⁺[slab_index(i⁺, j⁺)] =
        dydt_slab⁺[slab_index(i⁺, j⁺)] ⊞ (sgeom⁻.sWJ ⊠ numflux⁻)

    return nothing
end

"""
When on CUDA, dispatch `add_numerical_flux_internal!` for `KineticEnergyPreservingNumericalFlux`
to the GPU kernel; otherwise invoke the generic (CPU) implementation.
"""
function ClimaCore.Operators.add_numerical_flux_internal!(
    fn::KineticEnergyPreservingNumericalFlux,
    dydt,
    args...,
)
    space = axes(dydt)
    device = ClimaComms.device(space)
    if device isa ClimaComms.CUDADevice
        y_data = Fields.todata(args[1])
        parameters_data = length(args) > 1 ? args[2] : nothing
        ClimaCore.Operators.add_numerical_flux_internal_kernel!(
            device,
            fn,
            Fields.field_values(dydt),
            y_data,
            parameters_data,
            Spaces.grid(space).internal_surface_geometry,
            Spaces.topology(space),
            space,
        )
        return
    end
    invoke(
        ClimaCore.Operators.add_numerical_flux_internal!,
        Tuple{Any, typeof(dydt), Vararg{Any}},
        fn,
        dydt,
        args...,
    )
end

