import CUDA
import ClimaComms
import ClimaCore: Spaces, Quadratures, Topologies, DataLayouts, Fields
import ClimaCore.Operators:
    RoeNumericalFluxKernel,
    RusanovNumericalFluxKernel,
    compute_roe_flux,
    compute_rusanov_flux
import ClimaCore.RecursiveApply: ⊞, ⊟, ⊠, rdiv, rmap
import ClimaCore.DataLayouts: slab, DataSlab2D, slab_index
import ClimaCoreCUDAExt: auto_launch!, thread_index, kernel_indexes

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

    interior_faces_array = Array(Topologies.interior_faces(topology))
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

    i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
    i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

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

    interior_faces_array = Array(Topologies.interior_faces(topology))
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

    i⁻, j⁻ = Topologies.face_node_index(face⁻, Nq, q, false)
    i⁺, j⁺ = Topologies.face_node_index(face⁺, Nq, q, reversed)

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

