"""
    Numerical Flux Kernels

This module provides generalized kernel-based implementations of numerical flux
schemes (Roe and Rusanov) that follow the DSS (Direct Stiffness Summation) pattern.
These kernels operate on perimeter data structures and are device-agnostic,
similar to how `dss_local!` and `dss_ghost!` work.

The key difference from the standard numerical flux implementation is that these
kernels work with perimeter data layouts (VIFH/VIHF) and apply fluxes in a
gather-scatter pattern, making them more suitable for parallel execution and
better aligned with the DSS architecture.
"""

import .DataLayouts: slab_index, CartesianFieldIndex, DataSlab2D
import ..Topologies: perimeter_face_indices, interior_faces, ghost_faces, face_node_index,
    boundary_tags, boundary_faces
import ..RecursiveApply: ⊞, ⊟, ⊠, rdiv, rmap, rzero
import ..slab

"""
    RoeNumericalFluxKernel(fluxfn, roe_average_fn, wavespeed_fn)

A kernel-based Roe numerical flux implementation that follows the DSS pattern.

# Fields
- `fluxfn`: Function that computes the physical flux from state and parameters
- `roe_average_fn`: Function that computes Roe-averaged quantities
- `wavespeed_fn`: Function that computes the wave speed from state and parameters

# Usage
The kernel operates on perimeter data and applies Roe flux at interior faces.
"""
struct RoeNumericalFluxKernel{F, R, W}
    fluxfn::F
    roe_average_fn::R
    wavespeed_fn::W
end

"""
    RusanovNumericalFluxKernel(fluxfn, wavespeed_fn)

A kernel-based Rusanov numerical flux implementation that follows the DSS pattern.

# Fields
- `fluxfn`: Function that computes the physical flux from state and parameters
- `wavespeed_fn`: Function that computes the wave speed from state and parameters

# Usage
The kernel operates on perimeter data and applies Rusanov flux at interior faces.
"""
struct RusanovNumericalFluxKernel{F, W}
    fluxfn::F
    wavespeed_fn::W
end

"""
    roe_average(ρ⁻, ρ⁺, var⁻, var⁺)

Compute the Roe average of a variable using density-weighted averaging.

# Arguments
- `ρ⁻`: Density on the minus side
- `ρ⁺`: Density on the plus side
- `var⁻`: Variable value on the minus side
- `var⁺`: Variable value on the plus side

# Returns
The Roe-averaged value: (√ρ⁻ * var⁻ + √ρ⁺ * var⁺) / (√ρ⁻ + √ρ⁺)
"""
@inline function roe_average(ρ⁻, ρ⁺, var⁻, var⁺)
    sqrt_ρ⁻ = sqrt(ρ⁻)
    sqrt_ρ⁺ = sqrt(ρ⁺)
    return (sqrt_ρ⁻ * var⁻ + sqrt_ρ⁺ * var⁺) / (sqrt_ρ⁻ + sqrt_ρ⁺)
end

"""
    compute_roe_flux(
        normal,
        y⁻,
        y⁺,
        parameters⁻,
        parameters⁺,
        fluxfn,
        roe_average_fn,
        wavespeed_fn,
    )

Compute the Roe numerical flux at a face.

# Arguments
- `normal`: Unit normal vector pointing from minus to plus side
- `y⁻`: State tuple on the minus side
- `y⁺`: State tuple on the plus side
- `parameters⁻`: Parameters on the minus side
- `parameters⁺`: Parameters on the plus side
- `fluxfn`: Function to compute physical flux
- `roe_average_fn`: Function to compute Roe averages
- `wavespeed_fn`: Function to compute wave speed

# Returns
The Roe numerical flux vector
"""
@inline function compute_roe_flux(
    normal,
    y⁻,
    y⁺,
    parameters⁻,
    parameters⁺,
    fluxfn,
    roe_average_fn,
    wavespeed_fn,
)
    # Compute average flux
    Favg = rdiv(fluxfn(y⁻, parameters⁻) ⊞ fluxfn(y⁺, parameters⁺), 2)

    # Extract states
    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    # Compute primitive variables
    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * normal

    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * normal

    # Compute pressure and sound speed
    λ = sqrt(parameters⁻.g)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)
    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    # Construct Roe averages
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average_fn(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average_fn(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average_fn(ρ⁻, ρ⁺, c⁻, c⁺)

    # Construct normal velocity
    uₙ = u' * normal

    # Compute differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * normal

    # Compute wave strengths
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # Compute flux corrections
    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρu =
        (
            w1 * (u - c * normal) + w2 * (u + c * normal) + w3 * u +
            w4 * (Δu - Δuₙ * normal)
        ) * 0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ + w5) * 0.5

    Δf = (ρ = -fluxᵀn_ρ, ρu = -fluxᵀn_ρu, ρθ = -fluxᵀn_ρθ)

    # Return average flux plus Roe correction
    return rmap(f -> f' * normal, Favg) ⊞ Δf
end

"""
    compute_rusanov_flux(
        normal,
        y⁻,
        y⁺,
        parameters⁻,
        parameters⁺,
        fluxfn,
        wavespeed_fn,
    )

Compute the Rusanov numerical flux at a face.

# Arguments
- `normal`: Unit normal vector pointing from minus to plus side
- `y⁻`: State tuple on the minus side
- `y⁺`: State tuple on the plus side
- `parameters⁻`: Parameters on the minus side
- `parameters⁺`: Parameters on the plus side
- `fluxfn`: Function to compute physical flux
- `wavespeed_fn`: Function to compute wave speed

# Returns
The Rusanov numerical flux vector
"""
@inline function compute_rusanov_flux(
    normal,
    y⁻,
    y⁺,
    parameters⁻,
    parameters⁺,
    fluxfn,
    wavespeed_fn,
)
    # Compute average flux
    Favg = rdiv(fluxfn(y⁻, parameters⁻) ⊞ fluxfn(y⁺, parameters⁺), 2)

    # Compute maximum wave speed
    λ = max(wavespeed_fn(y⁻, parameters⁻), wavespeed_fn(y⁺, parameters⁺))

    # Return average flux plus Rusanov dissipation
    return rmap(f -> f' * normal, Favg) ⊞ (λ / 2) ⊠ (y⁻ ⊟ y⁺)
end

"""
    add_numerical_flux_internal_kernel!(
        device::ClimaComms.AbstractDevice,
        kernel::RoeNumericalFluxKernel,
        dydt_data,
        y_data,
        parameters_data,
        internal_surface_geometry,
        topology,
        space,
    )

Add Roe numerical flux at interior faces using kernel-based approach.

This function follows the DSS pattern by operating on data layouts directly
and applying fluxes in a gather-scatter pattern similar to `dss_local_faces!`.
"""
function add_numerical_flux_internal_kernel!(
    ::ClimaComms.AbstractCPUDevice,
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

    # Convert to Array to avoid scalar indexing on GPU
    interior_faces_array = Array(interior_faces(topology))
    @inbounds for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
                  enumerate(interior_faces_array)
        internal_surface_geometry_slab = slab(internal_surface_geometry, iface)

        y_slab⁻ = slab(y_data, elem⁻)
        y_slab⁺ = slab(y_data, elem⁺)
        dydt_slab⁻ = slab(dydt_data, elem⁻)
        dydt_slab⁺ = slab(dydt_data, elem⁺)

        for q in 1:Nq
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
        end
    end
    return nothing
end

"""
    add_numerical_flux_internal_kernel!(
        device::ClimaComms.AbstractDevice,
        kernel::RusanovNumericalFluxKernel,
        dydt_data,
        y_data,
        parameters_data,
        internal_surface_geometry,
        topology,
        space,
    )

Add Rusanov numerical flux at interior faces using kernel-based approach.

This function follows the DSS pattern by operating on data layouts directly
and applying fluxes in a gather-scatter pattern similar to `dss_local_faces!`.
"""
function add_numerical_flux_internal_kernel!(
    ::ClimaComms.AbstractCPUDevice,
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

    # Convert to Array to avoid scalar indexing on GPU
    interior_faces_array = Array(interior_faces(topology))
    @inbounds for (iface, (elem⁻, face⁻, elem⁺, face⁺, reversed)) in
                  enumerate(interior_faces_array)
        internal_surface_geometry_slab = slab(internal_surface_geometry, iface)

        y_slab⁻ = slab(y_data, elem⁻)
        y_slab⁺ = slab(y_data, elem⁺)
        dydt_slab⁻ = slab(dydt_data, elem⁻)
        dydt_slab⁺ = slab(dydt_data, elem⁺)

        for q in 1:Nq
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
        end
    end
    return nothing
end

"""
    add_numerical_flux_boundary_kernel!(
        device::ClimaComms.AbstractDevice,
        kernel::Union{RoeNumericalFluxKernel, RusanovNumericalFluxKernel},
        dydt_data,
        y_data,
        parameters_data,
        boundary_surface_geometries,
        topology,
        space,
        boundary_condition_fn,
    )

Add numerical flux at boundary faces using kernel-based approach.

# Arguments
- `boundary_condition_fn`: Function that takes `(normal, (y⁻, parameters))` and
  returns `(y⁺, parameters⁺)` for the boundary condition
"""
function add_numerical_flux_boundary_kernel!(
    ::ClimaComms.AbstractCPUDevice,
    kernel::RoeNumericalFluxKernel,
    dydt_data,
    y_data,
    parameters_data,
    boundary_surface_geometries,
    topology,
    space,
    boundary_condition_fn,
)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    (fluxfn, roe_average_fn, wavespeed_fn) =
        (kernel.fluxfn, kernel.roe_average_fn, kernel.wavespeed_fn)

    # Convert to Array to avoid scalar indexing on GPU
    boundary_tags_array = Array(boundary_tags(topology))
    @inbounds for (iboundary, boundarytag) in enumerate(boundary_tags_array)
        boundary_faces_array = Array(boundary_faces(topology, boundarytag))
        for (iface, (elem⁻, face⁻)) in enumerate(boundary_faces_array)
            boundary_surface_geometry_slab =
                slab(boundary_surface_geometries[iboundary], iface)

            y_slab⁻ = slab(y_data, elem⁻)
            dydt_slab⁻ = slab(dydt_data, elem⁻)

            for q in 1:Nq
                sgeom⁻ = boundary_surface_geometry_slab[slab_index(q)]
                i⁻, j⁻ = face_node_index(face⁻, Nq, q, false)

                y⁻ = y_slab⁻[slab_index(i⁻, j⁻)]
                parameters⁻ =
                    parameters_data isa DataSlab2D ?
                    parameters_data[slab_index(i⁻, j⁻)] :
                    parameters_data

                # Apply boundary condition to get y⁺
                y⁺, parameters⁺ = boundary_condition_fn(
                    sgeom⁻.normal,
                    (y⁻, parameters⁻),
                )

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
            end
        end
    end
    return nothing
end

function add_numerical_flux_boundary_kernel!(
    ::ClimaComms.AbstractCPUDevice,
    kernel::RusanovNumericalFluxKernel,
    dydt_data,
    y_data,
    parameters_data,
    boundary_surface_geometries,
    topology,
    space,
    boundary_condition_fn,
)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    (fluxfn, wavespeed_fn) = (kernel.fluxfn, kernel.wavespeed_fn)

    # Convert to Array to avoid scalar indexing on GPU
    boundary_tags_array = Array(boundary_tags(topology))
    @inbounds for (iboundary, boundarytag) in enumerate(boundary_tags_array)
        boundary_faces_array = Array(boundary_faces(topology, boundarytag))
        for (iface, (elem⁻, face⁻)) in enumerate(boundary_faces_array)
            boundary_surface_geometry_slab =
                slab(boundary_surface_geometries[iboundary], iface)

            y_slab⁻ = slab(y_data, elem⁻)
            dydt_slab⁻ = slab(dydt_data, elem⁻)

            for q in 1:Nq
                sgeom⁻ = boundary_surface_geometry_slab[slab_index(q)]
                i⁻, j⁻ = face_node_index(face⁻, Nq, q, false)

                y⁻ = y_slab⁻[slab_index(i⁻, j⁻)]
                parameters⁻ =
                    parameters_data isa DataSlab2D ?
                    parameters_data[slab_index(i⁻, j⁻)] :
                    parameters_data

                # Apply boundary condition to get y⁺
                y⁺, parameters⁺ = boundary_condition_fn(
                    sgeom⁻.normal,
                    (y⁻, parameters⁻),
                )

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
            end
        end
    end
    return nothing
end
