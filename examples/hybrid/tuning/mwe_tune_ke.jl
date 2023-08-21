using CUDA
using ClimaComms
using ClimaCore
using LinearAlgebra
using NVTX, Colors

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    DataLayouts,
    RecursiveApply

const C1 = ClimaCore.Geometry.Covariant1Vector
const C2 = ClimaCore.Geometry.Covariant2Vector
const C3 = ClimaCore.Geometry.Covariant3Vector
const C12 = ClimaCore.Geometry.Covariant12Vector
const C123 = ClimaCore.Geometry.Covariant123Vector
const CT123 = Geometry.Contravariant123Vector
const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F()

const ⊞ = RecursiveApply.radd

init_uθ(ϕ, z, R) = 1.0 / R
init_vθ(ϕ, z, R) = 1.0 / R
init_w(ϕ, z) = 1.0

function center_initial_condition(ᶜlocal_geometry, R)
    (; lat, long, z) = ᶜlocal_geometry.coordinates
    u₀ = @. init_uθ(lat, z, R)
    v₀ = @. init_vθ(lat, z, R)
    ᶜuₕ_local = @. Geometry.UVVector(u₀, v₀)
    ᶜuₕ = @. Geometry.Covariant12Vector(ᶜuₕ_local, ᶜlocal_geometry)
    return ᶜuₕ
end

function face_initial_condition(local_geometry)
    (; lat, long, z) = local_geometry.coordinates
    w = @. Geometry.Covariant3Vector(init_w(lat, z))
    return w
end

# initialize a scalar field (for KE)
function init_scalar_field(space)
    Y = map(Fields.local_geometry_field(space)) do local_geometry
        h = 0.0
        return h
    end
    return Y
end

function compute_kinetic_ca!(
    κ::Fields.Field,
    uₕ::Fields.Field,
    uᵥ::Fields.Field,
)
    @assert eltype(uₕ) <: Union{C1, C2, C12}
    @assert eltype(uᵥ) <: C3
    #NVTX.@range "compute_kinetic! kernel" color = colorant"brown" begin
    @. κ =
        1 / 2 * (
            dot(C123(uₕ), CT123(uₕ)) +
            ᶜinterp(dot(C123(uᵥ), CT123(uᵥ))) +
            2 * dot(CT123(uₕ), ᶜinterp(C123(uᵥ)))
        )
    #end
end

function compute_kinetic_ref!(
    κ::Fields.Field,
    uₕ::Fields.Field,
    uᵥ::Fields.Field,
)
    hvc_lgd = Spaces.local_geometry_data(axes(uₕ)) # local geometry data for center extruded FD space
    hvf_lgd = Spaces.local_geometry_data(axes(uᵥ)) # local geometry data for face extruded FD space

    κ_val = Fields.field_values(κ)   # field values
    uₕ_val = Fields.field_values(uₕ)
    uᵥ_val = Fields.field_values(uᵥ)


    (Nq, _, _, Nvc, Nh) = size(uₕ_val) # query dimensions
    (Nq, _, _, Nvf, Nh) = size(uᵥ_val)

    nitems = Nvc * Nq * Nq * Nh # # of items that can be independently processed
    max_threads = 256
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)

    @cuda threads = (nthreads,) blocks = (nblocks,) compute_kinetic_ref_kernel!(
        κ_val,
        uₕ_val,
        uᵥ_val,
        hvc_lgd,
        hvf_lgd,
        (Nq, Nvc, Nh),
    )

    return κ
end

function compute_kinetic_ref_kernel!(
    κ_val,
    uₕ_val,
    uᵥ_val,
    hvc_lgd,
    hvf_lgd,
    dims,
)
    gid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    (Nq, Nvc, Nh) = dims
    if gid ≤ Nvc * Nq * Nq * Nh
        h = cld(gid, Nvc * Nq * Nq)
        offset = (h - 1) * Nvc * Nq * Nq
        j = cld(gid - offset, Nvc * Nq)
        offset += (j - 1) * Nvc * Nq
        i = cld(gid - offset, Nvc)
        vc = gid - offset - (i - 1) * Nvc

        idx = CartesianIndex((i, j, 1, vc, h))
        idx⁻ = idx
        idx⁺ = CartesianIndex((i, j, 1, vc + 1, h))

        c_lgd = hvc_lgd[i, j, nothing, vc, h]

        f_lgd⁺ = hvf_lgd[i, j, nothing, vc + 1, h]
        f_lgd⁻ = hvf_lgd[i, j, nothing, vc, h]

        uₕ_val_idx = uₕ_val[idx]
        uᵥ_val_idx⁺ = uᵥ_val[idx⁺]
        uᵥ_val_idx⁻ = uᵥ_val[idx⁻]

        uₕ_c = C123(uₕ_val_idx, c_lgd)
        uₕ_ct = CT123(uₕ_val_idx, c_lgd)

        uᵥ_c⁺ = C123(uᵥ_val_idx⁺, f_lgd⁺)
        uᵥ_c⁻ = C123(uᵥ_val_idx⁻, f_lgd⁻)
        uᵥ_c = RecursiveApply.rdiv(uᵥ_c⁺ ⊞ uᵥ_c⁻, 2)

        uᵥ_ct⁺ = CT123(uᵥ_val_idx⁺, f_lgd⁺)
        uᵥ_ct⁻ = CT123(uᵥ_val_idx⁻, f_lgd⁻)

        κ_val[idx] =
            (
                dot(uₕ_c, uₕ_ct) ⊞ RecursiveApply.rdiv(
                    dot(uᵥ_c⁺, uᵥ_ct⁺) ⊞ dot(uᵥ_c⁻, uᵥ_ct⁻),
                    2,
                ) ⊞ 2 * dot(uₕ_ct, uᵥ_c)
            ) / 2
    end

    return nothing
end

function initialize_mwe(device, ::Type{FT}) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    R = FT(6.371229e6)

    npoly = 3
    z_max = FT(30e3)
    z_elem = 10
    h_elem = 12
    println(
        "initializing on $(context.device); h_elem = $h_elem; z_elem = $z_elem; npoly = $npoly; R = $R; z_max = $z_max; FT = $FT",
    )
    # horizontal space
    domain = Domains.SphereDomain(R)
    horizontal_mesh = Meshes.EquiangularCubedSphere(domain, h_elem)
    horizontal_topology = Topologies.Topology2D(
        context,
        horizontal_mesh,
        Topologies.spacefillingcurve(horizontal_mesh),
    )
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    h_space = Spaces.SpectralElementSpace2D(horizontal_topology, quad)

    # vertical space
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_tags = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)

    z_center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)

    z_face_space = Spaces.FaceFiniteDifferenceSpace(z_topology)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    ᶜlocal_geometry = Fields.local_geometry_field(hv_center_space)
    ᶠlocal_geometry = Fields.local_geometry_field(hv_face_space)
    uₕ = center_initial_condition(ᶜlocal_geometry, R)
    uᵥ = face_initial_condition(ᶠlocal_geometry)
    κ = init_scalar_field(hv_center_space)

    return (; κ = κ, uₕ = uₕ, uᵥ = uᵥ)
end

function profile_compute_kinetic(::Type{FT}) where {FT}
    κ, uₕ, uᵥ = initialize_mwe(ClimaComms.CUDADevice(), FT)
    κ_cpu, uₕ_cpu, uᵥ_cpu = initialize_mwe(ClimaComms.CPUSingleThreaded(), FT)
    # compute kinetic energy
    κ = compute_kinetic_ca!(κ, uₕ, uᵥ)
    κ_cpu = compute_kinetic_ca!(κ_cpu, uₕ_cpu, uᵥ_cpu)

    @show Array(parent(κ)) ≈ parent(κ_cpu)


    κ_ref, uₕ_ref, uᵥ_ref = initialize_mwe(ClimaComms.CUDADevice(), FT)

    @show "before ref version" maximum(parent(κ_ref))

    κ_ref = compute_kinetic_ref!(κ_ref, uₕ_ref, uᵥ_ref)
    @show "after ref version" maximum(parent(κ_ref))

    nreps = 10

    for i in 1:nreps
        NVTX.@range "compute_kinetic_ca!" color = colorant"blue" payload = i begin
            CUDA.@sync κ = compute_kinetic_ca!(κ, uₕ, uᵥ)
        end
    end

    for i in 1:nreps
        NVTX.@range "compute_kinetic_ref!" color = colorant"green" payload = i begin
            CUDA.@sync κ_ref = compute_kinetic_ref!(κ_ref, uₕ_ref, uᵥ_ref)
        end
    end


    for i in 1:nreps
        t_ca = CUDA.@elapsed κ = compute_kinetic_ca!(κ, uₕ, uᵥ)
        t_ref =
            CUDA.@elapsed κ_ref = compute_kinetic_ref!(κ_ref, uₕ_ref, uᵥ_ref)
        println("t_ca = $t_ca (sec); t_ref = $t_ref (sec)")
    end
    return nothing
end

#profile_compute_kinetic(Float64)

function profile_compute_divergence(::Type{FT}) where {FT}
    κ, uₕ, uᵥ = initialize_mwe(ClimaComms.CUDADevice(), FT)
    κ_cpu, uₕ_cpu, uᵥ_cpu = initialize_mwe(ClimaComms.CPUSingleThreaded(), FT)
    hdiv = Operators.Divergence()
    horz_div_cpu = hdiv.(uₕ_cpu)
    horz_div = hdiv.(uₕ)

    nreps = 100

    for i in 1:nreps
        t_div = CUDA.@elapsed horz_div .= hdiv.(uₕ)
        println("t_div = $t_div (sec)")
    end

    for i in 1:nreps
        NVTX.@range "compute_horizontal_divergence!" color = colorant"blue" payload =
            i begin
            CUDA.@sync horz_div .= hdiv.(uₕ)
        end
    end

    return nothing
end

profile_compute_divergence(Float64)
