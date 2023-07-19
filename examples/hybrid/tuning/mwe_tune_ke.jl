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
    DataLayouts

const C1 = ClimaCore.Geometry.Covariant1Vector
const C2 = ClimaCore.Geometry.Covariant2Vector
const C3 = ClimaCore.Geometry.Covariant3Vector
const C12 = ClimaCore.Geometry.Covariant12Vector
const C123 = ClimaCore.Geometry.Covariant123Vector
const CT123 = Geometry.Contravariant123Vector
const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F()

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

    nreps = 10

    for i in 1:nreps
        NVTX.@range "compute_kinetic_ca!" color = colorant"blue" payload = i begin
            CUDA.@sync κ = compute_kinetic_ca!(κ, uₕ, uᵥ)
        end
    end
end

profile_compute_kinetic(Float64)
