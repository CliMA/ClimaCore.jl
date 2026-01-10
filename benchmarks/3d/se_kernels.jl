using CUDA, BenchmarkTools
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
    Quadratures,
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

function initialize_mwe(device, ::Type{FT}) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    R = FT(6.371229e6)

    npoly = 3
    z_max = FT(30e3)
    z_elem = 64
    h_elem = 30
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
    quad = Quadratures.GLL{npoly + 1}()
    h_space = Spaces.SpectralElementSpace2D(horizontal_topology, quad)

    # vertical space
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
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
    s = init_scalar_field(hv_center_space)

    return (; s = s, uₕ = uₕ, uᵥ = uᵥ)
end

function se_op!(output, input, op)
    CUDA.@sync begin
        @. output = op.(input)
    end
    return nothing
end

FT = Float64
hdiv = Operators.Divergence()
grad = Operators.Gradient()

s, uₕ, uᵥ = initialize_mwe(ClimaComms.CUDADevice(), FT)

div.(grad.(s))

#=
function benchmark_se_kernels(::Type{FT}) where {FT}
    hdiv = Operators.Divergence()
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    wgrad = Operators.WeakGradient()
    curl = Operators.Curl()
    wcurl = Operators.WeakCurl()

    s, uₕ, uᵥ = initialize_mwe(ClimaComms.CUDADevice(), FT)


    println("Benchmarking hdiv")
    hdiv_gpu = hdiv.(uₕ)

    @btime se_op!($hdiv_gpu, $uₕ, $hdiv)

    println("Benchmarking wdiv")
    wdiv_gpu = wdiv.(uₕ)
    @btime se_op!($wdiv_gpu, $uₕ, $wdiv)

    println("Benchmarking gradient")
    gradient_gpu = grad.(s)
    @btime se_op!($gradient_gpu, $s, $grad)

    println("Benchmarking weak gradient")
    wgradient_gpu = wgrad.(s)
    @btime se_op!($wgradient_gpu, $s, $wgrad)

    println("Benchmarking curl")
    curl_gpu = curl.(uₕ)
    @btime se_op!($curl_gpu, $uₕ, $curl)

    println("Benchmarking weak curl")
    wcurl_gpu = wcurl.(uₕ)
    @btime se_op!($wcurl_gpu, $uₕ, $wcurl)

    return nothing
end

benchmark_se_kernels(Float64)
=#
