
using ClimaComms
using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Fields, Operators, Quadratures
using CUDA, BenchmarkTools

hdomain = Domains.SphereDomain(6.37122e6)
hmesh = Meshes.EquiangularCubedSphere(hdomain, 30)
htopology = Topologies.Topology2D(hmesh)
hspace = Spaces.SpectralElementSpace2D(htopology, Quadratures.GLL{4}())

vdomain = Domains.IntervalDomain(
    Geometry.ZPoint(0.0),
    Geometry.ZPoint(10e3);
    boundary_tags = (:bottom, :top),
)
vmesh = Meshes.IntervalMesh(vdomain; nelems = 45)
vtopology = Topologies.IntervalTopology(
    ClimaComms.SingletonCommsContext(ClimaComms.device()),
    vmesh,
)
vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

cspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)


u = map(Fields.coordinate_field(cspace)) do coord
    Geometry.Covariant123Vector(1.0, 1.0, 1.0)
end
temp_C123 = similar(u)
ᶜ∇²u = similar(u)

const C1 = Geometry.Covariant1Vector
const C2 = Geometry.Covariant2Vector
const C12 = Geometry.Covariant12Vector
const C3 = Geometry.Covariant3Vector
const C123 = Geometry.Covariant123Vector
const CT1 = Geometry.Contravariant1Vector
const CT2 = Geometry.Contravariant2Vector
const CT12 = Geometry.Contravariant12Vector
const CT3 = Geometry.Contravariant3Vector
const CT123 = Geometry.Contravariant123Vector

const divₕ = Operators.Divergence()
const wdivₕ = Operators.WeakDivergence()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()


function vector_laplacian!(ᶜ∇²u, u, temp_C123)
    # current ClimaAtmos code
    CUDA.@sync begin
        @. ᶜ∇²u = C123(wgradₕ(divₕ(u)))
        @. temp_C123 = C123(curlₕ(C12(u))) + C123(curlₕ(C3(u)))
        @. ᶜ∇²u -= C123(wcurlₕ(C12(temp_C123))) + C123(wcurlₕ(C3(temp_C123)))
    end
    return nothing
end

@benchmark vector_laplacian!(ᶜ∇²u, u, temp_C123)

function vector_laplacian_2!(ᶜ∇²u, u)
    # current ClimaAtmos code
    CUDA.@sync begin
        @. ᶜ∇²u = C123(wgradₕ(divₕ(u))) - C123(wcurlₕ(C123(curlₕ(u))))
    end
    return nothing
end

@benchmark vector_laplacian_2!(ᶜ∇²u, u)
