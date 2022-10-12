#=
using Revise;
include("docs/src/axis_tensor_conversions.jl")
=#
import ClimaCore
import ClimaCore: Domains, Spaces, Meshes, Topologies, Operators, Geometry, Fields
using StaticArrays, IntervalSets
using LinearAlgebra: ×, norm, norm_sqr, dot

# Avoid non-deterministic loops
ClimaCore.enable_threading() = false

# const divₕ = Operators.Divergence()
# const wdivₕ = Operators.WeakDivergence()
# const gradₕ = Operators.Gradient()
# const wgradₕ = Operators.WeakGradient()
# const curlₕ = Operators.Curl()
# const wcurlₕ = Operators.WeakCurl()

const ᶜinterp = Operators.InterpolateF2C()
# const ᶜdivᵥ = Operators.DivergenceF2C(
#     top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
#     bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
# )
const ᶜgradᵥ = Operators.GradientF2C()
# const ᶠcurlᵥ = Operators.CurlC2F(
#     bottom = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
#     top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
# )
# const ᶠfct_boris_book = Operators.FCTBorisBook(
#     bottom = Operators.FirstOrderOneSided(),
#     top = Operators.FirstOrderOneSided(),
# )
# const ᶠfct_zalesak = Operators.FCTZalesak(
#     bottom = Operators.FirstOrderOneSided(),
#     top = Operators.FirstOrderOneSided(),
# )

# const ᶜinterp_stencil = Operators.Operator2Stencil(ᶜinterp)
# const ᶠinterp_stencil = Operators.Operator2Stencil(ᶠinterp)
# const ᶜdivᵥ_stencil = Operators.Operator2Stencil(ᶜdivᵥ)
# const ᶠgradᵥ_stencil = Operators.Operator2Stencil(ᶠgradᵥ)

const C123 = Geometry.Covariant123Vector


function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

#= Return a sphere space =#
function sphere_space(::Type{FT}; zelem = 10) where {FT}

    # 1d domain space
    domain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        periodic = true,
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(mesh)

    quad = Spaces.Quadratures.GLL{4}()
    points, weights = Spaces.Quadratures.quadrature_points(FT, quad)

    space1 = Spaces.SpectralElementSpace1D(topology, quad)

    # finite difference spaces
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0) .. Geometry.ZPoint{FT}(5),
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(mesh)

    space2 = Spaces.CenterFiniteDifferenceSpace(topology)
    space3 = Spaces.FaceFiniteDifferenceSpace(topology)

    # 1×1 domain space
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        Geometry.YPoint{FT}(-2) .. Geometry.YPoint{FT}(8),
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    grid_topology = Topologies.Topology2D(mesh)

    quad = Spaces.Quadratures.GLL{4}()
    points, weights = Spaces.Quadratures.quadrature_points(FT, quad)

    space4 = Spaces.SpectralElementSpace2D(grid_topology, quad)

    # sphere space
    radius = FT(3)
    ne = 4
    Nq = 4
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space5 = Spaces.SpectralElementSpace2D(topology, quad)

    radius = FT(128)
    zlim = (0, 1)
    helem = 4
    Nq = 4

    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    space6 = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(radius)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space7 = Spaces.SpectralElementSpace2D(horztopology, quad)

    cspace = Spaces.ExtrudedFiniteDifferenceSpace(space7, space6)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    return (;cspace, fspace)
end

function main(::Type{FT})
    zelem = 10
    (;cspace, fspace) = sphere_space(FT; zelem)

    nt = (;
        ϕ = FT(0),
        K = FT(0),
        uₕ = Geometry.Covariant12Vector(FT(0), FT(0)),
        # ∇w = Geometry.Covariant3Vector(FT(0)),
        w = Geometry.Covariant3Vector(FT(0)),
        wphy = Geometry.WVector(FT(0)),
    )
    Yc = FieldFromNamedTuple(cspace, nt)
    Yf = FieldFromNamedTuple(fspace, nt)

    @. Yc.K = norm_sqr(C123(Yc.uₕ) + C123(ᶜinterp(Yf.w))) / 2
    # @. Yc.K = norm_sqr(C123(Yc.uₕ) + C123(ᶜinterp(Yf.wcov))) / 2

    Fields.bycolumn(axes(Yc)) do colidx
        if colidx == Fields.ColumnIndex((1, 1), 1)
            zc = Fields.coordinate_field(axes(Yc[colidx])).z
            zf = Fields.coordinate_field(axes(Yf[colidx])).z
            Yc_toa = Spaces.level(Yc[colidx], zelem)
            zc_toa = Spaces.level(zc[colidx], zelem)
            Yf_toa = Spaces.level(Yf[colidx], Spaces.PlusHalf(zelem))
            zf_toa = Spaces.level(zf[colidx], Spaces.PlusHalf(zelem))
            @show Yc_toa
            @show zc_toa
            @show Yf_toa
            @show zf_toa
        end
    end
end

main(Float64)
