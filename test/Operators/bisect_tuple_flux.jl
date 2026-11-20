#=
BISECT ONLY: the "NamedTuple fluxes" testset of gpu_edge_cases.jl with its
helpers, logging each device-vs-CPU error. The original header follows.
=#
# Edge-case meshes for GPU operator kernels, compared against a CPU reference:
#  - single-element meshes, where every element face is a periodic self-neighbor
#    (1x1 rectangle) or a panel seam (ne = 1 cubed sphere), so DSS and operator
#    kernels get no interior element connectivity;
#  - boundary-only columns, where the boundary windows of a finite difference
#    stencil cover the whole column and the interior loop is empty.
# Registered :gpu_only because the point is the CPU-vs-GPU comparison; on a CPU
# device it degenerates to comparing two CPU runs.
using Test
using LinearAlgebra: norm
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Fields,
    Geometry,
    Operators,
    MatrixFields,
    Quadratures
import ClimaCore.CommonSpaces:
    MultiColumnSpace, ColumnSpace, ExtrudedCubedSphereSpace, CellCenter

const test_device = ClimaComms.device()
const cpu_device = ClimaComms.CPUSingleThreaded()
@info "gpu_edge_cases: comparing $test_device against $cpu_device"

# Bisect instrumentation: log every device-vs-CPU difference in units of
# eps relative to the largest value (the quantity `rtol` bounds), with `@warn`
# since CI hides `@info`.
function device_matches_cpu(field, field_cpu; rtol)
    device_array = Array(parent(field))
    cpu_array = Array(parent(field_cpu))
    FT = eltype(cpu_array)
    scale = max(norm(device_array, Inf), norm(cpu_array, Inf))
    err = norm(device_array .- cpu_array, Inf)
    @warn "BISECT device vs CPU" testset = Test.get_testset().description FT eltype =
        eltype(field_cpu) err_in_eps = err / (eps(FT) * scale) rtol_in_eps =
        rtol / eps(FT) scale
    return isapprox(device_array, cpu_array; rtol, norm = x -> norm(x, Inf))
end

function periodic_1x1_space(::Type{FT}, device) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(-FT(π)),
            Geometry.XPoint(FT(π)),
            periodic = true,
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(-FT(π)),
            Geometry.YPoint(FT(π)),
            periodic = true,
        ),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())
end

function ne1_sphere_space(::Type{FT}, device) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    domain = Domains.SphereDomain(FT(1))
    mesh = Meshes.EquiangularCubedSphere(domain, 1)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())
end

function single_level_column_spaces(::Type{FT}, device) where {FT}
    context = ClimaComms.SingletonCommsContext(device)
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(1));
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(context, mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(topology)
    return (center_space, Spaces.FaceFiniteDifferenceSpace(center_space))
end

function se_operator_results(space)
    coords = Fields.coordinate_field(space)
    is_sphere = :lat in propertynames(coords)
    f =
        is_sphere ? (@. sind(coords.lat) + cosd(coords.long)) :
        (@. sin(coords.x) + cos(coords.y))
    u = @. Geometry.UVVector(f, 2 * f)
    grad_f = Operators.Gradient().(f)
    wdiv_u = Operators.Divergence{Operators.WeakForm}().(u)
    Spaces.weighted_dss!(grad_f)
    Spaces.weighted_dss!(wdiv_u)
    return (grad_f, wdiv_u)
end


function fd_f2c_results(center_space, face_space)
    FT = Spaces.undertype(center_space)
    zf = Fields.coordinate_field(face_space).z
    u = @. Geometry.WVector(zf + FT(0.5))
    w = @. zf^2 + 1
    divf2c = Operators.DivergenceF2C()
    gradf2c = Operators.GradientF2C()
    interpf2c = Operators.InterpolateF2C()
    return (divf2c.(u), gradf2c.(w), interpf2c.(w))
end

function fd_c2f_results(center_space, face_space)
    FT = Spaces.undertype(center_space)
    zc = Fields.coordinate_field(center_space).z
    f = @. zc^2 + 1
    gradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(1))),
        top = Operators.SetGradient(Geometry.WVector(FT(2))),
    )
    interpc2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    return (gradc2f.(f), interpc2f.(f))
end

# Advection and nested-multiply stencils, covering advection (pointwise and
# operator-matrix) and matrix multiplication in GPU kernels.
function fd_advection_inputs(center_space, face_space)
    FT = Spaces.undertype(center_space)
    zc = Fields.coordinate_field(center_space).z
    zf = Fields.coordinate_field(face_space).z
    x = @. zc / (zc + 1) + FT(0.5)
    w = @. Geometry.WVector(zf / (zf + 1) + FT(0.5))
    u³ = Geometry.Contravariant3Vector.(
        Geometry.contravariant3.(w, Fields.local_geometry_field(face_space)),
    )
    return (; FT, x, w, u³)
end

function fd_advection_results(center_space, face_space)
    (; FT, x, w) = fd_advection_inputs(center_space, face_space)
    up3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.Extrapolate(1),
        top = Operators.Extrapolate(1),
    )
    div_sv = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    )
    gradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(1))),
        top = Operators.SetGradient(Geometry.WVector(FT(0))),
    )
    divf2c = Operators.DivergenceF2C()
    return (
        (@. div_sv(up3(w, x))),
        (@. divf2c(gradc2f(x))),
        fd_pointwise_advection_results(center_space, face_space)...,
    )
end

# The pointwise-evaluated advection operators (no operator-matrix rewrite):
# LinVanLeerC2F, FCTZalesak (tuple-valued advected field and neighboring-face
# velocities) and TVDLimitedFluxC2F (contravariant extra parameter). Safe on
# 1- and 2-center columns, where they clamp all indices.
function fd_pointwise_advection_results(center_space, face_space)
    (; FT, x, w, u³) = fd_advection_inputs(center_space, face_space)
    lvl = Operators.LinVanLeerC2F(
        constraint = Operators.MonotoneLocalExtrema(),
    )
    fctz = Operators.FCTZalesak()
    tvd = Operators.TVDLimitedFluxC2F(method = Operators.MinModLimiter())
    return (
        lvl.(w, x, FT(0.1)),
        fctz.(u³, tuple.(x, x .* FT(0.9))),
        tvd.(u³, x, u³),
    )
end

# Fluxes of a NamedTuple-valued field under divergences whose SetValue
# conditions impose plain vectors, which apply to every component of the flux.
# On GPUs, a divergence caches its operand, where every imposed value must have
# the NamedTuple type of the flux.
tracer_densities(x) = (; tot = x, liq = x / 10, ice = x / 5)
function fd_tuple_flux_results(center_space, face_space)
    (; FT, x, u³) = fd_advection_inputs(center_space, face_space)
    ρq = tracer_densities.(x)
    zero_flux = Operators.SetValue(Geometry.Contravariant3Vector(FT(0)))
    div_sv = Operators.DivergenceF2C(bottom = zero_flux, top = zero_flux)
    div_sv_ex =
        Operators.DivergenceF2C(bottom = zero_flux, top = Operators.Extrapolate())
    lvl = Operators.LinVanLeerC2F(
        constraint = Operators.MonotoneLocalExtrema(),
    )
    up3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    return ((@. div_sv(lvl(u³, ρq, FT(0.5)))), (@. div_sv_ex(up3(u³, ρq))))
end

@testset "NamedTuple fluxes, vector boundary values (device vs CPU) [$FT]" for FT in (
    Float32,
    Float64,
)
    # The columns of the extruded space are short enough for a divergence to
    # keep its cached operand in registers (read with warp shuffles), and those
    # of the column space are too tall, so it is cached in shared memory.
    tuple_flux_spaces(device) = map(
        center_space -> (center_space, Spaces.face_space(center_space)),
        (
            ColumnSpace(
                FT;
                z_elem = 40,
                z_min = 0,
                z_max = 1,
                device,
                context = ClimaComms.SingletonCommsContext(device),
                staggering = CellCenter(),
            ),
            ExtrudedCubedSphereSpace(
                FT;
                z_elem = 10,
                z_min = 0,
                z_max = 1,
                radius = 10,
                h_elem = 2,
                n_quad_points = 3,
                device,
                context = ClimaComms.SingletonCommsContext(device),
                staggering = CellCenter(),
            ),
        ),
    )
    for (spaces, spaces_cpu) in
        zip(tuple_flux_spaces(test_device), tuple_flux_spaces(cpu_device))
        for (field, field_cpu) in zip(
            fd_tuple_flux_results(spaces...),
            fd_tuple_flux_results(spaces_cpu...),
        )
            @test device_matches_cpu(field, field_cpu; rtol = 100 * eps(FT))
            @test all(isfinite, Array(parent(field)))
        end
    end
end
