#=
julia --project
ENV["CLIMACOMMS_DEVICE"] = "CPU";
using Revise; include(joinpath("test", "Spaces", "unit_dss.jl"))
=#
using Test
using ClimaComms
using Random
ClimaComms.@import_required_backends

import ClimaCore
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

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

function get_space_cs(::Type{FT}; context, R = 300.0) where {FT}
    domain = Domains.SphereDomain{FT}(300.0)
    mesh = Meshes.EquiangularCubedSphere(domain, 3)
    topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{4}()
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

function one_to_n_dss(a::AbstractArray)
    _a = Array(a)
    Random.seed!(1234)
    for i in 1:length(_a)
        _a[i] = rand()
    end
    return typeof(a)(_a)
end

function test_dss_count(f::Fields.Field, buff::Topologies.DSSBuffer, nc)
    parent(f) .= one_to_n_dss(parent(f))
    @test allunique(parent(f))
    cf = copy(f)
    Spaces.weighted_dss!(f => buff)
    n_dss_unaffected = count(parent(f) .== parent(cf))
    n_dss_affected = length(parent(f)) - n_dss_unaffected
    return (; n_dss_affected)
end

function get_space_and_buffers3(::Type{FT}; context) where {FT}
    init_state_covariant12(local_geometry, p) =
        Geometry.Covariant12Vector(1.0, -1.0)
    init_state_covariant123(local_geometry, p) =
        Geometry.Covariant123Vector(1.0, -1.0, 1.0)
    init_state_covariant3(local_geometry, p) = Geometry.Covariant3Vector(1.0)

    R = FT(6.371229e6)
    npoly = 2
    z_max = FT(30e3)
    z_elem = 3
    h_elem = 2
    device = ClimaComms.device(context)
    @info "running dss-Covariant123Vector test on $(device)" h_elem z_elem npoly R z_max FT
    # horizontal space
    domain = Domains.SphereDomain{FT}(R)
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
        Geometry.ZPoint{FT}(zero(z_max)),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    z_center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_center_space)
    args = (Fields.local_geometry_field(space), Ref(nothing))
    y12 = init_state_covariant12.(args...)
    y123 = init_state_covariant123.(args...)
    y3 = init_state_covariant3.(args...)
    dss_buffer = (;
        y12 = Spaces.create_dss_buffer(y12),
        y123 = Spaces.create_dss_buffer(y123),
        y3 = Spaces.create_dss_buffer(y3),
    )
    return (; space, y12, y123, y3, dss_buffer)
end

@testset "DSS of AxisTensors on Cubed Sphere" begin
    FT = Float64
    device = ClimaComms.device()
    nt = get_space_and_buffers3(FT; context = ClimaComms.context(device))

    # test DSS for a Covariant12Vector
    # ensure physical velocity is continuous across SE boundary for initial state
    n_dss_affected_y12 =
        test_dss_count(nt.y12, nt.dss_buffer.y12, 2).n_dss_affected
    n_dss_affected_y123 =
        test_dss_count(nt.y123, nt.dss_buffer.y123, 3).n_dss_affected
    n_dss_affected_y3 =
        test_dss_count(nt.y3, nt.dss_buffer.y3, 1).n_dss_affected

    @test n_dss_affected_y12 * 3 / 2 ==
          n_dss_affected_y123 ==
          n_dss_affected_y3 * 3
end

function test_dss_conservation(space)
    Random.seed!(1) # ensures reproducibility
    field = zeros(space)
    FT = Spaces.undertype(space)
    parent(field) .= rand.(FT)
    integral_before_dss = sum(field)
    Spaces.weighted_dss!(field)
    integral_after_dss = sum(field)
    @test integral_after_dss ≈ integral_before_dss rtol = 18 * eps(FT)
end

@testset "DSS Conservation on Cubed Sphere" begin
    device = ClimaComms.device()
    context = ClimaComms.SingletonCommsContext(device)
    for FT in (Float32, Float64)
        bools = (false, true)
        for topography in bools, deep in bools, autodiff_metric in bools
            space_kwargs = (; context, topography, deep, autodiff_metric)
            center_space =
                TU.CenterExtrudedFiniteDifferenceSpace(FT; space_kwargs...)
            face_space = Spaces.face_space(center_space)
            level_space1 = Spaces.level(center_space, 2)
            level_space2 = Spaces.level(face_space, 1 + Spaces.half)
            for space in (center_space, face_space, level_space1, level_space2)
                test_dss_conservation(space)
            end
        end
    end
end
