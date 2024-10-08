#=
julia --project
using Revise; include(joinpath("test", "Spaces", "ddss1.jl"))
=#
using Logging
using Test

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

using ClimaComms
ClimaComms.@import_required_backends

function distributed_space(
    (n1, n2),
    (x1periodic, x2periodic),
    (Nq, Nv, Nf);
    x1min = -2π,
    x1max = 2π,
    x2min = -2π,
    x2max = 2π,
)
    device = ClimaComms.device()
    context = ClimaComms.SingletonCommsContext(device)
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(x1min),
            Geometry.XPoint(x1max),
            periodic = x1periodic,
            boundary_names = x1periodic ? nothing : (:west, :east),
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(x2min),
            Geometry.YPoint(x2max),
            periodic = x2periodic,
            boundary_names = x2periodic ? nothing : (:north, :south),
        ),
    )
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    topology = Topologies.Topology2D(context, mesh, Meshes.elements(mesh))
    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(topology, quad)

    return (space, context)
end

init_state_scalar(local_geometry, p) = (; ρ = 1.0)
init_state_vector(local_geometry, p) = Geometry.Covariant12Vector(1.0, -1.0)

#=
 _
|1|
|_|
|2|
|=|
|3|
|_|
|4|
|_|
=#
@testset "4x1 element mesh with periodic boundaries on 1 process" begin
    Nq = 3
    space, comms_ctx = distributed_space((4, 1), (true, true), (Nq, 1, 1))
    device = ClimaComms.device(comms_ctx)
    @test Topologies.nlocalelems(Spaces.topology(space)) == 4


    ClimaComms.allowscalar(device) do
        @test Topologies.local_neighboring_elements(
            Spaces.topology(space),
            1,
        ) == [2, 4]
        @test Topologies.local_neighboring_elements(
            Spaces.topology(space),
            2,
        ) == [1, 3]
        @test Topologies.local_neighboring_elements(
            Spaces.topology(space),
            3,
        ) == [2, 4]
        @test Topologies.local_neighboring_elements(
            Spaces.topology(space),
            4,
        ) == [1, 3]
    end

    y0 = init_state_scalar.(Fields.local_geometry_field(space), Ref(nothing))
    nel = Topologies.nlocalelems(Spaces.topology(space))
    yarr = parent(y0)
    yarr .= reshape(1:(Nq * Nq * nel), (Nq, Nq, 1, nel))

    dss_buffer = Spaces.create_dss_buffer(y0)
    Spaces.weighted_dss!(y0, dss_buffer) # DSS2
#! format: off
    @test Array(yarr[:]) == [18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 9.5, 
                             14.0, 18.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5, 18.5, 
                             23.0, 27.5, 18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 27.5, 
                             32.0, 18.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5]
#! format: on

    p = @allocated Spaces.weighted_dss!(y0, dss_buffer)
    @test p ≤ 266744 # cuda allocation
    @test p == 0 broken = device isa ClimaComms.CUDADevice
end

@testset "test if dss is no-op on an empty field" begin
    Nq = 3
    space, comms_ctx = distributed_space((4, 1), (true, true), (Nq, 1, 1))
    y0 = init_state_scalar.(Fields.local_geometry_field(space), Ref(nothing))
    empty_field = similar(y0, Tuple{})
    dss_buffer = Spaces.create_dss_buffer(empty_field)
    @test empty_field == Spaces.weighted_dss!(empty_field)
end


@testset "4x1 element mesh on 1 process - vector field" begin
    Nq = 3
    space, comms_ctx = distributed_space((4, 1), (true, true), (Nq, 1, 2))
    device = ClimaComms.device(comms_ctx)
    y0 = init_state_vector.(Fields.local_geometry_field(space), Ref(nothing))
    yx = copy(y0)

    dss_buffer = Spaces.create_dss_buffer(y0)
    Spaces.weighted_dss!(y0, dss_buffer)

    @test parent(yx) ≈ parent(y0)

    p = @allocated Spaces.weighted_dss!(y0, dss_buffer)
    @test p ≤ 266744 # cuda allocation
    @test p == 0 broken = device isa ClimaComms.CUDADevice
end
