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
    Topologies

using ClimaComms
ClimaComms.@import_required_backends

function distributed_space(
    ::Type{FT},
    (n1, n2),
    (x1periodic, x2periodic),
    (Nq, Nv, Nf);
    x1min = -FT(2π),
    x1max = FT(2π),
    x2min = -FT(2π),
    x2max = FT(2π),
) where {FT}
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

init_state_scalar(local_geometry, p, ::Type{FT}) where {FT} = (; ρ = FT(1.0))
init_state_vector(local_geometry, p, ::Type{FT}) where {FT} =
    Geometry.Covariant12Vector(FT(1.0), -FT(1.0))

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
@testset "4x1 element mesh with periodic boundaries on 1 process [$FT]" for FT in (
    Float32,
    Float64,
)
    Nq = 3
    space, comms_ctx = distributed_space(FT, (4, 1), (true, true), (Nq, 1, 1))
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

    y0 = init_state_scalar.(Fields.local_geometry_field(space), Ref(nothing), FT)
    nel = Topologies.nlocalelems(Spaces.topology(space))
    yarr = parent(y0)
    copyto!(yarr, reshape(FT.(1:(Nq * Nq * nel)), (1, Nq, Nq, 1, nel)))

    dss_buffer = Spaces.create_dss_buffer(y0)
    Spaces.weighted_dss!(y0, dss_buffer) # DSS2
#! format: off
    expected_vals = FT[18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 9.5, 
                       14.0, 18.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5, 18.5, 
                       23.0, 27.5, 18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 27.5, 
                       32.0, 18.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5]
    @test Array(yarr[:]) == expected_vals
#! format: on

    p = @allocated Spaces.weighted_dss!(y0, dss_buffer)
    @test p ≤ 266744 # cuda allocation
    @test p == 0 broken = device isa ClimaComms.CUDADevice
end

@testset "test if dss is no-op on an empty field [$FT]" for FT in (Float32, Float64)
    Nq = 3
    space, comms_ctx = distributed_space(FT, (4, 1), (true, true), (Nq, 1, 1))
    y0 = init_state_scalar.(Fields.local_geometry_field(space), Ref(nothing), FT)
    empty_field = similar(y0, Tuple{})
    dss_buffer = Spaces.create_dss_buffer(empty_field)
    @test empty_field == Spaces.weighted_dss!(empty_field)
end

@testset "4x1 element mesh on 1 process - vector field [$FT]" for FT in (Float32, Float64)
    Nq = 3
    space, comms_ctx = distributed_space(FT, (4, 1), (true, true), (Nq, 1, 2))
    device = ClimaComms.device(comms_ctx)
    y0 = init_state_vector.(Fields.local_geometry_field(space), Ref(nothing), FT)
    yx = copy(y0)

    dss_buffer = Spaces.create_dss_buffer(y0)
    Spaces.weighted_dss!(y0, dss_buffer)

    @test parent(yx) ≈ parent(y0)

    p = @allocated Spaces.weighted_dss!(y0, dss_buffer)
    @test p ≤ 266744 # cuda allocation
    @test p == 0 broken = device isa ClimaComms.CUDADevice
end
