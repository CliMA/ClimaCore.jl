include("ddss_setup.jl")

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
@testset "4x1 element mesh with periodic boundaries on 2 processes" begin
    Nq = 3
    space, comms_ctx = distributed_space((4, 1), (true, true), (Nq, 1, 1))

    @test Topologies.nlocalelems(Spaces.topology(space)) == 2

    @test Topologies.local_neighboring_elements(space.topology, 1) == [2]
    @test Topologies.local_neighboring_elements(space.topology, 2) == [1]

    @test Topologies.ghost_neighboring_elements(space.topology, 1) == [2]
    @test Topologies.ghost_neighboring_elements(space.topology, 2) == [1]

    init_state(local_geometry, p) = (ρ = 1.0)
    y0 = init_state.(Fields.local_geometry_field(space), Ref(nothing))

    nel = Topologies.nlocalelems(Spaces.topology(space))
    yarr = parent(y0)
    yarr .=
        reshape(1:(Nq * Nq * nel), (Nq, Nq, 1, nel)) .+
        (pid - 1) * Nq * Nq * nel

    Spaces.weighted_dss!(y0)
    #=
    [18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5,
     18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5]
    =#
#! format: off
    if pid == 1
        @test yarr[:] == [18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5]
    else
        @test yarr[:] == [18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5]
    end
#! format: on
end



@testset "4x1 element mesh on 2 processes - vector field" begin
    Nq = 3
    space, comms_ctx = distributed_space((4, 1), (true, true), (Nq, 1, 2))
    init_state(local_geometry, p) = Geometry.Covariant12Vector(1.0, -1.0)
    y0 = init_state.(Fields.local_geometry_field(space), Ref(nothing))
    yx = copy(y0)

    Spaces.weighted_dss!(y0)

    @test yx ≈ y0
end
