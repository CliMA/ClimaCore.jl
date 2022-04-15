include("ddss_setup.jl")

#=
 _ _
|_|_|
|_|_|
=#
@testset "2x2 element mesh with periodic boundaries" begin
    Nq = 4
    space, comms_ctx = distributed_space((2, 2), (true, true), (Nq, 1, 1))
    init_state(local_geometry, p) = (ρ = 1.0)
    y0 = init_state.(Fields.local_geometry_field(space), Ref(nothing))

    nel = Topologies.nlocalelems(Spaces.topology(space))
    yarr = parent(y0)
    yarr .=
        reshape(1:(Nq * Nq * nel), (Nq, Nq, 1, nel)) .+
        (pid - 1) * Nq * Nq * nel

    Spaces.weighted_dss!(y0)
    passed = 0
    #=
    output from single process run:
    [32.5  24.0  25.0  32.5  14.5  6.0  7.0  14.5  18.5  10.0  11.0  18.5  32.5  24.0  25.0  32.5]
    [32.5  40.0  41.0  32.5  14.5  22.0  23.0  14.5  18.5  26.0  27.0  18.5  32.5  40.0  41.0  32.5]
    [32.5  24.0  25.0  32.5  46.5  38.0  39.0  46.5  50.5  42.0  43.0  50.5  32.5  24.0  25.0  32.5]
    [32.5  40.0  41.0  32.5  46.5  54.0  55.0  46.5  50.5  58.0  59.0  50.5  32.5  40.0  41.0  32.5]
    =#
#! format: off
    if pid == 1
        if yarr[:] == [32.5, 24.0, 25.0, 32.5, 14.5, 6.0, 7.0, 14.5, 18.5, 10.0, 11.0, 18.5, 32.5, 24.0, 25.0,  32.5]
            passed += 1
        end
    elseif pid == 2
        if yarr[:] == [32.5, 40.0, 41.0, 32.5, 14.5, 22.0, 23.0, 14.5, 18.5, 26.0, 27.0, 18.5, 32.5, 40.0, 41.0, 32.5]
            passed += 1
        end
    elseif pid == 3
        if yarr[:] == [32.5, 24.0, 25.0, 32.5, 46.5, 38.0, 39.0, 46.5, 50.5, 42.0, 43.0, 50.5, 32.5, 24.0, 25.0, 32.5]
            passed += 1
        end
    else
        if yarr[:] == [32.5, 40.0, 41.0, 32.5, 46.5, 54.0, 55.0, 46.5, 50.5, 58.0, 59.0, 50.5, 32.5, 40.0, 41.0, 32.5]
            passed += 1
        end
    end
#! format: on
    passed = ClimaComms.reduce(comms_ctx, passed, +)
    if pid == 1
        @test passed == 4
    end
end
