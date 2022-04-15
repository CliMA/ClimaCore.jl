include("ddss_setup.jl")

#=
local node and face numbering
          f4
  v1o-------------o v4
    |             |
    |             |
  f1|             |f3
    |             |
    |             |
    o-------------o
   v2     f2      v3
global element numbering
-----------
|1|5| 9|13|
-----------
|2|6|10|14|
---------
|3|7|11|15|
-----------
|4|8|12|16|
-----------
partition numbers
---------
|1|1|2|3|
---------
|1|1|2|3|
---------
|1|2|2|3|
---------
|1|2|3|3|
---------
=#
@testset "4x4 element mesh with non-periodic boundaries on 3 processes" begin
    Nq = 3
    space, comms_ctx = distributed_space((4, 4), (false, false), (Nq, 1, 1))
    init_state(local_geometry, p) = (ρ = 1.0)
    y0 = init_state.(Fields.local_geometry_field(space), Ref(nothing))

    nel = Topologies.nlocalelems(Spaces.topology(space))
    yarr = parent(y0)
    yarr .=
        reshape(1:(Nq * Nq * nel), (Nq, Nq, 1, nel)) .+
        (pid - 1) * Nq * Nq * nel

    Spaces.weighted_dss!(y0)
    passed = 0
#! format: off
    if pid == 1
        if yarr[:] == [1.0, 2.0, 6.5, 4.0, 5.0, 9.5, 22.0, 23.0, 27.5, 6.5, 11.0, 15.5, 9.5, 14.0, 18.5, 27.5, 32.0, 34.25,
                       15.5, 20.0, 24.5, 18.5, 23.0, 27.5, 34.25, 36.5, 41.0, 24.5, 29.0, 30.0, 27.5, 32.0, 33.0, 41.0, 45.5,
                       46.5, 22.0, 23.0, 27.5, 40.0, 41.0, 45.5, 53.5, 54.5, 59.0, 27.5, 32.0, 34.25, 45.5, 50.0, 50.0, 59.0,
                       63.5, 65.75]
            passed += 1
        end
    elseif pid == 2
        if yarr[:] == [34.25, 36.5, 41.0, 50.0, 50.0, 54.5, 65.75, 68.0, 72.5, 41.0, 45.5, 46.5, 54.5, 59.0, 60.0, 72.5, 77.0,
                       78.0, 53.5, 54.5, 59.0, 67.0, 68.0, 72.5, 85.0, 86.0, 90.5, 59.0, 63.5, 65.75, 72.5, 77.0, 81.5, 90.5,
                       95.0, 99.5, 65.75, 68.0, 72.5, 81.5, 86.0, 90.5, 99.5, 104.0, 108.5]
            passed += 1
        end
    else
        if yarr[:] == [72.5, 77.0, 78.0, 90.5, 95.0, 96.0, 108.5, 113.0, 114.0, 85.0, 86.0, 90.5, 103.0, 104.0, 108.5, 106.0,
                       107.0, 111.5, 90.5, 95.0, 99.5, 108.5, 113.0, 117.5, 111.5, 116.0, 120.5, 99.5, 104.0, 108.5, 117.5,
                       122.0, 126.5, 120.5, 125.0, 129.5, 108.5, 113.0, 114.0, 126.5, 131.0, 132.0, 129.5, 134.0, 135.0]
            passed += 1
        end
    end
#! format: on
    passed = ClimaComms.reduce(comms_ctx, passed, +)
    if pid == 1
        @test passed == 3
    end
end
