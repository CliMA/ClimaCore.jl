using CUDA
using ClimaComms

import ClimaCore:
    Device,
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    DataLayouts

function shallow_water_driver_cuda(ARGS, ::Type{FT}) where {FT}
    device = Device.device()
    context = ClimaComms.SingletonCommsContext(device)
    println("running serial simulation on $device device")
    # Test case specifications
    test_name = get(ARGS, 1, "steady_state") # default test case to run
    test_angle_name = get(ARGS, 2, "alpha0") # default test case to run
    α = parse(FT, replace(test_angle_name, "alpha" => ""))

    println("Test name: $test_name, α = $(α)⁰")

    return nothing
end

shallow_water_driver_cuda(ARGS, Float64)
