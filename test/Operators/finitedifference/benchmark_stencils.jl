#=
julia --project
using Revise; include(joinpath("test", "Operators", "finitedifference", "benchmark_stencils.jl"))
=#
include("benchmark_stencils_utils.jl")

#! format: off
@testset "Benchmark operators" begin
    # column_benchmark_arrays(device, z_elems = 63, bm.float_type)
    # sphere_benchmark_arrays(device, z_elems = 63, helem = 30, Nq = 4, bm.float_type)

    @info "Column"
    bm = Benchmark(;float_type = Float64, device_name)
    # benchmark_operators_column(bm; z_elems = 63, helem = 30, Nq = 4, compile = true)
    (;t_min) = benchmark_operators_column(bm; z_elems = 63, helem = 30, Nq = 4)
    test_results_column(t_min)

    @info "sphere, VIJHF, Float64"
    bm = Benchmark(;float_type = Float64, device_name)
    (;t_min) = benchmark_operators_sphere(bm; z_elems = 63, helem = 30, Nq = 4, horizontal_layout_type = DataLayouts.VIJHF)
    test_results_sphere(t_min)

    @info "sphere, VIJFH, Float32"
    bm = Benchmark(;float_type = Float32, device_name)
    # benchmark_operators_sphere(bm; z_elems = 63, helem = 30, Nq = 4, compile = true)
    (;t_min) = benchmark_operators_sphere(bm; z_elems = 63, helem = 30, Nq = 4, horizontal_layout_type = DataLayouts.VIJFH)

    @info "sphere, VIJHF, Float32"
    bm = Benchmark(;float_type = Float32, device_name)
    # benchmark_operators_sphere(bm; z_elems = 63, helem = 30, Nq = 4, compile = true)
    (;t_min) = benchmark_operators_sphere(bm; z_elems = 63, helem = 30, Nq = 4, horizontal_layout_type = DataLayouts.VIJHF)
end
#! format: on

nothing
