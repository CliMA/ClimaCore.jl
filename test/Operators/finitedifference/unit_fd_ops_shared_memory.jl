#=
julia --project=.buildkite
julia --check-bounds=yes -g2 --project=.buildkite
using Revise; include("test/Operators/finitedifference/unit_fd_ops_shared_memory.jl")
=#
include("utils_fd_ops_shared_memory.jl")

@testset "FD shared memory: dispatch" begin # this ensures that we exercise the correct code-path
    FT = Float64
    device = ClimaComms.device()
    @test device isa ClimaComms.CUDADevice
    ᶜspace = get_space_column(device, FT)
    ᶠspace = Spaces.face_space(ᶜspace)
    f = Fields.Field(FT, ᶠspace)
    grad = Operators.GradientF2C()
    bc = @. lazy(grad(f))
    @test !Operators.any_fd_shmem_supported(bc)
    div = Operators.DivergenceF2C()
    bc = @. lazy(div(Geometry.WVector(f)))
    @test Operators.any_fd_shmem_supported(bc)
end

#! format: off
@testset "Correctness column" begin
    ᶜspace_cpu = get_space_column(ClimaComms.CPUSingleThreaded(), Float64);
    ᶠspace_cpu = Spaces.face_space(ᶜspace_cpu);
    fields_cpu = (; get_fields(ᶜspace_cpu)..., get_fields(ᶠspace_cpu)...);
    kernels!(fields_cpu)
    @info "Compiled CPU kernels"

    ᶜspace = get_space_column(ClimaComms.device(), Float64);
    ClimaComms.device(ᶜspace) isa ClimaComms.CPUSingleThreaded && @warn "Running on the CPU"
    ᶠspace = Spaces.face_space(ᶜspace);
    fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...);
    kernels!(fields)
    @info "Compiled GPU kernels"

    @test compare_cpu_gpu(fields_cpu.ᶜout1, fields.ᶜout1); @test !is_trivial(fields_cpu.ᶜout1)
    @test compare_cpu_gpu(fields_cpu.ᶜout2, fields.ᶜout2); @test !is_trivial(fields_cpu.ᶜout2)
    @test compare_cpu_gpu(fields_cpu.ᶜout3, fields.ᶜout3); @test !is_trivial(fields_cpu.ᶜout3)
    @test compare_cpu_gpu(fields_cpu.ᶜout4, fields.ᶜout4); @test !is_trivial(fields_cpu.ᶜout4)
    @test compare_cpu_gpu(fields_cpu.ᶜout5, fields.ᶜout5); @test !is_trivial(fields_cpu.ᶜout5)
    @test compare_cpu_gpu(fields_cpu.ᶜout6, fields.ᶜout6); @test !is_trivial(fields_cpu.ᶜout6)
    @test compare_cpu_gpu(fields_cpu.ᶜout7, fields.ᶜout7); @test !is_trivial(fields_cpu.ᶜout7)
    @test compare_cpu_gpu(fields_cpu.ᶜout8, fields.ᶜout8); @test !is_trivial(fields_cpu.ᶜout8)
end

@testset "Correctness extruded cubed sphere" begin
    ᶜspace_cpu = get_space_extruded(ClimaComms.CPUSingleThreaded(), Float64);
    ᶠspace_cpu = Spaces.face_space(ᶜspace_cpu);
    fields_cpu = (; get_fields(ᶜspace_cpu)..., get_fields(ᶠspace_cpu)...);
    kernels!(fields_cpu)
    @info "Compiled CPU kernels"

    ᶜspace = get_space_extruded(ClimaComms.device(), Float64);
    ᶠspace = Spaces.face_space(ᶜspace);
    fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...);
    kernels!(fields)
    @info "Compiled GPU kernels"

    @test compare_cpu_gpu(fields_cpu.ᶜout1, fields.ᶜout1); @test !is_trivial(fields_cpu.ᶜout1)
    @test compare_cpu_gpu(fields_cpu.ᶜout2, fields.ᶜout2); @test !is_trivial(fields_cpu.ᶜout2)
    @test compare_cpu_gpu(fields_cpu.ᶜout3, fields.ᶜout3); @test !is_trivial(fields_cpu.ᶜout3)
    @test compare_cpu_gpu(fields_cpu.ᶜout4, fields.ᶜout4); @test !is_trivial(fields_cpu.ᶜout4)
    @test compare_cpu_gpu(fields_cpu.ᶜout5, fields.ᶜout5); @test !is_trivial(fields_cpu.ᶜout5)
    @test compare_cpu_gpu(fields_cpu.ᶜout6, fields.ᶜout6); @test !is_trivial(fields_cpu.ᶜout6)
    @test compare_cpu_gpu(fields_cpu.ᶜout7, fields.ᶜout7); @test !is_trivial(fields_cpu.ᶜout7)
    @test compare_cpu_gpu(fields_cpu.ᶜout8, fields.ᶜout8); @test !is_trivial(fields_cpu.ᶜout8)

end

#! format: on
nothing
