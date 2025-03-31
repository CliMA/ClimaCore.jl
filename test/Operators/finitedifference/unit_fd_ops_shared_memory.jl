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
    c = Fields.Field(FT, ᶜspace)
    grad = Operators.GradientF2C()
    bc = @. lazy(grad(f))
    @test !Operators.any_fd_shmem_supported(bc)
    div = Operators.DivergenceF2C()
    bc = @. lazy(div(Geometry.WVector(f)))
    @test Operators.any_fd_shmem_supported(bc)
    bc = @. lazy(c + div(Geometry.WVector(f)))
    @test Operators.any_fd_shmem_supported(bc)
end

@testset "Disable for high resolution" begin
    FT = Float64
    device = ClimaComms.device()
    ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
    ᶜspace = get_space_column(device, FT; z_elem = 1000)
    ᶠspace = Spaces.face_space(ᶜspace)
    f = Fields.Field(FT, ᶠspace)
    c = Fields.Field(FT, ᶜspace)
    div = Operators.DivergenceF2C()
    bc = @. lazy(div(Geometry.WVector(f)))
    @test Operators.any_fd_shmem_supported(bc)
    @test !ext.any_fd_shmem_style(ext.disable_shmem_style(bc))
    @. c = div(Geometry.WVector(f))
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
    @test compare_cpu_gpu(fields_cpu.ᶠout1_contra, fields.ᶠout1_contra); @test !is_trivial(fields_cpu.ᶠout1_contra)
    @test compare_cpu_gpu(fields_cpu.ᶠout2_contra, fields.ᶠout2_contra); @test !is_trivial(fields_cpu.ᶠout2_contra)
    @test compare_cpu_gpu(fields_cpu.ᶜout9, fields.ᶜout9); @test !is_trivial(fields_cpu.ᶜout9)
    @test compare_cpu_gpu(fields_cpu.ᶜout10, fields.ᶜout10); @test !is_trivial(fields_cpu.ᶜout10)
    @test compare_cpu_gpu(fields_cpu.ᶜout_uₕ, fields.ᶜout_uₕ); @test !is_trivial(fields_cpu.ᶜout_uₕ)
end

@testset "Correctness plane" begin
    ᶜspace_cpu = get_space_plane(ClimaComms.CPUSingleThreaded(), Float64);
    ᶠspace_cpu = Spaces.face_space(ᶜspace_cpu);
    fields_cpu = (; get_fields(ᶜspace_cpu)..., get_fields(ᶠspace_cpu)...);
    kernels!(fields_cpu)
    @info "Compiled CPU kernels"

    ᶜspace = get_space_plane(ClimaComms.device(), Float64);
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
    @test compare_cpu_gpu(fields_cpu.ᶠout1_contra, fields.ᶠout1_contra); @test !is_trivial(fields_cpu.ᶠout1_contra)
    @test compare_cpu_gpu(fields_cpu.ᶠout2_contra, fields.ᶠout2_contra); @test !is_trivial(fields_cpu.ᶠout2_contra)
    @test compare_cpu_gpu(fields_cpu.ᶜout9, fields.ᶜout9); @test !is_trivial(fields_cpu.ᶜout9)
    @test compare_cpu_gpu(fields_cpu.ᶜout10, fields.ᶜout10); @test !is_trivial(fields_cpu.ᶜout10)
    @test compare_cpu_gpu(fields_cpu.ᶜout_uₕ, fields.ᶜout_uₕ); @test !is_trivial(fields_cpu.ᶜout_uₕ)
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
    @test compare_cpu_gpu(fields_cpu.ᶠout1_contra, fields.ᶠout1_contra); @test !is_trivial(fields_cpu.ᶠout1_contra)
    @test compare_cpu_gpu(fields_cpu.ᶠout2_contra, fields.ᶠout2_contra); @test !is_trivial(fields_cpu.ᶠout2_contra)
    @test compare_cpu_gpu(fields_cpu.ᶜout9, fields.ᶜout9); @test !is_trivial(fields_cpu.ᶜout9)
    @test compare_cpu_gpu(fields_cpu.ᶜout10, fields.ᶜout10); @test !is_trivial(fields_cpu.ᶜout10)
    @test compare_cpu_gpu(fields_cpu.ᶜout_uₕ, fields.ᶜout_uₕ); @test !is_trivial(fields_cpu.ᶜout_uₕ)

end

#! format: on
nothing
