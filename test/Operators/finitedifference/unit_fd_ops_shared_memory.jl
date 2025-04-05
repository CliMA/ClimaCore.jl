#=
julia --project=.buildkite
julia --check-bounds=yes -g2 --project=.buildkite
using Revise; include("test/Operators/finitedifference/unit_fd_ops_shared_memory.jl")
=#
include("utils_fd_ops_shared_memory.jl")

Operators.use_fd_shmem() = true
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
    ᶠgrad = Operators.GradientC2F(;
        bottom = Operators.SetValue(FT(0)),
        top = Operators.SetValue(FT(0)),
    )
    bc = @. lazy(ᶠgrad(c))
    @test Operators.any_fd_shmem_supported(bc)
    @test Operators.fd_shmem_is_supported(bc)
end

@testset "Utility functions" begin
    FT = Float64
    device = ClimaComms.device()
    ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
    ᶜspace = get_space_column(device, FT; z_elem = 10)
    ᶠspace = Spaces.face_space(ᶜspace)
    f = Fields.Field(FT, ᶠspace)
    c = Fields.Field(FT, ᶜspace)
    fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...)
    (; ϕ, ρ) = fields

    div = Operators.DivergenceF2C()
    bc = @. lazy(div(Geometry.WVector(f)))
    test_face_windows(bc)

    div_bcs = Operators.DivergenceF2C(;
        bottom = Operators.SetValue(Geometry.Covariant3Vector(FT(10))),
        top = Operators.SetValue(Geometry.Covariant3Vector(FT(10))),
    )
    bc = @. lazy(div_bcs(Geometry.WVector(f) * 2))
    test_face_windows(bc)

    ᶠgrad = Operators.GradientC2F(;
        bottom = Operators.SetValue(FT(0)),
        top = Operators.SetValue(FT(0)),
    )
    bc = @. lazy(ᶠgrad(c))
    test_center_windows(bc)

    div = Operators.DivergenceF2C()
    ᶠwinterp = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    bc = @. lazy(div(Geometry.WVector(ᶠwinterp(ϕ, ρ))))
    test_center_windows(bc)
    # highly nested cases
    ᶜinterp = Operators.InterpolateF2C()
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    bc = @. lazy(ᶠgrad(ᶜinterp(ᶠinterp(ᶜinterp(f))))) # exercises very nested operation
    test_center_windows(bc)
    test_face_windows(bc.args[1])
    test_center_windows(bc.args[1].args[1])
    # end

    # #! format: off
    # @testset "Correctness column" begin
    ᶜspace_cpu = get_space_column(ClimaComms.CPUSingleThreaded(), Float64)
    ᶠspace_cpu = Spaces.face_space(ᶜspace_cpu)
    fields_cpu = (; get_fields(ᶜspace_cpu)..., get_fields(ᶠspace_cpu)...)
    kernels!(fields_cpu)
    @info "Compiled CPU kernels"

    ᶜspace = get_space_column(ClimaComms.device(), Float64)
    ClimaComms.device(ᶜspace) isa ClimaComms.CPUSingleThreaded &&
        @warn "Running on the CPU"
    ᶠspace = Spaces.face_space(ᶜspace)
    fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...)
    kernels!(fields)
    @info "Compiled GPU kernels"

    @test compare_cpu_gpu(fields_cpu.ᶜout1, fields.ᶜout1)
    @test !is_trivial(fields_cpu.ᶜout1)
    @test compare_cpu_gpu(fields_cpu.ᶜout2, fields.ᶜout2)
    @test !is_trivial(fields_cpu.ᶜout2)
    @test compare_cpu_gpu(fields_cpu.ᶜout3, fields.ᶜout3)
    @test !is_trivial(fields_cpu.ᶜout3)
    @test compare_cpu_gpu(fields_cpu.ᶜout4, fields.ᶜout4)
    @test !is_trivial(fields_cpu.ᶜout4)
    @test compare_cpu_gpu(fields_cpu.ᶜout5, fields.ᶜout5)
    @test !is_trivial(fields_cpu.ᶜout5)
    @test compare_cpu_gpu(fields_cpu.ᶜout6, fields.ᶜout6)
    @test !is_trivial(fields_cpu.ᶜout6)
    @test compare_cpu_gpu(fields_cpu.ᶜout7, fields.ᶜout7)
    @test !is_trivial(fields_cpu.ᶜout7)
    @test compare_cpu_gpu(fields_cpu.ᶜout8, fields.ᶜout8)
    @test !is_trivial(fields_cpu.ᶜout8)
    @test compare_cpu_gpu(fields_cpu.ᶠout1_contra, fields.ᶠout1_contra)
    @test !is_trivial(fields_cpu.ᶠout1_contra)
    @test compare_cpu_gpu(fields_cpu.ᶠout2_contra, fields.ᶠout2_contra)
    @test !is_trivial(fields_cpu.ᶠout2_contra)
    @test compare_cpu_gpu(fields_cpu.ᶜout9, fields.ᶜout9)
    @test !is_trivial(fields_cpu.ᶜout9)
    @test compare_cpu_gpu(fields_cpu.ᶜout10, fields.ᶜout10)
    @test !is_trivial(fields_cpu.ᶜout10)
    @test compare_cpu_gpu(fields_cpu.ᶜout11, fields.ᶜout11)
    @test !is_trivial(fields_cpu.ᶜout11)
    @test compare_cpu_gpu(fields_cpu.ᶜout12, fields.ᶜout12)
    @test !is_trivial(fields_cpu.ᶜout12)
    @test compare_cpu_gpu(fields_cpu.ᶜout13, fields.ᶜout13)
    @test !is_trivial(fields_cpu.ᶜout13)
    @test compare_cpu_gpu(fields_cpu.ᶜout_uₕ, fields.ᶜout_uₕ)
    @test !is_trivial(fields_cpu.ᶜout_uₕ)
    @test compare_cpu_gpu(fields_cpu.ᶠout3_cov, fields.ᶠout3_cov)
    @test !is_trivial(fields_cpu.ᶠout3_cov)
    @test compare_cpu_gpu(fields_cpu.ᶠout4_cov, fields.ᶠout4_cov)
    @test !is_trivial(fields_cpu.ᶠout4_cov)
    @test compare_cpu_gpu(fields_cpu.ᶠout5_cov, fields.ᶠout5_cov)
    @test !is_trivial(fields_cpu.ᶠout5_cov)
end

@testset "Correctness plane" begin
    ᶜspace_cpu = get_space_plane(ClimaComms.CPUSingleThreaded(), Float64)
    ᶠspace_cpu = Spaces.face_space(ᶜspace_cpu)
    fields_cpu = (; get_fields(ᶜspace_cpu)..., get_fields(ᶠspace_cpu)...)
    kernels!(fields_cpu)
    @info "Compiled CPU kernels"

    ᶜspace = get_space_plane(ClimaComms.device(), Float64)
    ClimaComms.device(ᶜspace) isa ClimaComms.CPUSingleThreaded &&
        @warn "Running on the CPU"
    ᶠspace = Spaces.face_space(ᶜspace)
    fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...)
    kernels!(fields)
    @info "Compiled GPU kernels"

    @test compare_cpu_gpu(fields_cpu.ᶜout1, fields.ᶜout1)
    @test !is_trivial(fields_cpu.ᶜout1)
    @test compare_cpu_gpu(fields_cpu.ᶜout2, fields.ᶜout2)
    @test !is_trivial(fields_cpu.ᶜout2)
    @test compare_cpu_gpu(fields_cpu.ᶜout3, fields.ᶜout3)
    @test !is_trivial(fields_cpu.ᶜout3)
    @test compare_cpu_gpu(fields_cpu.ᶜout4, fields.ᶜout4)
    @test !is_trivial(fields_cpu.ᶜout4)
    @test compare_cpu_gpu(fields_cpu.ᶜout5, fields.ᶜout5)
    @test !is_trivial(fields_cpu.ᶜout5)
    @test compare_cpu_gpu(fields_cpu.ᶜout6, fields.ᶜout6)
    @test !is_trivial(fields_cpu.ᶜout6)
    @test compare_cpu_gpu(fields_cpu.ᶜout7, fields.ᶜout7)
    @test !is_trivial(fields_cpu.ᶜout7)
    @test compare_cpu_gpu(fields_cpu.ᶜout8, fields.ᶜout8)
    @test !is_trivial(fields_cpu.ᶜout8)
    @test compare_cpu_gpu(fields_cpu.ᶠout1_contra, fields.ᶠout1_contra)
    @test !is_trivial(fields_cpu.ᶠout1_contra)
    @test compare_cpu_gpu(fields_cpu.ᶠout2_contra, fields.ᶠout2_contra)
    @test !is_trivial(fields_cpu.ᶠout2_contra)
    @test compare_cpu_gpu(fields_cpu.ᶜout9, fields.ᶜout9)
    @test !is_trivial(fields_cpu.ᶜout9)
    @test compare_cpu_gpu(fields_cpu.ᶜout10, fields.ᶜout10)
    @test !is_trivial(fields_cpu.ᶜout10)
    @test compare_cpu_gpu(fields_cpu.ᶜout11, fields.ᶜout11)
    @test !is_trivial(fields_cpu.ᶜout11)
    @test compare_cpu_gpu(fields_cpu.ᶜout12, fields.ᶜout12)
    @test !is_trivial(fields_cpu.ᶜout12)
    @test compare_cpu_gpu(fields_cpu.ᶜout13, fields.ᶜout13)
    @test !is_trivial(fields_cpu.ᶜout13)
    @test compare_cpu_gpu(fields_cpu.ᶜout_uₕ, fields.ᶜout_uₕ)
    @test !is_trivial(fields_cpu.ᶜout_uₕ)
    @test compare_cpu_gpu(fields_cpu.ᶠout3_cov, fields.ᶠout3_cov)
    @test !is_trivial(fields_cpu.ᶠout3_cov)
    @test compare_cpu_gpu(fields_cpu.ᶠout4_cov, fields.ᶠout4_cov)
    @test !is_trivial(fields_cpu.ᶠout4_cov)
    @test compare_cpu_gpu(fields_cpu.ᶠout5_cov, fields.ᶠout5_cov)
    @test !is_trivial(fields_cpu.ᶠout5_cov)
end

@testset "Correctness extruded cubed sphere" begin
    ᶜspace_cpu = get_space_extruded(ClimaComms.CPUSingleThreaded(), Float64)
    ᶠspace_cpu = Spaces.face_space(ᶜspace_cpu)
    fields_cpu = (; get_fields(ᶜspace_cpu)..., get_fields(ᶠspace_cpu)...)
    kernels!(fields_cpu)
    @info "Compiled CPU kernels"

    ᶜspace = get_space_extruded(ClimaComms.device(), Float64)
    ᶠspace = Spaces.face_space(ᶜspace)
    fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...)
    kernels!(fields)
    @info "Compiled GPU kernels"

    @test compare_cpu_gpu(fields_cpu.ᶜout1, fields.ᶜout1)
    @test !is_trivial(fields_cpu.ᶜout1)
    @test compare_cpu_gpu(fields_cpu.ᶜout2, fields.ᶜout2)
    @test !is_trivial(fields_cpu.ᶜout2)
    @test compare_cpu_gpu(fields_cpu.ᶜout3, fields.ᶜout3)
    @test !is_trivial(fields_cpu.ᶜout3)
    @test compare_cpu_gpu(fields_cpu.ᶜout4, fields.ᶜout4)
    @test !is_trivial(fields_cpu.ᶜout4)
    @test compare_cpu_gpu(fields_cpu.ᶜout5, fields.ᶜout5)
    @test !is_trivial(fields_cpu.ᶜout5)
    @test compare_cpu_gpu(fields_cpu.ᶜout6, fields.ᶜout6)
    @test !is_trivial(fields_cpu.ᶜout6)
    @test compare_cpu_gpu(fields_cpu.ᶜout7, fields.ᶜout7)
    @test !is_trivial(fields_cpu.ᶜout7)
    @test compare_cpu_gpu(fields_cpu.ᶜout8, fields.ᶜout8)
    @test !is_trivial(fields_cpu.ᶜout8)
    @test compare_cpu_gpu(fields_cpu.ᶠout1_contra, fields.ᶠout1_contra)
    @test !is_trivial(fields_cpu.ᶠout1_contra)
    @test compare_cpu_gpu(fields_cpu.ᶠout2_contra, fields.ᶠout2_contra)
    @test !is_trivial(fields_cpu.ᶠout2_contra)
    @test compare_cpu_gpu(fields_cpu.ᶜout9, fields.ᶜout9)
    @test !is_trivial(fields_cpu.ᶜout9)
    @test compare_cpu_gpu(fields_cpu.ᶜout10, fields.ᶜout10)
    @test !is_trivial(fields_cpu.ᶜout10)
    @test compare_cpu_gpu(fields_cpu.ᶜout11, fields.ᶜout11)
    @test !is_trivial(fields_cpu.ᶜout11)
    @test compare_cpu_gpu(fields_cpu.ᶜout12, fields.ᶜout12)
    @test !is_trivial(fields_cpu.ᶜout12)
    @test compare_cpu_gpu(fields_cpu.ᶜout13, fields.ᶜout13)
    @test !is_trivial(fields_cpu.ᶜout13)
    @test compare_cpu_gpu(fields_cpu.ᶜout_uₕ, fields.ᶜout_uₕ)
    @test !is_trivial(fields_cpu.ᶜout_uₕ)
    @test compare_cpu_gpu(fields_cpu.ᶠout3_cov, fields.ᶠout3_cov)
    @test !is_trivial(fields_cpu.ᶠout3_cov)
    @test compare_cpu_gpu(fields_cpu.ᶠout4_cov, fields.ᶠout4_cov)
    @test !is_trivial(fields_cpu.ᶠout4_cov)
    @test compare_cpu_gpu(fields_cpu.ᶠout5_cov, fields.ᶠout5_cov)
    @test !is_trivial(fields_cpu.ᶠout5_cov)
end

#! format: on
nothing
