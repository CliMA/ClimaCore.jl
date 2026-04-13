using ClimaCore
using ClimaCore.CommonSpaces
import ClimaComms
using Test
ClimaComms.@import_required_backends


@testset "kernel renaming" begin
    ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
    @assert !isnothing(ext) # cuda must be loaded to test this extension
    @assert ext.NAME_KERNELS_FROM_STACK_TRACE[]
    space = ExtrudedCubedSphereSpace(Float32;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        staggering = CellCenter(),
    )

    empty!(ext.kernel_names)
    scalar_field = fill(1.0f0, space)
    fill_kernel_name = iterate(ext.kernel_names)[1][2]
    @test occursin("fill", fill_kernel_name)
    @test occursin("src_Fields_Fields", fill_kernel_name)
end
