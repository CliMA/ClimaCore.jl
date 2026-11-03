#=
julia --project=.buildkite
using Revise; include(joinpath("test", "gpu", "local_geometry_loads.jl"))
=#
using Test
import CUDA
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: CommonSpaces, DataLayouts, Fields, Geometry, Operators, Spaces

@testset "every local geometry value is loaded once per kernel" begin
    FT = Float32
    space = CommonSpaces.ExtrudedCubedSphereSpace(
        FT;
        z_elem = 4,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 2,
        n_quad_points = 4,
        staggering = CommonSpaces.CellCenter(),
    )
    u = fill(Geometry.Covariant12Vector(FT(1), FT(2)), space)
    v = fill(Geometry.Covariant12Vector(FT(3), FT(4)), space)
    div_u = similar(u, FT)
    div_v = similar(v, FT)
    div = Operators.Divergence()
    # Both divergences read the Jacobian and the metric terms of every point, with
    # barriers between them that keep LLVM from reusing ordinary loads.
    ir = sprint() do io
        CUDA.@device_code_llvm io = io debuginfo = :none DataLayouts.foreach_slab(
            div_u,
            div_v,
            u,
            v,
        ) do div_u, div_v, u, v
            @. div_u = div(u)
            @. div_v = div(v)
        end
    end
    # The pointer operands of the global memory loads of every compiled kernel
    load = r"= load [^,]+, (?:ptr addrspace\(1\)|[^,]+ addrspace\(1\)\*) (%[\w.]+)"
    kernels = map(split(ir, r"(?m)^define "; keepempty = false)[2:end]) do kernel
        map(m -> m[1], eachmatch(load, kernel))
    end
    @test !isempty(kernels)
    for pointers in kernels
        @test !isempty(pointers)
        @test allunique(pointers)
    end
end

@testset "host code reads the local geometry of GPU spaces like CuArrays" begin
    FT = Float32
    space = CommonSpaces.ExtrudedCubedSphereSpace(
        FT;
        z_elem = 4,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 2,
        n_quad_points = 4,
        staggering = CommonSpaces.CellCenter(),
    )
    @test parent(Spaces.local_geometry_data(space)) isa
          DataLayouts.InvariantArray{FT, <:Any, <:CUDA.CuArray}
    z = Fields.coordinate_field(space).z
    z_copy = copy(z) # a field of a CuArray, without the wrapper
    @test !(parent(z_copy) isa DataLayouts.InvariantArray)
    # Copies of a reshaped view of the geometry (as in ClimaAtmos initial conditions),
    # reductions of the geometry (as in ClimaLand domains), and host reductions,
    # collections, comparisons (also with CuArrays), and broadcasts of views of the
    # geometry, and copies of them into views of CuArrays (as in RRTMGP inputs)
    @test Array(Fields.field2array(z)) == Array(Fields.field2array(z_copy))
    @test extrema(z) == extrema(z_copy)
    @test maximum(parent(z)) == maximum(parent(z_copy))
    @test collect(parent(z)) == Array(parent(z_copy))
    @test Array(parent(z) .+ 1) == Array(parent(z_copy) .+ 1)
    @test view(parent(z), :, 1, 1, 1, 1) == view(parent(z), :, 1, 1, 1, 1)
    @test parent(z) == parent(z_copy) && parent(z_copy) == parent(z)
    @test isequal(parent(z), parent(z_copy)) && isequal(parent(z_copy), parent(z))
    @test view(parent(z), :, 1, 1, 1, 1) == view(parent(z_copy), :, 1, 1, 1, 1)
    @test Fields.field_values(z) == Fields.field_values(z)
    z_column = view(similar(parent(z_copy), size(parent(z_copy), 1), 2), :, 1:1)
    copyto!(z_column, reshape(view(parent(z), :, 1, 1, 1, 1), :, 1))
    @test Array(z_column) == Array(reshape(view(parent(z_copy), :, 1, 1, 1, 1), :, 1))
end
