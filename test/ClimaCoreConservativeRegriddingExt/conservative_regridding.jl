using Test
using ClimaCore
using ClimaCore:
    CommonSpaces, Remapping, Fields, Spaces, RecursiveApply, Meshes, Quadratures
using ConservativeRegridding

const src_space = CommonSpaces.CubedSphereSpace(;
    radius = 10,
    n_quad_points = 3,
    h_elem = 8,
)
const dst_space = CommonSpaces.CubedSphereSpace(;
    radius = 10,
    n_quad_points = 4,
    h_elem = 6,
)

@testset "test get_element_vertices" begin
    vertices = Remapping.get_element_vertices(src_space)
    @test length(vertices) == Meshes.nelements(src_space.grid.topology.mesh)

    # Check that there are 5 vertices per element (quadrilaterals with repeated first vertex)
    @test all(length(vertex) == 5 for vertex in vertices)
    @test all(vertex[1] == vertex[5] for vertex in vertices)
end

@testset "test integrate_each_element" begin
    # Test integrating a field of ones
    ones_field = Fields.ones(src_space)
    integral_each_element =
        Remapping.integrate_each_element(ones_field)
    @test isapprox(sum(integral_each_element), sum(ones_field), atol = 1e-11)

    # Test integrating a field of latitude
    field = Fields.coordinate_field(src_space).lat
    integral_each_element = Remapping.integrate_each_element(field)
    @test isapprox(sum(integral_each_element), sum(field), atol = 1e-12)
end

@testset "test get_value_per_element!" begin
    field = Fields.coordinate_field(src_space).lat
    ones_field = Fields.ones(src_space)
    value_per_element = zeros(Float64, Meshes.nelements(src_space.grid.topology.mesh))
    Remapping.get_value_per_element!(
        value_per_element,
        field,
        ones_field,
    )

    @test isapprox(sum(value_per_element), sum(field), atol = 1e-12)
end

@testset "test set_value_per_element!" begin
    field = Fields.coordinate_field(src_space).lat
    value_per_element = ones(Float64, Meshes.nelements(src_space.grid.topology.mesh))
    Remapping.set_value_per_element!(field, value_per_element)

    @test isapprox(sum(field), sum(value_per_element), atol = 1e-12)
    @test all(field .== 1.0)
end

@testset "test Regridder constructor" begin
    regridder = ConservativeRegridding.Regridder(dst_space, src_space)
    @test regridder isa ConservativeRegridding.Regridder
end

@testset "test regrid!" begin
    src_field = Fields.coordinate_field(src_space).lat
    dst_field = Fields.zeros(dst_space)

    # Test regrid! without pre-allocated buffers
    regridder = ConservativeRegridding.Regridder(dst_space, src_space)
    ConservativeRegridding.regrid!(dst_field, regridder, src_field)
    @test isapprox(sum(dst_field), sum(src_field), atol = 1e-12)

    # Test regrid! with pre-allocated buffers
    value_per_element_src = zeros(Float64, Meshes.nelements(src_space.grid.topology.mesh))
    value_per_element_dst = zeros(Float64, Meshes.nelements(dst_space.grid.topology.mesh))
    ones_src = ones(src_space)
    regridder_tuple = (;
        regridder,
        value_per_element_src,
        value_per_element_dst,
        ones_src,
    )
    ConservativeRegridding.regrid!(dst_field, regridder_tuple, src_field)
    @test isapprox(sum(dst_field), sum(src_field), atol = 1e-12)
end

@testset "test regrid! onto the same space" begin
    src_field = Fields.coordinate_field(src_space).lat
    dst_field = Fields.zeros(src_space)

    # Test regrid! without pre-allocated buffers
    regridder = ConservativeRegridding.Regridder(src_space, src_space)
    ConservativeRegridding.regrid!(dst_field, regridder, src_field)
    @test isapprox(sum(dst_field), sum(src_field), atol = 1e-12)
end

@testset "test regrid! of a constant field" begin
    src_field = ones(src_space)
    dst_field = Fields.zeros(src_space)

    # Test regrid! without pre-allocated buffers
    regridder = ConservativeRegridding.Regridder(src_space, src_space)
    ConservativeRegridding.regrid!(dst_field, regridder, src_field)
    @test isapprox(sum(dst_field), sum(src_field), atol = 1e-12)
end

@testset "test regrid! from source to destination and back" begin
    src_field = Fields.coordinate_field(src_space).lat
    dst_field = Fields.zeros(dst_space)

    # Regrid from source to destination
    regridder = ConservativeRegridding.Regridder(dst_space, src_space)
    ConservativeRegridding.regrid!(dst_field, regridder, src_field)
    @test isapprox(sum(dst_field), sum(src_field), atol = 1e-12)

    # Regrid from destination to source using the transpose of the regridder
    ConservativeRegridding.regrid!(src_field, transpose(regridder), dst_field)
    @test isapprox(sum(src_field), sum(dst_field), atol = 1e-12)
end
