using Test
using ClimaComms
using IntervalSets

import ClimaCore
import ClimaCore:
    Domains, Meshes, Topologies, Quadratures, Spaces, Fields, Geometry
using ClimaCoreMakie

import CairoMakie
OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))
@show OUTPUT_DIR

@testset "shim delegates to ClimaCoreMakieExt" begin
    @test !isnothing(Base.get_extension(ClimaCore, :ClimaCoreMakieExt))
    @test fieldheatmap === ClimaCore.Visualize.fieldheatmap
    @test fieldheatmap! === ClimaCore.Visualize.fieldheatmap!
    @test fieldcontourf === ClimaCore.Visualize.fieldcontourf
    @test fieldcontourf! === ClimaCore.Visualize.fieldcontourf!
    @test ClimaCoreMakie.fieldline === ClimaCore.Visualize.fieldline
    @test ClimaCoreMakie.fieldline! === ClimaCore.Visualize.fieldline!
    @test ClimaCoreMakie.FieldContourf isa UnionAll
    @test ClimaCoreMakie.FieldHeatmap isa UnionAll
    @test ClimaCoreMakie.FieldLine isa UnionAll
end

@testset "spectral element rectangle 2D" begin
    domain = Domains.RectangleDomain(
        Geometry.XPoint(0) .. Geometry.XPoint(2π),
        Geometry.YPoint(0) .. Geometry.YPoint(2π),
        x1periodic = true,
        x2periodic = true,
    )

    n1, n2 = 2, 2
    Nq = 4
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology =
        Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)
    quad = Quadratures.ClosedUniform{Nq + 1}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

    sinxy = map(coords) do coord
        cos(coord.x + coord.y)
    end

    fig = ClimaCoreMakie.fieldheatmap(sinxy)
    @test fig !== nothing

    fig_png = joinpath(OUTPUT_DIR, "2D_rectangle.png")
    CairoMakie.save(fig_png, fig)
    @test isfile(fig_png)
end
