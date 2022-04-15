using Test
using StaticArrays, IntervalSets, LinearAlgebra

import ClimaCore: slab, Domains, Meshes, Topologies, Spaces, Fields

import ClimaCore.Geometry: Geometry
import ClimaCore.DataLayouts: IJFH

@testset "1d domain space" begin
    FT = Float64
    domain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        periodic = true,
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 1)
    topology = Topologies.IntervalTopology(mesh)

    quad = Spaces.Quadratures.GLL{4}()
    points, weights = Spaces.Quadratures.quadrature_points(FT, quad)

    space = Spaces.SpectralElementSpace1D(topology, quad)

    coord_data = Spaces.coordinates_data(space)
    @test eltype(coord_data) == Geometry.XPoint{Float64}

    array = parent(Spaces.coordinates_data(space))
    @test size(array) == (4, 1, 1)
    coord_slab = slab(Spaces.coordinates_data(space), 1)
    @test coord_slab[1] == Geometry.XPoint{FT}(-3)
    @test coord_slab[4] == Geometry.XPoint{FT}(5)

    local_geometry_slab = slab(space.local_geometry, 1)
    dss_weights_slab = slab(space.dss_weights, 1)

    for i in 1:4
        @test Geometry.components(local_geometry_slab[i].∂x∂ξ) ≈
              @SMatrix [8 / 2]
        @test Geometry.components(local_geometry_slab[i].∂ξ∂x) ≈
              @SMatrix [2 / 8]
        @test local_geometry_slab[i].J ≈ (8 / 2)
        @test local_geometry_slab[i].WJ ≈ (8 / 2) * weights[i]
        if i in (1, 4)
            @test dss_weights_slab[i] ≈ 1 / 2
        else
            @test dss_weights_slab[i] ≈ 1
        end
    end

end


@testset "1×1 domain space" begin
    FT = Float32
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        Geometry.YPoint{FT}(-2) .. Geometry.YPoint{FT}(8),
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    grid_topology = Topologies.Topology2D(mesh)

    quad = Spaces.Quadratures.GLL{4}()
    points, weights = Spaces.Quadratures.quadrature_points(FT, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    array = parent(Spaces.coordinates_data(space))
    @test size(array) == (4, 4, 2, 1)
    coord_slab = slab(Spaces.coordinates_data(space), 1)
    @test coord_slab[1, 1] ≈ Geometry.XYPoint{FT}(-3.0, -2.0)
    @test coord_slab[4, 1] ≈ Geometry.XYPoint{FT}(5.0, -2.0)
    @test coord_slab[1, 4] ≈ Geometry.XYPoint{FT}(-3.0, 8.0)
    @test coord_slab[4, 4] ≈ Geometry.XYPoint{FT}(5.0, 8.0)

    local_geometry_slab = slab(space.local_geometry, 1)
    dss_weights_slab = slab(space.local_dss_weights, 1)


    for i in 1:4, j in 1:4
        @test Geometry.components(local_geometry_slab[i, j].∂x∂ξ) ≈
              @SMatrix [8/2 0; 0 10/2]
        @test Geometry.components(local_geometry_slab[i, j].∂ξ∂x) ≈
              @SMatrix [2/8 0; 0 2/10]
        @test local_geometry_slab[i, j].J ≈ (10 / 2) * (8 / 2)
        @test local_geometry_slab[i, j].WJ ≈
              (10 / 2) * (8 / 2) * weights[i] * weights[j]
        if i in (1, 4)
            @test dss_weights_slab[i, j] ≈ 1 / 2
        else
            @test dss_weights_slab[i, j] ≈ 1
        end
    end

    @test length(space.boundary_surface_geometries) == 2
    @test keys(space.boundary_surface_geometries) == (:south, :north)
    @test sum(parent(space.boundary_surface_geometries.north.sWJ)) ≈ 8
    @test parent(space.boundary_surface_geometries.north.normal)[1, :, 1] ≈
          [0.0, 1.0]
end

#=
@testset "dss on 2×2 rectangular mesh (unstructured)" begin
    FT = Float64
    n1, n2 = 2, 2
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(0) .. Geometry.XPoint{FT}(4),
        Geometry.YPoint{FT}(0) .. Geometry.YPoint{FT}(4),
        x1periodic = false,
        x2periodic = false,
        x1boundary = (:west, :east),
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = Topologies.Topology2D(mesh)

    quad = Spaces.Quadratures.GLL{4}()
    points, weights = Spaces.Quadratures.quadrature_points(FT, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    array = parent(Spaces.coordinates_data(space))
    @test size(array) == (4, 4, 2, 4)

    Nij = length(points)
    field = Fields.Field(IJFH{FT, Nij}(ones(Nij, Nij, 1, n1 * n2)), space)
    field_values = Fields.field_values(field)
    Spaces.horizontal_dss!(field)

    @testset "dss should not modify interior degrees of freedom of any element" begin
        result = true
        for el in 1:(n1 * n2)
            slb = slab(field_values, 1, el)
            for i in 2:(Nij - 1), j in 2:(Nij - 1)
                if slb[i, j] ≠ 1
                    result = false
                end
            end
        end
        @test result
    end
    s1 = slab(field_values, 1, 1)
    s2 = slab(field_values, 1, 2)
    s3 = slab(field_values, 1, 3)
    s4 = slab(field_values, 1, 4)

    @testset "vertex common to all (4) elements" begin
        @test (s1[Nij, Nij] == s2[1, Nij] == s3[Nij, 1] == s4[1, 1])
    end

    @testset "vertices common to (2) elements" begin
        @test s1[Nij, 1] == s2[1, 1]
        @test s1[1, Nij] == s3[1, 1]
        @test s2[Nij, Nij] == s4[Nij, 1]
        @test s3[Nij, Nij] == s4[1, Nij]
    end

    @testset "boundary faces" begin
        for fc in 2:(Nij - 1)
            @test s1[1, fc] == 1 # element 1 face 1
            @test s1[fc, 1] == 1 # element 1 face 3
            @test s2[Nij, fc] == 1 # element 2 face 2
            @test s2[fc, 1] == 1 # element 2 face 3
            @test s3[1, fc] == 1 # element 3 face 1
            @test s3[fc, Nij] == 1 # element 3 face 4
            @test s4[Nij, fc] == 1 # element 4 face 2
            @test s4[fc, Nij] == 1 # element 4 face 4
        end
    end

    @testset "interior faces" begin
        for fc in 2:(Nij - 1)
            @test (s1[Nij, fc] == s2[1, fc] == 2) # (e1, f2) == (e2, f1) == 2
            @test (s1[fc, Nij] == s3[fc, 1] == 2) # (e1, f4) == (e3, f3) == 2
            @test (s2[fc, Nij] == s4[fc, 1] == 2) # (e2, f4) == (e4, f3) == 2
            @test (s3[Nij, fc] == s4[1, fc] == 2) # (e3, f2) == (e4, f1) == 2
        end
    end
end


@testset "dss on 2×2 rectangular mesh" begin
    FT = Float64
    n1, n2 = 2, 2
    Nij = 4
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(0) .. Geometry.XPoint{FT}(4),
        Geometry.YPoint{FT}(0) .. Geometry.YPoint{FT}(4),
        x1periodic = false,
        x2periodic = false,
        x1boundary = (:west, :east),
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = Topologies.Topology2D(mesh)

    quad = Spaces.Quadratures.GLL{Nij}()
    points, weights = Spaces.Quadratures.quadrature_points(FT, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    array = parent(Spaces.coordinates_data(space))
    @test size(array) == (Nij, Nij, 2, n1 * n2)

    data = zeros(Nij, Nij, 3, n1 * n2)
    data[:, :, 1, :] .= 1:Nij
    data[:, :, 2, :] .= (1:Nij)'
    data[:, :, 3, :] .= reshape(1:(n1 * n2), 1, 1, :)
    field = Fields.Field(IJFH{Tuple{FT, FT, FT}, Nij}(data), space)
    field_dss = Spaces.horizontal_dss!(copy(field))
    data_dss = parent(field_dss)

    @testset "slab 1" begin
        @test data_dss[1:(Nij - 1), 1:(Nij - 1), :, 1] ==
              data[1:(Nij - 1), 1:(Nij - 1), :, 1]
        @test data_dss[Nij, 1:(Nij - 1), :, 1] ==
              data[Nij, 1:(Nij - 1), :, 1] .+ data[1, 1:(Nij - 1), :, 2]
        @test data_dss[1:(Nij - 1), Nij, :, 1] ==
              data[1:(Nij - 1), Nij, :, 1] .+ data[1:(Nij - 1), 1, :, 3]
        @test data_dss[Nij, Nij, :, 1] ==
              data[Nij, Nij, :, 1] .+ data[1, Nij, :, 2] .+
              data[Nij, 1, :, 3] .+ data[1, 1, :, 4]
    end

    @testset "slab 2" begin
        @test data_dss[2:Nij, 1:(Nij - 1), :, 2] ==
              data[2:Nij, 1:(Nij - 1), :, 2]
        @test data_dss[1, 1:(Nij - 1), :, 2] ==
              data[Nij, 1:(Nij - 1), :, 1] .+ data[1, 1:(Nij - 1), :, 2]
        @test data_dss[2:Nij, Nij, :, 2] ==
              data[2:Nij, Nij, :, 2] .+ data[2:Nij, 1, :, 4]
        @test data_dss[1, Nij, :, 2] ==
              data[Nij, Nij, :, 1] .+ data[1, Nij, :, 2] .+
              data[Nij, 1, :, 3] .+ data[1, 1, :, 4]
    end

    @testset "slab 3" begin
        @test data_dss[1:(Nij - 1), 2:Nij, :, 3] ==
              data[1:(Nij - 1), 2:Nij, :, 3]
        @test data_dss[Nij, 2:Nij, :, 3] ==
              data[Nij, 2:Nij, :, 3] .+ data[1, 2:Nij, :, 4]
        @test data_dss[1:(Nij - 1), 1, :, 3] ==
              data[1:(Nij - 1), Nij, :, 1] .+ data[1:(Nij - 1), 1, :, 3]
        @test data_dss[Nij, 1, :, 3] ==
              data[Nij, Nij, :, 1] .+ data[1, Nij, :, 2] .+
              data[Nij, 1, :, 3] .+ data[1, 1, :, 4]
    end

    @testset "slab 3" begin
        @test data_dss[2:Nij, 2:Nij, :, 4] == data[2:Nij, 2:Nij, :, 4]
        @test data_dss[1, 2:Nij, :, 4] ==
              data[Nij, 2:Nij, :, 3] .+ data[1, 2:Nij, :, 4]
        @test data_dss[2:Nij, 1, :, 4] ==
              data[2:Nij, Nij, :, 2] .+ data[2:Nij, 1, :, 4]
        @test data_dss[1, 1, :, 4] ==
              data[Nij, Nij, :, 1] .+ data[1, Nij, :, 2] .+
              data[Nij, 1, :, 3] .+ data[1, 1, :, 4]
    end
end
=#
