using Test
using ClimaCore:
    Domains, Meshes, Topologies, Geometry, Operators, Spaces, Fields
using ClimaCore.Operators: local_weights, LinearRemap, remap, remap!
using ClimaCore.Topologies: Topology2D
using ClimaCore.Spaces: AbstractSpace
using ClimaCore.DataLayouts: IJFH
using IntervalSets, LinearAlgebra, SparseArrays

FT = Float64

"""
Checks if a linear remapping operator is conservative.

A linear operator `R` is conservative the local weight of each element in the
source mesh is distributed conservatively amongst the target mesh elements.
"""
function conservative(R::LinearRemap)
    Jt = local_weights(R.target)
    Js = local_weights(R.source)
    return (R.map)' * Jt ≈ Js
end

"""
Checks if a linear remapping operator is consistent.

A linear operator `R` is consistent if all its rows sum to one.
"""
function consistent(R::LinearRemap)
    row_sums = sum(R.map, dims = 2)
    v = ones(size(row_sums))
    return row_sums ≈ v
end

"""
Checks if a linear remapping operator is monotone.

A linear operator `R` is monotone if it is consistent and each element
`R_{ij} ≥ 0`.
"""
function monotone(R::LinearRemap)
    consistent(R) && all(i -> i >= 0, R.map)
end

@testset "Linear Operator Properties" begin
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        Geometry.YPoint{FT}(-2) .. Geometry.YPoint{FT}(8),
        x1boundary = (:bottom, :top),
        x2boundary = (:left, :right),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    topology = Topologies.Topology2D(mesh)
    space = Spaces.SpectralElementSpace2D(topology, Spaces.Quadratures.GL{1}())

    op = [0.5 0.25 0.0 0.25; 0.0 1.0 0.0 0.0; 0.25 0.25 0.25 0.25]
    R = LinearRemap(space, space, op)
    @test consistent(R)
    @test monotone(R)

    op = [0.5 0.25 0.0 0.25; 1.0 1.0 0.0 -1.0; 0.25 0.25 0.25 0.25]
    R = LinearRemap(space, space, op)
    @test consistent(R)
    @test !monotone(R)

    op = [0.5 0.25 0.0 0.25; 0.0 2.0 0.0 0.0; 0.25 0.25 0.25 0.25]
    R = LinearRemap(space, space, op)
    @test !consistent(R)
    @test !monotone(R)
end

@testset "Finite Volume Remapping" begin
    quad = Spaces.Quadratures.GL{1}() # FV specification

    @testset "1D Domains" begin
        @testset "Aligned Intervals Different Resolutions" begin
            domain = Domains.IntervalDomain(
                Geometry.XPoint(0.0) .. Geometry.XPoint(1.0),
                boundary_tags = (:left, :right),
            )

            n1 = 4
            mesh1 = Meshes.IntervalMesh(domain; nelems = n1)
            source_topo = Topologies.IntervalTopology(mesh1)
            source = Spaces.SpectralElementSpace1D(source_topo, quad)

            n2 = 2
            mesh2 = Meshes.IntervalMesh(domain; nelems = n2)
            target_topo = Topologies.IntervalTopology(mesh2)
            target = Spaces.SpectralElementSpace1D(target_topo, quad)

            @test local_weights(source) ≈ 0.25 * ones(4, 1)
            @test local_weights(target) ≈ 0.5 * ones(2, 1)

            # create remapping operator from 4-elem to 2-elem interval
            R = LinearRemap(target, source)
            R_true = [
                0.5 0.5 0.0 0.0
                0.0 0.0 0.5 0.5
            ]

            @test R.map ≈ R_true
            @test nnz(R.map) == 4 # check R is sparse

            @test conservative(R)
            @test consistent(R)
            @test monotone(R)

            @testset "Scalar Remap Operator Application" begin
                n = length(source.local_geometry)
                source_field =
                    Fields.Field(IJFH{FT, 1}(ones(1, 1, n, 1)), source)

                # test consistent remap
                target_field = remap(R, source_field)
                @test vec(parent(target_field)) ≈
                      ones(length(target.local_geometry))

                # test simple remap
                vec(parent(source_field)) .= [1.0; 2.0; 3.0; 4.0]
                remap!(target_field, R, source_field)
                @test vec(parent(target_field)) ≈ [1.5; 3.5]
            end
        end

        @testset "Unaligned Intervals Same Resolution" begin
            domain1 = Domains.IntervalDomain(
                Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
                boundary_tags = (:left, :right),
            )

            n1 = 3
            mesh1 = Meshes.IntervalMesh(domain1; nelems = n1)
            source_topo = Topologies.IntervalTopology(mesh1)
            source = Spaces.SpectralElementSpace1D(source_topo, quad)

            domain2 = Domains.IntervalDomain(
                Geometry.XPoint(0.0) .. Geometry.XPoint(2.0),
                boundary_tags = (:left, :right),
            )
            n2 = 3
            mesh2 = Meshes.IntervalMesh(domain2; nelems = n2)
            target_topo = Topologies.IntervalTopology(mesh2)
            target = Spaces.SpectralElementSpace1D(target_topo, quad)

            R = LinearRemap(target, source)
            R_true = spzeros(FT, 3, 3)
            R_true[1, 2] = 0.5
            R_true[1, 3] = 0.5
            R_true[2, 3] = 0.5

            @test R.map ≈ R_true
            @test nnz(R.map) == 3

            @test !conservative(R)
            @test !consistent(R)
            @test !monotone(R)
        end

        @testset "Concentric Domains of Different Length" begin
            domain1 = Domains.IntervalDomain(
                Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
                boundary_tags = (:left, :right),
            )

            n1 = 4
            mesh1 = Meshes.IntervalMesh(domain1; nelems = n1)
            source_topo = Topologies.IntervalTopology(mesh1)
            source = Spaces.SpectralElementSpace1D(source_topo, quad)


            domain2 = Domains.IntervalDomain(
                Geometry.XPoint(-0.5) .. Geometry.XPoint(0.5),
                boundary_tags = (:left, :right),
            )
            n2 = 4
            mesh2 = Meshes.IntervalMesh(domain2; nelems = n2)
            target_topo = Topologies.IntervalTopology(mesh2)
            target = Spaces.SpectralElementSpace1D(target_topo, quad)

            R = LinearRemap(target, source)
            R_true = spzeros(4, 4)
            R_true[1, 2] = 1.0
            R_true[2, 2] = 1.0
            R_true[3, 3] = 1.0
            R_true[4, 3] = 1.0

            @test R.map ≈ R_true
            @test nnz(R.map) == 4

            @test !conservative(R)
            @test consistent(R)
            @test monotone(R)

            @testset "Scalar Remap Operator Application" begin
                n = length(source.local_geometry)
                source_field =
                    Fields.Field(IJFH{FT, 1}(ones(1, 1, n, 1)), source)

                # test consistent remap
                target_field = remap(R, source_field)
                @test vec(parent(target_field)) ≈
                      ones(length(target.local_geometry))

                # test simple remap
                vec(parent(source_field)) .= [1.0; 2.0; 3.0; 4.0]
                remap!(target_field, R, source_field)
                @test vec(parent(target_field)) ≈ [2.0; 2.0; 3.0; 3.0]
            end
        end
    end

    @testset "2D Domains" begin
        @testset "Aligned Grids Different Resolutions" begin
            domain = Domains.RectangleDomain(
                Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
                Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )

            m1, n1 = 2, 2
            mesh1 = Meshes.RectilinearMesh(domain, m1, n1)
            source_topo = Topologies.Topology2D(mesh1)
            source = Spaces.SpectralElementSpace2D(source_topo, quad)

            m2, n2 = 3, 3
            mesh2 = Meshes.RectilinearMesh(domain, m2, n2)
            target_topo = Topologies.Topology2D(mesh2)
            target = Spaces.SpectralElementSpace2D(target_topo, quad)

            @test local_weights(source) ≈ ones(4, 1)
            @test local_weights(target) ≈ (2 / 3)^2 * ones(9, 1)

            # create remapping operator from 2x2 to 3x3 grid
            R = LinearRemap(target, source)
            R_true = [
                1.0 0.0 0.0 0.0
                0.5 0.5 0.0 0.0
                0.0 1.0 0.0 0.0
                0.5 0.0 0.5 0.0
                0.25 0.25 0.25 0.25
                0.0 0.5 0.0 0.5
                0.0 0.0 1.0 0.0
                0.0 0.0 0.5 0.5
                0.0 0.0 0.0 1.0
            ]

            @test R.map ≈ R_true
            @test nnz(R.map) == 16 # check R is sparse

            @test conservative(R)
            @test consistent(R)
            @test monotone(R)

            @testset "Scalar Remap Operator Application" begin
                n = length(source.local_geometry)
                source_field =
                    Fields.Field(IJFH{FT, 1}(ones(1, 1, n, 1)), source)

                # test consistent remap
                target_field = remap(R, source_field)
                @test vec(parent(target_field)) ≈
                      ones(length(target.local_geometry))

                # test simple remap
                vec(parent(source_field)) .= [1.0; 2.0; 3.0; 4.0]
                remap!(target_field, R, source_field)
                @test vec(parent(target_field)) ≈
                      [1.0; 1.5; 2.0; 2.0; 2.5; 3.0; 3.0; 3.5; 4.0]
            end
        end

        @testset "Unaligned Grids Same Resolution" begin
            domain1 = Domains.RectangleDomain(
                Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
                Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )

            m1, n1 = 2, 2
            mesh1 = Meshes.RectilinearMesh(domain1, m1, n1)
            source_topo = Topologies.Topology2D(mesh1)
            source = Spaces.SpectralElementSpace2D(source_topo, quad)

            domain2 = Domains.RectangleDomain(
                Geometry.XPoint(0.0) .. Geometry.XPoint(2.0),
                Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )
            m2, n2 = 2, 2
            mesh2 = Meshes.RectilinearMesh(domain2, m2, n2)
            target_topo = Topologies.Topology2D(mesh2)
            target = Spaces.SpectralElementSpace2D(target_topo, quad)

            R = LinearRemap(target, source)
            R_true = spzeros(FT, 4, 4)
            R_true[1, 2] = 1.0
            R_true[3, 4] = 1.0

            @test R.map ≈ R_true
            @test nnz(R.map) == 2

            @test !conservative(R)
            @test !consistent(R)
            @test !monotone(R)
        end

        @testset "Concentric Domains of Different Area" begin
            domain1 = Domains.RectangleDomain(
                Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
                Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )

            m1, n1 = 2, 2
            mesh1 = Meshes.RectilinearMesh(domain1, m1, n1)
            source_topo = Topologies.Topology2D(mesh1)
            source = Spaces.SpectralElementSpace2D(source_topo, quad)

            domain2 = Domains.RectangleDomain(
                Geometry.XPoint(-0.5) .. Geometry.XPoint(0.5),
                Geometry.YPoint(-0.5) .. Geometry.YPoint(0.5),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )
            m2, n2 = 2, 2
            mesh2 = Meshes.RectilinearMesh(domain2, m2, n2)
            target_topo = Topologies.Topology2D(mesh2)
            target = Spaces.SpectralElementSpace2D(target_topo, quad)

            R = LinearRemap(target, source)
            R_true = I

            @test R.map ≈ R_true
            @test nnz(R.map) == 4

            @test !conservative(R)
            @test consistent(R)
            @test monotone(R)

            @testset "Scalar Remap Operator Application" begin
                n = length(source.local_geometry)
                source_field =
                    Fields.Field(IJFH{FT, 1}(ones(1, 1, n, 1)), source)

                # test consistent remap
                target_field = remap(R, source_field)
                @test vec(parent(target_field)) ≈
                      ones(length(target.local_geometry))

                # test simple remap
                vec(parent(source_field)) .= [1.0; 2.0; 3.0; 4.0]
                remap!(target_field, R, source_field)
                @test vec(parent(target_field)) ≈ [1.0; 2.0; 3.0; 4.0]
            end
        end
    end
end

@testset "Finite Volume <-> Spectral Elements Remapping" begin

    @testset "1D Domains" begin
        domain = Domains.IntervalDomain(
            Geometry.XPoint(0.0) .. Geometry.XPoint(1.0),
            boundary_tags = (:left, :right),
        )

        @testset "Single aligned elements" begin
            quad1 = Spaces.Quadratures.GL{1}() # FV specification
            n1 = 1
            mesh1 = Meshes.IntervalMesh(domain; nelems = n1)
            topo1 = Topologies.IntervalTopology(mesh1)
            fv_space = Spaces.SpectralElementSpace1D(topo1, quad1)

            @test local_weights(fv_space) ≈ (1 / n1) * ones(n1, 1)

            for nq2 in [2, 5, 10]

                quad2 = Spaces.Quadratures.GLL{nq2}() # Spectral
                n2 = 1
                mesh2 = Meshes.IntervalMesh(domain; nelems = n2)
                topo2 = Topologies.IntervalTopology(mesh2)
                se_space = Spaces.SpectralElementSpace1D(topo2, quad2)

                _, w = Spaces.Quadratures.quadrature_points(FT, quad2)
                @test local_weights(se_space) ≈ w ./ 2

                # FV -> SE
                R = LinearRemap(se_space, fv_space)
                R_true = ones(nq2, 1)
                @test R.map ≈ R_true
                @test nnz(R.map) == nq2

                @test conservative(R)
                @test consistent(R)
                @test monotone(R)

                # SE -> FV
                R = LinearRemap(fv_space, se_space)
                R_true = local_weights(se_space)'
                @test R.map ≈ R_true
                @test nnz(R.map) == nq2

                @test conservative(R)
                @test consistent(R)
                @test monotone(R)
            end
        end

        @testset "Multiple elements" begin
            quad1 = Spaces.Quadratures.GL{1}() # FV specification
            n1 = 2
            mesh1 = Meshes.IntervalMesh(domain; nelems = n1)
            topo1 = Topologies.IntervalTopology(mesh1)
            fv_space = Spaces.SpectralElementSpace1D(topo1, quad1)

            nq2 = 2
            quad2 = Spaces.Quadratures.GLL{nq2}() # Spectral
            n2 = 3
            mesh2 = Meshes.IntervalMesh(domain; nelems = n2)
            topo2 = Topologies.IntervalTopology(mesh2)
            se_space = Spaces.SpectralElementSpace1D(topo2, quad2)

            # FV -> SE
            R = LinearRemap(se_space, fv_space)
            @test nnz(R.map) == 8
            @test conservative(R)
            @test consistent(R)
            @test monotone(R)

            # SE -> FV
            R = LinearRemap(fv_space, se_space)
            @test nnz(R.map) == 8
            @test conservative(R)
            @test consistent(R)
            @test monotone(R)

            nq2 = 3
            quad2 = Spaces.Quadratures.GLL{nq2}() # Spectral
            n2 = 3
            mesh2 = Meshes.IntervalMesh(domain; nelems = n2)
            topo2 = Topologies.IntervalTopology(mesh2)
            se_space = Spaces.SpectralElementSpace1D(topo2, quad2)

            # FV -> SE
            R = LinearRemap(se_space, fv_space)
            @test nnz(R.map) == 12
            @test conservative(R)
            @test consistent(R)
            @test !monotone(R)

            # SE -> FV
            R = LinearRemap(fv_space, se_space)
            @test nnz(R.map) == 12
            @test conservative(R)
            @test consistent(R)
            @test !monotone(R)
        end
    end
end
