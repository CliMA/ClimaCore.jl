using Test
using ClimaCore:
    Domains, Meshes, Topologies, Geometry, Operators, Spaces, Fields
using ClimaCore.Operators: local_weights, LinearRemap, remap, remap!
using ClimaCore.Topologies: Topology2D
using ClimaCore.Spaces: AbstractSpace, Quadratures
using ClimaCore.DataLayouts: IJFH
using IntervalSets, LinearAlgebra, SparseArrays

FT = Float64

function se_space(domain::Domains.IntervalDomain, nq, nelems = 1)
    quad = Quadratures.GLL{nq}()
    mesh = Meshes.IntervalMesh(domain; nelems = nelems)
    topo = Topologies.IntervalTopology(mesh)
    space = Spaces.SpectralElementSpace1D(topo, quad)
    return space
end

function se_space(domain::Domains.RectangleDomain, nq, nxelems = 1, nyelems = 1)
    quad = Quadratures.GLL{nq}()
    mesh = Meshes.RectilinearMesh(domain, nxelems, nyelems)
    topology = Topologies.Topology2D(mesh)
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

function fv_space(domain::Domains.IntervalDomain, nelems = 1)
    quad = Quadratures.GL{1}()
    mesh = Meshes.IntervalMesh(domain; nelems = nelems)
    topo = Topologies.IntervalTopology(mesh)
    space = Spaces.SpectralElementSpace1D(topo, quad)
    return space
end

function fv_space(domain::Domains.RectangleDomain, nxelems = 1, nyelems = 1)
    quad = Quadratures.GL{1}()
    mesh = Meshes.RectilinearMesh(domain, nxelems, nyelems)
    topology = Topologies.Topology2D(mesh)
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

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
    space = fv_space(domain)

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

@testset "Finite Volume <-> Finite Volume Remapping" begin
    @testset "1D Domains" begin
        @testset "Aligned Intervals Different Resolutions" begin
            domain = Domains.IntervalDomain(
                Geometry.XPoint(0.0) .. Geometry.XPoint(1.0),
                boundary_tags = (:left, :right),
            )

            source = fv_space(domain, 4)
            target = fv_space(domain, 2)

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
            source = fv_space(domain1, 3)

            domain2 = Domains.IntervalDomain(
                Geometry.XPoint(0.0) .. Geometry.XPoint(2.0),
                boundary_tags = (:left, :right),
            )
            target = fv_space(domain2, 3)

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
            source = fv_space(domain1, 4)

            domain2 = Domains.IntervalDomain(
                Geometry.XPoint(-0.5) .. Geometry.XPoint(0.5),
                boundary_tags = (:left, :right),
            )
            target = fv_space(domain2, 4)

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

            source = fv_space(domain, 2, 2)
            target = fv_space(domain, 3, 3)

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
            source = fv_space(domain1, 2, 2)

            domain2 = Domains.RectangleDomain(
                Geometry.XPoint(0.0) .. Geometry.XPoint(2.0),
                Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )
            target = fv_space(domain2, 2, 2)

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
            source = fv_space(domain1, 2, 2)

            domain2 = Domains.RectangleDomain(
                Geometry.XPoint(-0.5) .. Geometry.XPoint(0.5),
                Geometry.YPoint(-0.5) .. Geometry.YPoint(0.5),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )
            target = fv_space(domain2, 2, 2)

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
            space1 = fv_space(domain)

            @test local_weights(space1) ≈ ones(1)

            for nq in [2, 5, 10]
                space2 = se_space(domain, nq)

                _, w = Spaces.Quadratures.quadrature_points(
                    FT,
                    Quadratures.GLL{nq}(),
                )
                @test local_weights(space2) ≈ w ./ 2

                # FV -> SE
                R = LinearRemap(space2, space1)
                R_true = ones(nq, 1)
                @test R.map ≈ R_true
                @test nnz(R.map) == nq

                @test conservative(R)
                @test consistent(R)
                @test monotone(R)

                # SE -> FV
                R = LinearRemap(space1, space2)
                R_true = local_weights(space2)'
                @test R.map ≈ R_true
                @test nnz(R.map) == nq

                @test conservative(R)
                @test consistent(R)
                @test monotone(R)
            end
        end

        @testset "Multiple elements" begin
            space1 = fv_space(domain, 2)
            space2 = se_space(domain, 2, 3)

            # FV -> SE
            R = LinearRemap(space2, space1)
            @test nnz(R.map) == 8
            @test conservative(R)
            @test consistent(R)
            @test monotone(R)

            # SE -> FV
            R = LinearRemap(space1, space2)
            @test nnz(R.map) == 8
            @test conservative(R)
            @test consistent(R)
            @test monotone(R)

            space2 = se_space(domain, 3, 3)

            # FV -> SE
            R = LinearRemap(space2, space1)
            @test nnz(R.map) == 12
            @test conservative(R)
            @test consistent(R)
            @test !monotone(R)

            # SE -> FV
            R = LinearRemap(space1, space2)
            @test nnz(R.map) == 12
            @test conservative(R)
            @test consistent(R)
            @test !monotone(R)
        end
    end
end

@testset "Spectral Elements <-> Spectral Elements Remapping" begin

    @testset "1D Domains" begin
        domain = Domains.IntervalDomain(
            Geometry.XPoint(0.0) .. Geometry.XPoint(1.0),
            boundary_tags = (:left, :right),
        )

        @testset "Single aligned elements" begin

            for nq1 in [3, 5, 9]
                space1 = se_space(domain, nq1)

                _, w = Spaces.Quadratures.quadrature_points(
                    FT,
                    Quadratures.GLL{nq1}(),
                )
                @test local_weights(space1) ≈ w ./ 2

                for nq2 in [2, 4, 10]
                    space2 = se_space(domain, nq2)

                    _, w = Spaces.Quadratures.quadrature_points(
                        FT,
                        Quadratures.GLL{nq2}(),
                    )
                    @test local_weights(space2) ≈ w ./ 2

                    # SE1 -> SE2
                    R = LinearRemap(space2, space1)
                    @test count(!isapprox(0; atol = sqrt(eps(FT))), R.map) ==
                          (max(nq1, nq2) - 2) * min(nq1, nq2) + 2

                    @test conservative(R)
                    @test consistent(R)

                    # SE2 -> SE1
                    R = LinearRemap(space1, space2)
                    @test count(!isapprox(0; atol = sqrt(eps(FT))), R.map) ==
                          (max(nq1, nq2) - 2) * min(nq1, nq2) + 2

                    @test conservative(R)
                    @test consistent(R)
                end
            end
        end

        @testset "Multiple elements" begin
            space1 = se_space(domain, 3, 2)
            space2 = se_space(domain, 2, 3)

            # SE1 -> SE2
            R = LinearRemap(space2, space1)
            @test count(!isapprox(0; atol = sqrt(eps(FT))), R.map) == 22
            @test conservative(R)
            @test consistent(R)

            # SE2 -> SE1
            R = LinearRemap(space1, space2)
            @test count(!isapprox(0; atol = sqrt(eps(FT))), R.map) == 22
            @test conservative(R)
            @test consistent(R)
        end
    end
end
