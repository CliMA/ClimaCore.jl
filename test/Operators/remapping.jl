using Test
using ClimaComms
using ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Geometry,
    Operators,
    Spaces,
    Fields,
    Quadratures
using ClimaCore.Operators: local_weights, LinearRemap, remap, remap!
using ClimaCore.Topologies: Topology2D
using ClimaCore.Spaces: AbstractSpace
using ClimaCore.DataLayouts: IJFH
using IntervalSets, LinearAlgebra, SparseArrays

FT = Float64

function make_space(domain::Domains.IntervalDomain, nq, nelems = 1)
    nq == 1 ? (quad = Quadratures.GL{1}()) : (quad = Quadratures.GLL{nq}())
    mesh = Meshes.IntervalMesh(domain; nelems = nelems)
    topo = Topologies.IntervalTopology(mesh)
    space = Spaces.SpectralElementSpace1D(topo, quad)
    return space
end

function make_space(
    domain::Domains.RectangleDomain,
    nq,
    nxelems = 1,
    nyelems = 1,
)
    nq == 1 ? (quad = Quadratures.GL{1}()) : (quad = Quadratures.GLL{nq}())
    mesh = Meshes.RectilinearMesh(domain, nxelems, nyelems)
    topology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
        mesh,
    )
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

function sparsity_pattern(R::SparseMatrixCSC)
    droptol!(R, sqrt(eps(eltype(R))))
    I, J, _ = findnz(R)
    return I, J
end

function sparsity_pattern(R::LinearRemap)
    sparsity_pattern(R.map)
end

"""
Checks if a linear remapping operator is conservative.

A linear operator `R` is conservative the local weight of each element in the
source mesh is distributed conservatively amongst the target mesh elements.

See [Ullrich2015](@cite) eq. 9.
"""
function conservative(R::LinearRemap)
    Jt = local_weights(R.target)
    Js = local_weights(R.source)
    return (R.map)' * Jt ≈ Js
end

"""
Checks if a linear remapping operator is consistent.

A linear operator `R` is consistent if all its rows sum to one.

See [Ullrich2015](@cite) eq. 12.
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

See [Ullrich2015](@cite) eq. 14.
"""
function monotone(R::LinearRemap)
    consistent(R) && all(i -> i >= 0, R.map)
end

function operator_size(R)
    QS_t = Spaces.quadrature_style(R.target)
    QS_s = Spaces.quadrature_style(R.source)
    Nq_t = Quadratures.degrees_of_freedom(QS_t)
    Nq_s = Quadratures.degrees_of_freedom(QS_s)
    nelems_s = Topologies.nlocalelems(Spaces.topology(R.source))
    nelems_t = Topologies.nlocalelems(Spaces.topology(R.target))
    return (Nq_t^2 * nelems_t, Nq_s^2 * nelems_s)
end

function test_identity(space)
    R = LinearRemap(space, space)
    @test R.map ≈ I
end

@testset "Linear Operator Properties" begin
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-3) .. Geometry.XPoint{FT}(5),
        Geometry.YPoint{FT}(-2) .. Geometry.YPoint{FT}(8),
        x1boundary = (:bottom, :top),
        x2boundary = (:left, :right),
    )
    space = make_space(domain, 1)

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

@testset "Identity Operator" begin
    domain1D = Domains.IntervalDomain(
        Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
        boundary_tags = (:left, :right),
    )
    domain2D = Domains.RectangleDomain(
        Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
        Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
        x1boundary = (:bottom, :top),
        x2boundary = (:left, :right),
    )
    test_identity(make_space(domain1D, 1, 1))
    test_identity(make_space(domain1D, 1, 8))
    test_identity(make_space(domain2D, 1, 5, 5))
    test_identity(make_space(domain2D, 1, 5, 8))

    test_identity(make_space(domain1D, 3, 1))
    test_identity(make_space(domain1D, 3, 5))
    test_identity(make_space(domain2D, 5, 5, 5))
    test_identity(make_space(domain2D, 5, 8, 3))
end

@testset "Finite Volume <-> Finite Volume Remapping" begin
    @testset "1D Domains" begin
        @testset "Aligned Intervals Different Resolutions" begin
            domain = Domains.IntervalDomain(
                Geometry.XPoint(0.0) .. Geometry.XPoint(1.0),
                boundary_tags = (:left, :right),
            )

            source = make_space(domain, 1, 4)
            target = make_space(domain, 1, 2)

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
                n = length(Spaces.local_geometry_data(source))
                source_field =
                    Fields.Field(IJFH{FT, 1}(ones(1, 1, n, 1)), source)

                # test consistent remap
                target_field = remap(R, source_field)
                @test vec(parent(target_field)) ≈
                      ones(length(Spaces.local_geometry_data(target)))

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
            source = make_space(domain1, 1, 3)

            domain2 = Domains.IntervalDomain(
                Geometry.XPoint(0.0) .. Geometry.XPoint(2.0),
                boundary_tags = (:left, :right),
            )
            target = make_space(domain2, 1, 3)

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
            source = make_space(domain1, 1, 4)

            domain2 = Domains.IntervalDomain(
                Geometry.XPoint(-0.5) .. Geometry.XPoint(0.5),
                boundary_tags = (:left, :right),
            )
            target = make_space(domain2, 1, 4)

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
                n = length(Spaces.local_geometry_data(source))
                source_field =
                    Fields.Field(IJFH{FT, 1}(ones(1, 1, n, 1)), source)

                # test consistent remap
                target_field = remap(R, source_field)
                @test vec(parent(target_field)) ≈
                      ones(length(Spaces.local_geometry_data(target)))

                # test simple remap
                vec(parent(source_field)) .= [1.0; 2.0; 3.0; 4.0]
                remap!(target_field, R, source_field)
                @test vec(parent(target_field)) ≈ [2.0; 2.0; 3.0; 3.0]
            end
        end
    end

    @testset "2D Domains" begin
        @testset "Aligned Grids" begin
            domain = Domains.RectangleDomain(
                Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
                Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )

            source = make_space(domain, 1, 2, 2)
            target = make_space(domain, 1, 3, 3)

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
                n = length(Spaces.local_geometry_data(source))
                source_field =
                    Fields.Field(IJFH{FT, 1}(ones(1, 1, n, 1)), source)

                # test consistent remap
                target_field = remap(R, source_field)
                @test vec(parent(target_field)) ≈
                      ones(length(Spaces.local_geometry_data(target)))

                # test simple remap
                vec(parent(source_field)) .= [1.0; 2.0; 3.0; 4.0]
                remap!(target_field, R, source_field)
                @test vec(parent(target_field)) ≈
                      [1.0; 1.5; 2.0; 2.0; 2.5; 3.0; 3.0; 3.5; 4.0]
            end
        end

        @testset "Unaligned Grids" begin
            domain1 = Domains.RectangleDomain(
                Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
                Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )
            source = make_space(domain1, 1, 2, 2)

            domain2 = Domains.RectangleDomain(
                Geometry.XPoint(0.0) .. Geometry.XPoint(2.0),
                Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )
            target = make_space(domain2, 1, 2, 2)

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
            source = make_space(domain1, 1, 2, 2)

            domain2 = Domains.RectangleDomain(
                Geometry.XPoint(-0.5) .. Geometry.XPoint(0.5),
                Geometry.YPoint(-0.5) .. Geometry.YPoint(0.5),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )
            target = make_space(domain2, 1, 2, 2)

            R = LinearRemap(target, source)
            R_true = I

            @test R.map ≈ R_true
            @test nnz(R.map) == 4

            @test !conservative(R)
            @test consistent(R)
            @test monotone(R)

            @testset "Scalar Remap Operator Application" begin
                n = length(Spaces.local_geometry_data(source))
                source_field =
                    Fields.Field(IJFH{FT, 1}(ones(1, 1, n, 1)), source)

                # test consistent remap
                target_field = remap(R, source_field)
                @test vec(parent(target_field)) ≈
                      ones(length(Spaces.local_geometry_data(target)))

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
            space1 = make_space(domain, 1)

            @test local_weights(space1) ≈ ones(1)

            for nq in [2, 5, 10]
                space2 = make_space(domain, nq)

                _, w = Quadratures.quadrature_points(FT, Quadratures.GLL{nq}())
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
            space1 = make_space(domain, 1, 2)
            space2 = make_space(domain, 2, 3)

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

            space2 = make_space(domain, 3, 3)

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

    @testset "2D Domains" begin
        @testset "Aligned Grids" begin
            domain = Domains.RectangleDomain(
                Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
                Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
                x1boundary = (:bottom, :top),
                x2boundary = (:left, :right),
            )

            space1 = make_space(domain, 2, 1, 1)
            space2 = make_space(domain, 1, 2, 2)

            # SE -> FV
            R = LinearRemap(space2, space1)
            # R_true = []

            # @test R.map ≈ R_true
            @test nnz(R.map) == 16

            @test conservative(R)
            @test consistent(R)

            # FV -> SE
            R = LinearRemap(space1, space2)
            @test conservative(R)
            @test consistent(R)
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
                space1 = make_space(domain, nq1)

                _, w = Quadratures.quadrature_points(FT, Quadratures.GLL{nq1}())
                @test local_weights(space1) ≈ w ./ 2

                for nq2 in [2, 4, 10]
                    space2 = make_space(domain, nq2)

                    _, w = Quadratures.quadrature_points(
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
            space1 = make_space(domain, 3, 2)
            space2 = make_space(domain, 2, 3)

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

    @testset "2D Domains" begin
        domain = Domains.RectangleDomain(
            Geometry.XPoint(-1.0) .. Geometry.XPoint(1.0),
            Geometry.YPoint(-1.0) .. Geometry.YPoint(1.0),
            x1boundary = (:bottom, :top),
            x2boundary = (:left, :right),
        )

        @testset "Single aligned elements" begin

            nq1 = 2
            space1 = make_space(domain, nq1)
            nq2 = 3
            space2 = make_space(domain, nq2)

            # SE1 -> SE2
            @test Operators.x_overlap(space2, space1) ==
                  Operators.y_overlap(space2, space1)
            R = LinearRemap(space2, space1)

            I_true = [1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9] # row
            J_true = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4] # col
            @test (I_true, J_true) == sparsity_pattern(R)
            @test size(R.map) == operator_size(R)

            @test conservative(R)
            @test consistent(R)

            # SE2 -> SE1
            R = LinearRemap(space1, space2)
            @test size(R.map) == operator_size(R)

            @test conservative(R)
            @test consistent(R)
        end

        @testset "Multiple elements" begin
            @testset "Aligned Elements" begin
                nq1 = 2
                space1 = make_space(domain, 2, 2, 2)
                nq2 = 3
                space2 = make_space(domain, 3, 2, 2)

                # SE1 -> SE2
                @test Operators.x_overlap(space2, space1) ==
                      Operators.y_overlap(space2, space1)
                R = LinearRemap(space2, space1)

                I_true = [1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9]
                I_true = [
                    I_true...,
                    (nq2^2) .+ I_true...,
                    (nq2^2 * 2) .+ I_true...,
                    (nq2^2 * 3) .+ I_true...,
                ]
                J_true = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
                J_true = [
                    J_true...,
                    (nq1^2) .+ J_true...,
                    (nq1^2 * 2) .+ J_true...,
                    (nq1^2 * 3) .+ J_true...,
                ]
                @test (I_true, J_true) == sparsity_pattern(R)
                @test size(R.map) == operator_size(R)
                @test conservative(R)
                @test consistent(R)

                # SE2 -> SE1
                R = LinearRemap(space1, space2)
                @test size(R.map) == operator_size(R)
                @test conservative(R)
                @test consistent(R)
            end

            @testset "Unaligned Elements" begin
                # Square element layout
                nq1 = 2
                nq2 = 2
                space1 = make_space(domain, nq1, 2, 2)
                space2 = make_space(domain, nq2, 3, 3)

                Xov = Operators.x_overlap(space2, space1)
                Yov = Operators.y_overlap(space2, space1)
                @test Xov == Yov
                I_true = [1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6]
                J_true = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
                @test sparsity_pattern(Xov) == (I_true, J_true)

                # SE1 -> SE2
                R = LinearRemap(space2, space1)
                #! format: off
                I_true = [1, 2, 3, 4, 5, 7, 13, 14, 17,
                          2, 4, 5, 6, 7, 8, 14, 17, 18,
                          3, 4, 7, 13, 14, 15, 16, 17, 19,
                          4, 7, 8, 14, 16, 17, 18, 19, 20,
                ]
                J_true = [1, 1, 1, 1, 1, 1, 1, 1, 1,
                          2, 2, 2, 2, 2, 2, 2, 2, 2,
                          3, 3, 3, 3, 3, 3, 3, 3, 3,
                          4, 4, 4, 4, 4, 4, 4, 4, 4,
                ]
                #! format: on
                I_true = [
                    I_true...,
                    (nq2^2) .+ I_true...,
                    (nq2^2 * 3) .+ I_true...,
                    (nq2^2 * 4) .+ I_true...,
                ]
                J_true = [
                    J_true...,
                    (nq1^2) .+ J_true...,
                    (nq1^2 * 2) .+ J_true...,
                    (nq1^2 * 3) .+ J_true...,
                ]
                @test sparsity_pattern(R) == (I_true, J_true)
                @test size(R.map) == operator_size(R)
                @test conservative(R)
                @test consistent(R)

                # SE2 -> SE1
                R = LinearRemap(space1, space2)
                @test size(R.map) == operator_size(R)
                @test conservative(R)
                @test consistent(R)

                # Rectangular element layout
                space1 = make_space(domain, 2, 2, 3)
                space2 = make_space(domain, 2, 3, 2)

                Xov = Operators.x_overlap(space2, space1)
                Yov = Operators.y_overlap(space2, space1)
                @test Xov == Yov'

                # SE1 -> SE2
                R = LinearRemap(space2, space1)
                @test size(R.map) == operator_size(R)
                @test conservative(R)
                @test consistent(R)

                # SE2 -> SE1
                R = LinearRemap(space1, space2)
                @test size(R.map) == operator_size(R)
                @test conservative(R)
                @test consistent(R)

                # Differing higher orders
                space1 = make_space(domain, 5, 4, 3)
                space2 = make_space(domain, 8, 5, 8)

                R = LinearRemap(space2, space1)
                @test size(R.map) == operator_size(R)
                @test conservative(R)
                @test consistent(R)

                R = LinearRemap(space1, space2)
                @test size(R.map) == operator_size(R)
                @test conservative(R)
                @test consistent(R)
            end
        end
    end
end
