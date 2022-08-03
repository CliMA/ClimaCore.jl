using Test

import Arrow
import PrettyTables

import ClimaCore
import ClimaCoreTables

OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

function round_trip(filename, data)
    arrow_file = joinpath(OUTPUT_DIR, filename)
    open(arrow_file, write = true) do fh
        # write in the arrow file format rather than the streaming format
        Arrow.write(fh, data, file = true)
    end
    tbl = Arrow.Table(arrow_file)
    # PrettyTables.pretty_table(tbl)
    @test ClimaCoreTables.Tables.matrix(data) ==
          ClimaCoreTables.Tables.matrix(tbl)
end

@testset "Basic ClimaCoreTable interface implementation" begin
    FT = Float64
    data1 = ones(FT, 4, 1)
    data1d = ClimaCore.DataLayouts.VF{FT}(data1)

    # create table
    tbl = ClimaCoreTables.Tables.table(data1d)
    @test ClimaCoreTables.Tables.istable(tbl)

    # test column interface
    @test ClimaCoreTables.Tables.columnaccess(typeof(tbl))

    # also allow for columns of datatypes
    # (used for duck typing to avoid explicit creation of CCTable in many Table impls)
    @test ClimaCoreTables.Tables.columnaccess(typeof(data1d))

    @test ClimaCoreTables.Tables.columns(tbl) === tbl
    @test ClimaCoreTables.Tables.getcolumn(tbl, 1) ===
          ClimaCoreTables._table_columns(tbl)[1]
    @test ClimaCoreTables.Tables.getcolumn(tbl, :data) ===
          ClimaCoreTables._table_columns(tbl)[1]

    @test ClimaCoreTables.Tables.columnnames(tbl) == [:data]

    # test row interface
    @test ClimaCoreTables.Tables.rowaccess(typeof(tbl))
    @test ClimaCoreTables.Tables.rows(tbl) === tbl

    row = first(tbl)
    @test eltype(tbl) == typeof(row)
    @test row.data == 1
    @test ClimaCoreTables.Tables.getcolumn(row, 1) == 1
    @test ClimaCoreTables.Tables.getcolumn(row, :data) == 1
    @test propertynames(tbl) == propertynames(row) == [:data]
end

@testset "Data1D scalar type" begin
    for FT in (Float32, Float64)
        data1 = ones(FT, 4, 1)
        S = FT
        data1d = ClimaCore.DataLayouts.VF{S}(data1)
        round_trip("data1d_scalar_$(FT).arrow", data1d)
    end
end

@testset "Data1D tuple type" begin
    for FT in (Float32, Float64)
        data1 = ones(FT, 4, 3)
        S = Tuple{FT, Complex{FT}}
        data1d = ClimaCore.DataLayouts.VF{S}(data1)
        round_trip("data1d_tuple_$(FT).arrow", data1d)
    end
end

@testset "Data1D named tuple type" begin
    for FT in (Float32, Float64)
        data1 = ones(FT, 4, 3)
        S = typeof((a = one(FT), b = complex(one(FT))))
        data1d = ClimaCore.DataLayouts.VF{S}(data1)
        round_trip("data1d_namedtuple_$(FT).arrow", data1d)
    end
end

function sphere_space(FT; ne = 4, Nq = 4, radius = FT(3))
    domain = ClimaCore.Domains.SphereDomain(radius)
    mesh = ClimaCore.Meshes.EquiangularCubedSphere(domain, ne)
    topology = ClimaCore.Topologies.Topology2D(mesh)
    quad = ClimaCore.Spaces.Quadratures.GLL{Nq}()
    ClimaCore.Spaces.SpectralElementSpace2D(topology, quad)
end

@testset "Sphere Space / Field" begin
    FT = Float64
    space = sphere_space(FT)
    field = ones(FT, space)
    round_trip("space_sphere_$(FT).arrow", space)
    round_trip("field_sphere_$(FT).arrow", field)
end
