using Test
import ClimaCore
using ClimaCore: Fields, Geometry, Grids, CommonSpaces, InputOutput, Spaces

using ClimaComms
ClimaComms.@import_required_backends
import Random

const comms_ctx = ClimaComms.SingletonCommsContext(ClimaComms.device())
filename = tempname(; cleanup = true)

@testset "HDF5 restart test for multi-column point-cloud spaces" begin
    Random.seed!(42)
    FT = Float32
    points = [
        Geometry.LatLongPoint(FT(0), FT(0)),
        Geometry.LatLongPoint(FT(10), FT(20)),
        Geometry.LatLongPoint(FT(-5), FT(90)),
    ]
    radius = FT(6.371229e6)

    center_space = CommonSpaces.MultiColumnSpace(
        FT;
        points,
        radius,
        z_elem = 10,
        z_min = 0,
        z_max = 10_000,
        staggering = Grids.CellCenter(),
        device = ClimaComms.device(comms_ctx),
    )
    face_space = Spaces.face_space(center_space)
    level_space = Spaces.level(center_space, 1) # MultiPointSpace

    Y = Fields.FieldVector(;
        c = Fields.Field(FT, center_space),
        f = Fields.Field(FT, face_space),
        l = Fields.Field(FT, level_space),
    )
    for field in (Y.c, Y.f, Y.l)
        parent(field) .= rand.(FT)
    end

    InputOutput.HDF5Writer(filename, comms_ctx) do writer
        InputOutput.write!(writer, Y, "Y")
    end

    InputOutput.HDF5Reader(filename, comms_ctx) do reader
        restart_Y = InputOutput.read_field(reader, "Y")
        @test axes(restart_Y.c) isa Spaces.CenterMultiColumnFiniteDifferenceSpace
        @test axes(restart_Y.f) isa Spaces.FaceMultiColumnFiniteDifferenceSpace
        @test axes(restart_Y.l) isa Spaces.MultiPointSpace
        hgrid = Spaces.grid(axes(restart_Y.c)).horizontal_grid
        @test hgrid isa Grids.MultiPointGrid
        @test hgrid.global_geometry.radius == radius
        @test Spaces.coordinates_data(Spaces.horizontal_space(axes(restart_Y.c))) ==
              Spaces.coordinates_data(Spaces.horizontal_space(center_space))
        @test restart_Y == Y # test if restart is exact
    end
end
