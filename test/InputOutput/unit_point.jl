using Test
import ClimaCore: Spaces, Fields, InputOutput, Geometry
using ClimaComms
const comms_ctx = ClimaComms.context()


@testset "HDF5 write/read test for 0d PointSpace" begin

    FT = Float32

    space = Spaces.PointSpace(comms_ctx, Geometry.ZPoint(FT(1)))
    field_0d = Fields.local_geometry_field(space)
    Y = Fields.FieldVector(; p = field_0d)

    filename = tempname()

    InputOutput.HDF5Writer(filename, comms_ctx) do writer
        InputOutput.write!(writer, "Y" => Y) # write field vector from hdf5 file
    end

    InputOutput.HDF5Reader(filename, comms_ctx) do reader
        restart_Y = InputOutput.read_field(reader, "Y") # read fieldvector from hdf5 file
        @test restart_Y == Y # test if restart is exact
        # test if space is the same by comparing local geometry data
        # note that the read spaces are not "===" to the written spaces because there is no grid to cache
        lg_point = Spaces.local_geometry_data(axes(Y.p))[]
        lg_restart = Spaces.local_geometry_data(axes(restart_Y.p))[]
        @test typeof(lg_point) == typeof(lg_restart)
        @test all(
            getproperty(lg_point, p) == getproperty(lg_restart, p) for
            p in propertynames(lg_point)
        )
    end
end
