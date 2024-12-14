using Test
import ClimaCore
import ClimaCore.Fields

using ClimaComms
const comms_ctx = ClimaComms.context(ClimaComms.CPUSingleThreaded())
pid, nprocs = ClimaComms.init(comms_ctx)
filename = ClimaComms.bcast(comms_ctx, tempname(pwd()))
if ClimaComms.iamroot(comms_ctx)
    @info "Comms context" comms_ctx nprocs filename
end

@testset "HDF5 restart test for 1d finite difference space" begin
    FT = Float32

    z_min = FT(0)
    z_max = FT(30e3)
    z_elem = 10
    center_staggering = ClimaCore.Grids.CellCenter()
    face_staggering = ClimaCore.Grids.CellFace()

    center_space = ClimaCore.CommonSpaces.ColumnSpace(;
        z_min,
        z_max,
        z_elem,
        staggering = center_staggering,
    )

    face_space = ClimaCore.CommonSpaces.ColumnSpace(;
        z_min,
        z_max,
        z_elem,
        staggering = face_staggering,
    )

    center_field = Fields.local_geometry_field(center_space)
    face_field = Fields.local_geometry_field(face_space)

    Y = ClimaCore.Fields.FieldVector(; c = center_field, f = face_field)

    # write field vector to hdf5 file
    writer = ClimaCore.InputOutput.HDF5Writer(filename, comms_ctx)
    ClimaCore.InputOutput.write!(writer, Y, "Y")
    close(writer)

    reader = ClimaCore.InputOutput.HDF5Reader(filename, comms_ctx)
    restart_Y = ClimaCore.InputOutput.read_field(reader, "Y") # read fieldvector from hdf5 file
    close(reader)
    @test restart_Y == Y # test if restart is exact
end
