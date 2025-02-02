#=
julia --project=.buildkite
using Revise; include("test/InputOutput/unit_finitedifference.jl")
=#
using Test
import ClimaCore
using ClimaCore: Fields, Meshes, Geometry, Grids, CommonSpaces, InputOutput
using ClimaCore: Domains, Topologies, Spaces

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
    center_staggering = Grids.CellCenter()
    face_staggering = Grids.CellFace()

    center_space = CommonSpaces.ColumnSpace(;
        z_min,
        z_max,
        z_elem,
        staggering = center_staggering,
    )

    face_space = CommonSpaces.ColumnSpace(;
        z_min,
        z_max,
        z_elem,
        staggering = face_staggering,
    )

    center_field = Fields.local_geometry_field(center_space)
    face_field = Fields.local_geometry_field(face_space)

    Y = Fields.FieldVector(; c = center_field, f = face_field)

    # write field vector to hdf5 file
    InputOutput.HDF5Writer(filename, comms_ctx) do writer
        InputOutput.write!(writer, Y, "Y")
    end

    InputOutput.HDF5Reader(filename, comms_ctx) do reader
        restart_Y = InputOutput.read_field(reader, "Y") # read fieldvector from hdf5 file
        @test restart_Y == Y # test if restart is exact
    end
end

@testset "HDF5 restart test for 1d finite difference space with unknown mesh" begin
    FT = Float32
    z_min = FT(0)
    z_max = FT(30e3)
    z_elem = 10
    center_staggering = Grids.CellCenter()
    face_staggering = Grids.CellFace()

    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(10e3);
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain; nelems = 45)
    vmesh = Meshes.IntervalMesh(vdomain, vmesh.faces) # pass in faces directly
    @test vmesh.stretch isa Meshes.UnknownStretch
    context = ClimaComms.context()
    vtopology = Topologies.IntervalTopology(context, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    f = Fields.Field(FT, vspace)

    # write field vector to hdf5 file
    InputOutput.HDF5Writer(filename, comms_ctx) do writer
        InputOutput.write!(writer, f, "f")
    end

    InputOutput.HDF5Reader(filename, comms_ctx) do reader
        restart_f = InputOutput.read_field(reader, "f") # read field from hdf5 file
        @test axes(restart_f).grid.topology.mesh.faces == vmesh.faces
    end
end
