using Test
import ClimaCore
import ClimaCore: Fields, InputOutput
using ClimaComms
ClimaComms.@import_required_backends
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

compare_read_type(x) = InputOutput.read_type(string(eltype(x))) == eltype(x)
@testset "Read field element types" begin
    context = ClimaComms.context()
    FT = Float64

    for space in TU.all_spaces(FT; context)
        lg_field = Fields.local_geometry_field(space)
        @test compare_read_type(lg_field)
        coord_field = Fields.coordinate_field(space)
        @test compare_read_type(coord_field)
    end

    space = TU.ColumnCenterFiniteDifferenceSpace(FT; context)
    @test compare_read_type(fill((1.0, 2.0, (3.0, 4.0, (5.0,))), space))
    @test compare_read_type(fill((; a = FT(0), b = (; c = FT(1))), space))
    # test that attempting to read a type from a string with an expression that executes code throws an error
    @test_throws ErrorException InputOutput.read_type("1+2")
    @test_throws ErrorException InputOutput.read_type("Base.rm(\"foo.jl\")")
end

@testset "backwards compatibility with `value_type` attribute" begin
    context = ClimaComms.context()
    FT = Float64

    space = TU.ColumnCenterFiniteDifferenceSpace(FT; context)
    Y = fill((1.0, 2.0, (3.0, 4.0, (5.0,))), space)
    filename = tempname()
    InputOutput.HDF5Writer(filename, context) do writer
        InputOutput.write!(writer, "Y" => Y) # write field vector from hdf5 file
        obj = writer.file["fields/Y"]
        InputOutput.HDF5.delete_attribute(obj, "field_eltype") # remove the field_eltype attribute
        InputOutput.HDF5.write_attribute(obj, "value_type", string(eltype(Y))) # add the deprecated value_type attribute
    end
    InputOutput.HDF5Reader(filename, context) do reader
        restart_Y = InputOutput.read_field(reader, "Y")
        @test eltype(restart_Y) == eltype(Y)
    end
end
