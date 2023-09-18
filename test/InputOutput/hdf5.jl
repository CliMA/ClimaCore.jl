using Test
using ClimaCore: InputOutput

filename = tempname(pwd())

@testset "General HDF5 features" begin
    writer = InputOutput.HDF5Writer(filename)

    # Write some data
    InputOutput.HDF5.create_dataset(writer.file, "test", [1, 2, 3])

    # Write attributes
    attributes = Dict("my_attr" => 1)
    InputOutput.write_attributes!(writer, "/test", attributes)
    close(writer)

    reader = InputOutput.HDF5Reader(filename)
    attributes_read = InputOutput.read_attributes(reader, "test")
    @test attributes_read == attributes
    close(reader)
end
