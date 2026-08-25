using Test
using ClimaCore: InputOutput
using ClimaComms

filename = tempname(pwd())

@testset "General HDF5 features" begin
    context = ClimaComms.context()
    local attributes
    InputOutput.HDF5Writer(filename, context) do writer

        # Write some data
        InputOutput.HDF5.create_dataset(writer.file, "test", [1, 2, 3])

        # Write attributes
        attributes = Dict("my_attr" => 1)
        InputOutput.write_attributes!(writer, "/test", attributes)
    end

    InputOutput.HDF5Reader(filename, context) do reader
        attributes_read = InputOutput.read_attributes(reader, "test")
        @test attributes_read == attributes
    end
end

@testset "read_space uses the reader's space cache" begin
    context = ClimaComms.context()
    filename2 = tempname(pwd())
    InputOutput.HDF5Writer(filename2, context) do writer
        InputOutput.HDF5.create_dataset(writer.file, "test", [1, 2, 3])
    end
    InputOutput.HDF5Reader(filename2, context) do reader
        # The legacy `spaces/` read path goes through reader.space_cache; a
        # missing group must surface as a KeyError from the file lookup, not
        # as a field error on the reader struct.
        @test reader.space_cache isa Dict
        @test_throws KeyError InputOutput.read_space(reader, "nonexistent")
    end
    rm(filename2; force = true)
end
