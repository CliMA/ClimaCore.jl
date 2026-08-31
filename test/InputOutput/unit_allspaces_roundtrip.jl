# Round-trip property test: for a field on every space that `TestUtilities`
# can construct — and hence for every data layout those spaces use — reading
# back a written field must reproduce it exactly (bit-identical values). The
# per-space InputOutput tests exercise richer field types on specific spaces;
# this sweep guards the write/read path itself against layout permutation and
# reshaping bugs on every space at once.
using Test
import ClimaCore
import ClimaCore: Spaces, Fields, InputOutput
using ClimaComms
ClimaComms.@import_required_backends
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;
import Random: seed!

@testset "HDF5 round-trip on all spaces [$FT]" for FT in (Float32, Float64)
    context = ClimaComms.context()
    seed!(1)
    for space in TU.all_spaces(FT; context)
        # TODO: InputOutput cannot serialize a MultiColumnFiniteDifferenceSpace
        # yet: its MultiPointGrid horizontal grid has no topology, which
        # `write!` requires (`Spaces.topology` errors). Remove this skip once
        # HDF5 support for multi-point grids lands.
        if space isa Spaces.MultiColumnFiniteDifferenceSpace
            @test_skip "HDF5 round-trip on MultiColumnFiniteDifferenceSpace"
            continue
        end
        field = Fields.Field(FT, space)
        parent(field) .= rand.(FT)
        Y = Fields.FieldVector(; f = field)

        filename = tempname(; cleanup = true)
        InputOutput.HDF5Writer(filename, context) do writer
            InputOutput.write!(writer, "Y" => Y)
        end
        InputOutput.HDF5Reader(filename, context) do reader
            restart_Y = InputOutput.read_field(reader, "Y")
            @testset "$(nameof(typeof(space))) ($(nameof(typeof(Fields.field_values(field)))))" begin
                @test parent(restart_Y.f) == parent(Y.f) # bit-identical
                @test typeof(Fields.field_values(restart_Y.f)) ==
                      typeof(Fields.field_values(Y.f))
            end
        end
    end
end
