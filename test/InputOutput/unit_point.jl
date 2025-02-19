using Test
import ClimaCore

using ClimaComms
const comms_ctx = ClimaComms.context(ClimaComms.CPUSingleThreaded())


@testset "HDF5 write/read test for 0d PointSpace" begin
    # need to define equality because with no Grid to cache, the read space will not be "egal"
    # to the written space.
    function Base.:(==)(
        lg1::ClimaCore.Geometry.LocalGeometry{I, C, FT, <:Any},
        lg2::ClimaCore.Geometry.LocalGeometry{I, C, FT, <:Any},
    ) where {I, C, FT}
        for f in [:coordinates, :J, :WJ, :invJ, :∂x∂ξ, :∂ξ∂x, :gⁱʲ, :gᵢⱼ]
            if getfield(lg1, f) != getfield(lg2, f)
                return false
            end
        end
        return true
    end

    FT = Float32

    space =
        ClimaCore.Spaces.PointSpace(comms_ctx, ClimaCore.Geometry.ZPoint(FT(1)))
    field_0d = ClimaCore.Fields.local_geometry_field(space)
    Y = ClimaCore.Fields.FieldVector(; p = field_0d)

    filename = tempname()

    ClimaCore.InputOutput.HDF5Writer(filename, comms_ctx) do writer
        ClimaCore.InputOutput.write!(writer, "Y" => Y) # write field vector from hdf5 file
    end

    ClimaCore.InputOutput.HDF5Reader(filename, comms_ctx) do reader
        restart_Y = ClimaCore.InputOutput.read_field(reader, "Y") # read fieldvector from hdf5 file
        @test restart_Y == Y # test if restart is exact
        # test if space is the same by comparing local geometry data
        @test ClimaCore.Spaces.local_geometry_data(axes(restart_Y.p))[] ==
              ClimaCore.Spaces.local_geometry_data(axes(Y.p))[]
    end
end
