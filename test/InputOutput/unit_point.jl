using Test
import ClimaCore

using ClimaComms
const comms_ctx = ClimaComms.context(ClimaComms.CPUSingleThreaded())


@testset "HDF5 write/read test for 0d PointSpace" begin
    # need to define equality because with no Grid to cache, the read space will not be "egal"
    # to the written space. Furthermore, the typeof the read Field will be different from the
    # written Field's because the written Field contains SubArrays of the column
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

    Base.:(==)(
        data1::ClimaCore.DataLayouts.DataF,
        data2::ClimaCore.DataLayouts.DataF,
    ) = data1[] == data2[]

    Base.:(==)(
        space1::ClimaCore.Spaces.PointSpace,
        space2::ClimaCore.Spaces.PointSpace,
    ) =
        ClimaComms.context(space1) == ClimaComms.context(space2) &&
        ClimaCore.Spaces.local_geometry_data(space1) ==
        ClimaCore.Spaces.local_geometry_data(space2)

    Base.:(==)(
        field1::ClimaCore.Fields.Field{
            <:ClimaCore.DataLayouts.AbstractData,
            <:ClimaCore.Spaces.PointSpace,
        },
        field2::ClimaCore.Fields.Field{
            <:ClimaCore.DataLayouts.AbstractData,
            <:ClimaCore.Spaces.PointSpace,
        },
    ) = axes(field1) == axes(field2) && parent(field1) == parent(field2)

    FT = Float32

    # instead of directly constructing a PointSpace, we construct Field with a ColumnSpace,
    # and call the level function to get a Field with a PointSpace
    column_space = ClimaCore.CommonSpaces.ColumnSpace(;
        z_min = FT(0),
        z_max = FT(100),
        z_elem = 10,
        staggering = ClimaCore.Grids.CellCenter(),
    )

    field_1d = ClimaCore.Fields.local_geometry_field(column_space)
    field_0d = ClimaCore.Fields.level(field_1d, 5)


    filename = tempname()

    writer = ClimaCore.InputOutput.HDF5Writer(filename, comms_ctx)
    ClimaCore.InputOutput.write!(writer, field_0d, "field_0d")
    close(writer)

    reader = ClimaCore.InputOutput.HDF5Reader(filename, comms_ctx)
    restart_field_0d = ClimaCore.InputOutput.read_field(reader, "field_0d")
    close(reader)

    @test restart_field_0d == field_0d
end
