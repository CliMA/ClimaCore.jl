import Arrow

# To write out an arrow file use:
#    Arrow.write(io, Field / Space / ClimaCoreTable..., file=true)
#
# To read into a ex. pandas dataframe:
#    import pyarrow as pa
#    arrow_table = pa.ipc.open_file("field.arrow").read_all()
#    dataframe = arrow_table.to_pandas()

# TODO: add custom schema serialization metadata to allow for roudtrip
# between representations this can be attached to the Arrow schema
# (ex. JSON / TOML description):
#    See:
#       https://arrow.juliadata.org/stable/manual/#Custom-application-metadata

function Arrow.getmetadata(
    table::ClimaCoreTable{T},
) where {T <: ClimaCore.DataLayouts.AbstractData}
    return nothing
end

function Arrow.getmetadata(
    table::ClimaCoreTable{T},
) where {T <: ClimaCore.Spaces.AbstractSpace}
    return nothing
end

function Arrow.getmetadata(
    table::ClimaCoreTable{T},
) where {T <: ClimaCore.Fields.Field}
    return nothing
end
