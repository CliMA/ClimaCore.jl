#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "unit_field_array.jl"))
=#
using Test
using ClimaCore.DataLayouts: field_array
using ClimaCore.DataLayouts: FieldArray
import ClimaComms
ClimaComms.@import_required_backends

@testset "FieldArray" begin

end
