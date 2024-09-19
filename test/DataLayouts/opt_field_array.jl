#=
julia --project
using Revise; include(joinpath("test", "DataLayouts", "opt_field_array.jl"))
=#
using Test
using ClimaCore.DataLayouts: field_array
using ClimaCore.DataLayouts: FieldArray, ArraySize
import ClimaComms
import JET
ClimaComms.@import_required_backends

@testset "FieldArray" begin
    array = rand(3, 4, 5)
    as = ArraySize{2, size(array, 2), size(array)}()
    field_array(array, 2)
    JET.@test_opt field_array(array, 2)
    JET.@test_opt field_array(array, as)
end
