# Test truncated field type printing:
import Preferences
Preferences.@set_preferences!(Pair("TruncateFieldPrinting" => true))
using Test
using ClimaCore: Fields, Spaces, Geometry

function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

include(joinpath(@__DIR__, "util_spaces.jl"))

@testset "Truncated printing" begin
    nt = (; x = Float64(0), y = Float64(0))
    Y = FieldFromNamedTuple(spectral_space_2D(), nt)
    @test sprint(show, typeof(Y)) == "Field{(:x, :y)} (trunc disp)"
end
