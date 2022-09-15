using Test
using StaticArrays, IntervalSets
import ClimaCore
import ClimaCore.Utilities: PlusHalf
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, slab, Domains, Topologies, Meshes, Operators, Spaces, Geometry

using LinearAlgebra: norm
using Statistics: mean
using ForwardDiff

function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

include(joinpath(@__DIR__, "util_spaces.jl"))

# https://github.com/CliMA/ClimaCore.jl/issues/946
@testset "Allocations with broadcasting Refs" begin
    FT = Float64
    function foo!(Yx::Fields.Field)
        Yx .= Ref(1) .+ Yx
        return nothing
    end
    function foocolumn!(Yx::Fields.Field)
        Fields.bycolumn(axes(Yx)) do colidx
            Yx[colidx] .= Ref(1) .+ Yx[colidx]
            nothing
        end
        return nothing
    end
    for space in all_spaces(FT)
        (
            space isa Spaces.ExtrudedFiniteDifferenceSpace ||
            space isa Spaces.SpectralElementSpace1D ||
            space isa Spaces.SpectralElementSpace2D
        ) || continue
        Y = FieldFromNamedTuple(space, (; x = FT(2)))

        # Plain broadcast
        Yx = Y.x
        foo!(Yx) # compile first
        p = @allocated foo!(Yx)
        @test p == 0

        # bycolumn
        foocolumn!(Yx) # compile first
        p = @allocated foocolumn!(Yx)
        @test p == 0
    end
end
