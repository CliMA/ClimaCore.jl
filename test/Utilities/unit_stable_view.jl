#=
julia --project
using Revise; include(joinpath("test", "Utilities", "unit_stable_view.jl"))
=#
using Test
import ClimaCore.Utilities: stable_view

@testset "equivalence with view" begin
    array = rand(3, 4, 1, 5)
    for indices in (
        (:, 2, 1, 4),
        (1, :, :, :),
        (2, 3:4, 1, :),
        (:, :, :, :),
        (2, 3, 1, 4),
        (CartesianIndex(2, 3, 1, 4),),
        (CartesianIndices((2:3, 1:4, 1:1, 5:5)),),
        (7,),
        (4:9,),
        (2:3:50,),
    )
        @test stable_view(array, indices...) == view(array, indices...)
    end

    slice = reshape(view(array, :, 2, 1, :), 3, 1, 1, 5)
    @test stable_view(slice, :, 1, 1, 4) == view(slice, :, 1, 1, 4)
    @test stable_view(slice, 2:3) == view(slice, 2:3)
end

@testset "views along linear indices" begin
    array = rand(3, 4, 1, 5)

    # A view along linear indices should not allocate a reshaped copy of the
    # original Array object. (The comparison uses < instead of == because
    # @allocated has a small constant overhead in local scopes.)
    @test parent(view(array, 4:6)) isa Vector
    @test parent(stable_view(array, 4:6)) isa Base.ReshapedArray
    view_value(array) = view(array, 4:6)[3]
    stable_view_value(array) = stable_view(array, 4:6)[3]
    view_value(array)
    stable_view_value(array)
    @test (@allocated stable_view_value(array)) <
          (@allocated view_value(array))

    # A view along the linear indices of a ReshapedArray should be a view of
    # the ReshapedArray's parent, since a reshape stores the same values in
    # the same linear order as its parent.
    slice = reshape(view(array, :, 2, 1, :), 3, 1, 1, 5)
    @test parent(stable_view(slice, 2:3)) isa
          Base.ReshapedArray{Float64, 1, <:SubArray{Float64, <:Any, <:Array}}
end
