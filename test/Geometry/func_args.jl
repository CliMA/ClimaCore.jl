#=
This file is for providing a list of arguments
to call different functions with.
=#

#! format: off

function all_axes()
    all_Is() = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    collect(Iterators.flatten(map(all_Is()) do I
        (
            CovariantAxis{I}(),
            ContravariantAxis{I}(),
            LocalAxis{I}(),
            CartesianAxis{I}()
        )
    end))
end

all_observed_axistensors(::Type{FT}) where {FT} =
    vcat(map(x-> rand(last(x)), used_project_arg_types(FT)),
         map(x-> rand(last(x)), used_transform_arg_types(FT)))

func_args(FT, ::typeof(Geometry.project)) =
    map(used_project_arg_types(FT)) do (axt, axtt)
        (axt(), rand(axtt))
    end
func_args(FT, ::typeof(Geometry.transform)) =
    map(used_transform_arg_types(FT)) do (axt, axtt)
        (axt(), rand(axtt))
    end

function all_possible_func_args(FT, ::typeof(Geometry.contravariant3))
    # TODO: this is not accurate yet, since we don't yet
    # vary over all possible LocalGeometry's.
    M = @SMatrix [
        FT(4) FT(1)
        FT(0.5) FT(2)
    ]
    J = LinearAlgebra.det(M)
    ∂x∂ξ = rand(Geometry.AxisTensor{FT, 2, Tuple{Geometry.LocalAxis{(3,)}, Geometry.CovariantAxis{(3,)}}, SMatrix{1, 1, FT, 1}})
    lg = Geometry.LocalGeometry(Geometry.XYPoint(FT(0), FT(0)), J, J, ∂x∂ξ)
    # Geometry.LocalGeometry{(3,), Geometry.ZPoint{FT}, FT, SMatrix{1, 1, FT, 1}}
    Iterators.flatten(
        map(used_project_arg_types(FT)) do (axt, axtt)
            map(all_axes()) do ax
                (rand(axtt), lg)
            end
        end
    )
end

function func_args(FT, f::typeof(Geometry.contravariant3))
    # TODO: fix this..
    apfa = all_possible_func_args(FT, f)
    args_dict = Dict()
    for args in apfa
        hasmethod(f, typeof(args)) || continue
        args_dict[dict_key(f, args)] = args
    end
    return values(args_dict)
end

#! format: on
