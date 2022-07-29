# This was collected by running ClimaCore's test suite
# and printing all input argument types for `project`
function used_project_arg_types(::Type{FT}) where {FT}
    return [
        (CovariantAxis{(1, 2)}, Covariant1Vector{FT}),
        (CovariantAxis{(1, 2)}, Covariant12Vector{FT}),
        (CovariantAxis{(1, 2)}, Covariant123Vector{FT}),
        (CovariantAxis{(1, 2)}, Covariant13Vector{FT}),
        (
            CovariantAxis{(1, 2)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(1,)}, CartesianAxis{(1,)}},
                StaticArraysCore.SMatrix{1, 1, FT, 1},
            },
        ),
        (
            CovariantAxis{(1, 2)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(1, 3)}, CartesianAxis{(1,)}},
                StaticArraysCore.SMatrix{2, 1, FT, 2},
            },
        ),
        (
            LocalAxis{(1, 2)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(1, 2, 3)}, CovariantAxis{(1, 2)}},
                StaticArraysCore.SMatrix{3, 2, FT, 6},
            },
        ),
        (LocalAxis{(1, 2)}, UVVector{FT}),
        (ContravariantAxis{(1,)}, Contravariant12Vector{FT}),
        (ContravariantAxis{(2,)}, Contravariant12Vector{FT}),
        (CovariantAxis{(1, 2, 3)}, Covariant3Vector{FT}),
        (ContravariantAxis{(3,)}, Contravariant123Vector{FT}),
        (CovariantAxis{(3,)}, Covariant3Vector{FT}),
        (ContravariantAxis{(3,)}, Contravariant3Vector{FT}),
        (LocalAxis{(3,)}, WVector{FT}),
        (
            CovariantAxis{(3,)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(3,)}, LocalAxis{(1, 2)}},
                StaticArraysCore.SMatrix{1, 2, FT, 2},
            },
        ),
        (
            ContravariantAxis{(3,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(3,)}, LocalAxis{(1, 2)}},
                StaticArraysCore.SMatrix{1, 2, FT, 2},
            },
        ),
        (
            LocalAxis{(3,)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(3,)}, LocalAxis{(1, 2)}},
                StaticArraysCore.SMatrix{1, 2, FT, 2},
            },
        ),
        (
            CovariantAxis{(3,)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(3,)}, LocalAxis{(3,)}},
                StaticArraysCore.SMatrix{1, 1, FT, 1},
            },
        ),
        (
            ContravariantAxis{(3,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(3,)}, LocalAxis{(3,)}},
                StaticArraysCore.SMatrix{1, 1, FT, 1},
            },
        ),
        (
            LocalAxis{(1, 2)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(1, 2)}, LocalAxis{(1, 2)}},
                StaticArraysCore.SMatrix{2, 2, FT, 4},
            },
        ),
        (
            ContravariantAxis{(1,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 2)}, LocalAxis{(1, 2)}},
                StaticArraysCore.SMatrix{2, 2, FT, 4},
            },
        ),
        (
            ContravariantAxis{(2,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 2)}, LocalAxis{(1, 2)}},
                StaticArraysCore.SMatrix{2, 2, FT, 4},
            },
        ),
        (CovariantAxis{(1, 2, 3)}, Covariant123Vector{FT}),
        (ContravariantAxis{(1,)}, Contravariant123Vector{FT}),
        (ContravariantAxis{(2,)}, Contravariant123Vector{FT}),
        (CovariantAxis{(1, 2, 3)}, Covariant12Vector{FT}),
        (CovariantAxis{(1, 3)}, Covariant1Vector{FT}),
        (ContravariantAxis{(1,)}, Contravariant13Vector{FT}),
        (
            CovariantAxis{(1, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(1,)}, LocalAxis{(1,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (
            ContravariantAxis{(1,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 3)}, LocalAxis{(1,)}},
                SMatrix{2, 1, FT, 2},
            },
        ),
        (
            CovariantAxis{(1, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(1,)}, LocalAxis{(3,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (
            ContravariantAxis{(1,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 3)}, LocalAxis{(3,)}},
                SMatrix{2, 1, FT, 2},
            },
        ),
        (LocalAxis{(1, 3)}, WVector{FT}),
        (ContravariantAxis{(3,)}, Contravariant13Vector{FT}),
        (LocalAxis{(1, 3)}, UVector{FT}),
        (
            LocalAxis{(1, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(3,)}, LocalAxis{(1,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (
            ContravariantAxis{(3,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 3)}, LocalAxis{(1,)}},
                SMatrix{2, 1, FT, 2},
            },
        ),
        (
            LocalAxis{(1, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(1,)}, LocalAxis{(1,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (
            LocalAxis{(1, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(3,)}, LocalAxis{(3,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (
            ContravariantAxis{(3,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 3)}, LocalAxis{(3,)}},
                SMatrix{2, 1, FT, 2},
            },
        ),
        (
            LocalAxis{(1, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(1,)}, LocalAxis{(3,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (
            CovariantAxis{(1, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(3,)}, LocalAxis{(1,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (
            CovariantAxis{(1, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(3,)}, LocalAxis{(3,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (CovariantAxis{(1, 3)}, Covariant3Vector{FT}),
        (
            CovariantAxis{(1, 2, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(1, 2)}, LocalAxis{(1, 2)}},
                SMatrix{2, 2, FT, 4},
            },
        ),
        (
            ContravariantAxis{(1,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 2, 3)}, LocalAxis{(1, 2)}},
                SMatrix{3, 2, FT, 6},
            },
        ),
        (
            ContravariantAxis{(2,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 2, 3)}, LocalAxis{(1, 2)}},
                SMatrix{3, 2, FT, 6},
            },
        ),
        (
            CovariantAxis{(1, 2, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(1, 2)}, LocalAxis{(3,)}},
                SMatrix{2, 1, FT, 2},
            },
        ),
        (
            ContravariantAxis{(1,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 2, 3)}, LocalAxis{(3,)}},
                SMatrix{3, 1, FT, 3},
            },
        ),
        (
            ContravariantAxis{(2,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 2, 3)}, LocalAxis{(3,)}},
                SMatrix{3, 1, FT, 3},
            },
        ),
        (LocalAxis{(1, 2, 3)}, WVector{FT}),
        (LocalAxis{(1, 2, 3)}, UVVector{FT}),
        (
            LocalAxis{(1, 2, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(3,)}, LocalAxis{(1, 2)}},
                SMatrix{1, 2, FT, 2},
            },
        ),
        (
            ContravariantAxis{(3,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 2, 3)}, LocalAxis{(1, 2)}},
                SMatrix{3, 2, FT, 6},
            },
        ),
        (
            LocalAxis{(1, 2, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(1, 2)}, LocalAxis{(1, 2)}},
                SMatrix{2, 2, FT, 4},
            },
        ),
        (
            LocalAxis{(1, 2, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(3,)}, LocalAxis{(3,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (
            ContravariantAxis{(3,)},
            AxisTensor{
                FT,
                2,
                Tuple{ContravariantAxis{(1, 2, 3)}, LocalAxis{(3,)}},
                SMatrix{3, 1, FT, 3},
            },
        ),
        (
            LocalAxis{(1, 2, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{LocalAxis{(1, 2)}, LocalAxis{(3,)}},
                SMatrix{2, 1, FT, 2},
            },
        ),
        (
            CovariantAxis{(1, 2, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(3,)}, LocalAxis{(1, 2)}},
                SMatrix{1, 2, FT, 2},
            },
        ),
        (
            CovariantAxis{(1, 2, 3)},
            AxisTensor{
                FT,
                2,
                Tuple{CovariantAxis{(3,)}, LocalAxis{(3,)}},
                SMatrix{1, 1, FT, 1},
            },
        ),
        (CovariantAxis{(1, 3)}, Covariant13Vector{FT}),
        (LocalAxis{(3,)}, UWVector{FT}),
        (LocalAxis{(1,)}, UWVector{FT}),
    ]
end
