# include("src/precompile.jl")
module PrecompileClimaCore

using PrecompileTools

using LinearAlgebra: det, dot
using StaticArrays: SMatrix, SVector, SArray
using ..DataLayouts

using ..Geometry
using ..Geometry:
    ContravariantAxis,
    Contravariant3Vector,
    Contravariant123Vector,
    Contravariant13Vector,
    Contravariant12Vector,
    CovariantAxis,
    CartesianAxis,
    Covariant3Vector,
    Covariant123Vector,
    Covariant12Vector,
    Covariant1Vector,
    Covariant13Vector,
    Axis2Tensor,
    AxisTensor,
    WVector,
    UVVector,
    UVector,
    LocalAxis,
    LocalGeometry,
    XZPoint,
    XYZPoint,
    LatLongZPoint,
    XYPoint,
    ZPoint,
    LatLongPoint,
    XPoint,
    YPoint,
    LatPoint,
    LongPoint,
    contravariant1,
    contravariant2,
    contravariant3,
    Jcontravariant3

FT = Float64
include("geometry_precompile_list.jl")
FT = Float32
include("geometry_precompile_list.jl")

@setup_workload begin
    @compile_workload begin
        for FT in (Float32, Float64)
            # DataLayouts
            a1 = rand(FT, 1)
            DataLayouts.get_struct(a1, FT, Val(1), CartesianIndex(1))
            a2 = rand(FT, 1, 1)
            DataLayouts.get_struct(a2, FT, Val(1), CartesianIndex(1, 1))
            DataLayouts.get_struct(a2, FT, Val(2), CartesianIndex(1, 1))
            a3 = rand(FT, 1, 1, 1)
            DataLayouts.get_struct(a3, FT, Val(1), CartesianIndex(1, 1, 1))
            DataLayouts.get_struct(a3, FT, Val(2), CartesianIndex(1, 1, 1))
            DataLayouts.get_struct(a3, FT, Val(3), CartesianIndex(1, 1, 1))
            a4 = rand(FT, 1, 1, 1, 1)
            DataLayouts.get_struct(a4, FT, Val(1), CartesianIndex(1, 1, 1))
            DataLayouts.get_struct(a4, FT, Val(2), CartesianIndex(1, 1, 1))
            DataLayouts.get_struct(a4, FT, Val(3), CartesianIndex(1, 1, 1))
            DataLayouts.get_struct(a4, FT, Val(4), CartesianIndex(1, 1, 1))
            a5 = rand(FT, 1, 1, 1, 1, 1)
            DataLayouts.get_struct(a5, FT, Val(1), CartesianIndex(1, 1, 1))
            DataLayouts.get_struct(a5, FT, Val(2), CartesianIndex(1, 1, 1))
            DataLayouts.get_struct(a5, FT, Val(3), CartesianIndex(1, 1, 1))
            DataLayouts.get_struct(a5, FT, Val(4), CartesianIndex(1, 1, 1))
            DataLayouts.get_struct(a5, FT, Val(5), CartesianIndex(1, 1, 1))

            # DataLayouts
            a1 = rand(FT, 1)
            DataLayouts.set_struct!(a1, a1[1], Val(1), CartesianIndex(1))
            a2 = rand(FT, 1, 1)
            DataLayouts.set_struct!(a2, a2[1], Val(1), CartesianIndex(1, 1))
            DataLayouts.set_struct!(a2, a2[1], Val(2), CartesianIndex(1, 1))
            a3 = rand(FT, 1, 1, 1)
            DataLayouts.set_struct!(a3, a3[1], Val(1), CartesianIndex(1, 1, 1))
            DataLayouts.set_struct!(a3, a3[1], Val(2), CartesianIndex(1, 1, 1))
            DataLayouts.set_struct!(a3, a3[1], Val(3), CartesianIndex(1, 1, 1))
            a4 = rand(FT, 1, 1, 1, 1)
            DataLayouts.set_struct!(a4, a4[1], Val(1), CartesianIndex(1, 1, 1))
            DataLayouts.set_struct!(a4, a4[1], Val(2), CartesianIndex(1, 1, 1))
            DataLayouts.set_struct!(a4, a4[1], Val(3), CartesianIndex(1, 1, 1))
            DataLayouts.set_struct!(a4, a4[1], Val(4), CartesianIndex(1, 1, 1))
            a5 = rand(FT, 1, 1, 1, 1, 1)
            DataLayouts.set_struct!(a5, a5[1], Val(1), CartesianIndex(1, 1, 1))
            DataLayouts.set_struct!(a5, a5[1], Val(2), CartesianIndex(1, 1, 1))
            DataLayouts.set_struct!(a5, a5[1], Val(3), CartesianIndex(1, 1, 1))
            DataLayouts.set_struct!(a5, a5[1], Val(4), CartesianIndex(1, 1, 1))
            DataLayouts.set_struct!(a5, a5[1], Val(5), CartesianIndex(1, 1, 1))
        end # FT
    end # @compile_workload
end # @setup_workload

end # module
