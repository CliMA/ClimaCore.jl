# Deprecated methods

import ClimaComms
import .Grids: FiniteDifferenceGrid, CellFace, CellCenter
import .Spaces:
    CenterFiniteDifferenceSpace,
    FaceFiniteDifferenceSpace,
    FiniteDifferenceSpace
import .Topologies: IntervalTopology
import .Meshes: IntervalMesh
import .DataLayouts

@deprecate IntervalTopology(mesh::IntervalMesh) IntervalTopology(
    ClimaComms.SingletonCommsContext(ClimaComms.device()),
    mesh,
)

@deprecate FaceFiniteDifferenceSpace(mesh::IntervalMesh) FiniteDifferenceSpace(
    FiniteDifferenceGrid(ClimaComms.device(), mesh),
    CellFace(),
)
@deprecate CenterFiniteDifferenceSpace(mesh::IntervalMesh) FiniteDifferenceSpace(
    FiniteDifferenceGrid(ClimaComms.device(), mesh),
    CellCenter(),
)

@deprecate FiniteDifferenceGrid(mesh::IntervalMesh) FiniteDifferenceGrid(
    IntervalTopology(ClimaComms.device(), mesh),
)

@deprecate DataLayouts.IJFH{S, Nij}(
    ::Type{ArrayType},
    nelements,
) where {S, Nij, ArrayType} DataLayouts.IJFH{S, Nij, nelements}(ArrayType) false

@deprecate DataLayouts.IJFH{S, Nij}(
    array::AbstractArray{T, 4},
) where {S, Nij, T} DataLayouts.IJFH{S, Nij, size(array, 4)}(array) false
@deprecate DataLayouts.IFH{S, Ni}(array::AbstractArray{T, 3}) where {S, Ni, T} DataLayouts.IFH{
    S,
    Ni,
    size(array, 3),
}(
    array,
) false
@deprecate DataLayouts.IFH{S, Ni}(
    ::Type{ArrayType},
    nelements,
) where {S, Ni, ArrayType} DataLayouts.IFH{S, Ni, nelements}(ArrayType) false
@deprecate DataLayouts.VIJFH{S, Nv, Nij}(
    array::AbstractArray{T, 5},
) where {S, Nv, Nij, T} DataLayouts.VIJFH{S, Nv, Nij, size(array, 5)}(array) false
@deprecate DataLayouts.VIFH{S, Nv, Ni}(
    array::AbstractArray{T, 4},
) where {S, Nv, Ni, T} DataLayouts.VIFH{S, Nv, Ni, size(array, 4)}(array) false
