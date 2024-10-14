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
