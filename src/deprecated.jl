# Deprecated methods
#
# TODO: delete. The mesh-only constructor below (deprecated in v0.14.10) is only used
# in-repo by test/Fields/inference_repro.jl.

import ClimaComms
import .Grids: FiniteDifferenceGrid, CellCenter
import .Spaces: CenterFiniteDifferenceSpace, FiniteDifferenceSpace
import .Meshes: IntervalMesh

@deprecate CenterFiniteDifferenceSpace(mesh::IntervalMesh) FiniteDifferenceSpace(
    FiniteDifferenceGrid(ClimaComms.device(), mesh),
    CellCenter(),
)
