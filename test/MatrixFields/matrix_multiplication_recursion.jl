import ClimaCore
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

import ClimaCore: Spaces, Geometry, Operators, Fields, MatrixFields
import ClimaComms
ClimaComms.@import_required_backends

# Set up operators
const FT = Float64
# Create divergence operator
const divf2c_op = Operators.DivergenceF2C()
const divf2c_matrix = MatrixFields.operator_matrix(divf2c_op)
# Create gradient operator, and set gradient at boundaries to 0
const gradc2f_op = Operators.GradientC2F(
    top = Operators.SetGradient(Geometry.WVector(FT(0))),
    bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
)
const gradc2f_matrix = MatrixFields.operator_matrix(gradc2f_op)
# Create interpolation operator
const interpc2f_op = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

# Construct 3D space (column)
boundary_names = (:bottom, :top)
zlim = (FT(-1.5), FT(0))
column = ClimaCore.Domains.IntervalDomain(
    ClimaCore.Geometry.ZPoint{FT}(zlim[1]),
    ClimaCore.Geometry.ZPoint{FT}(zlim[2]);
    boundary_names = boundary_names,
)

nelements = 150
mesh = ClimaCore.Meshes.IntervalMesh(column; nelems = nelements)
device = ClimaComms.device()
subsurface_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(device, mesh)
obtain_face_space(cs::ClimaCore.Spaces.CenterFiniteDifferenceSpace) =
    ClimaCore.Spaces.FaceFiniteDifferenceSpace(cs)
function obtain_surface_space(cs::ClimaCore.Spaces.CenterFiniteDifferenceSpace)
    fs = obtain_face_space(cs)
    return ClimaCore.Spaces.level(
        fs,
        ClimaCore.Utilities.PlusHalf(ClimaCore.Spaces.nlevels(fs) - 1),
    )
end
surface_space = obtain_surface_space(subsurface_space)

# Set up additional operators on the space
const dfluxBCdY = Geometry.Covariant3Vector.(Fields.ones(surface_space))
const topBC_op = Operators.SetBoundaryOperator(
    top = Operators.SetValue(dfluxBCdY),
    bottom = Operators.SetValue(Geometry.Covariant3Vector(zero(FT))),
)

# Set up fields
K = Fields.zeros(subsurface_space)
dψdϑ_res = Fields.zeros(subsurface_space) .+ FT(115.901)

tridiag_type = MatrixFields.TridiagonalMatrixRow{FT}
dest_field = Fields.Field(tridiag_type, subsurface_space)
fill!(parent(dest_field), NaN)

# Test field update with multiple nested operations
function update_field!(dest_field, K, dψdϑ_res)
    @. dest_field =
        divf2c_matrix() * (
            MatrixFields.DiagonalMatrixRow(interpc2f_op(-K)) *
            gradc2f_matrix() *
            MatrixFields.DiagonalMatrixRow(dψdϑ_res) +
            MatrixFields.LowerDiagonalMatrixRow(
                topBC_op(Geometry.Covariant3Vector(zero(interpc2f_op(K)))),
            )
        )
end

update_field!(dest_field, K, dψdϑ_res)
