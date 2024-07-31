import BandedMatrices: band
import LinearAlgebra: I, mul!

include(joinpath("..", "matrix_field_test_utils.jl"))

const FT = Float64
const center_space, face_space = test_spaces(FT)

const ᶜlg = Fields.local_geometry_field(center_space)
const ᶠlg = Fields.local_geometry_field(face_space)

seed!(1) # ensures reproducibility
const ᶜvec = random_field(FT, center_space)
const ᶠvec = random_field(FT, face_space)
const ᶜᶠmat = random_field(BidiagonalMatrixRow{FT}, center_space)
const ᶜᶠmat2 = random_field(BidiagonalMatrixRow{FT}, center_space)
const ᶜᶠmat3 = random_field(BidiagonalMatrixRow{FT}, center_space)
const ᶠᶜmat = random_field(QuaddiagonalMatrixRow{FT}, face_space)
const ᶠᶜmat2 = random_field(QuaddiagonalMatrixRow{FT}, face_space)
const ᶠᶜmat3 = random_field(QuaddiagonalMatrixRow{FT}, face_space)

const ᶜᶠmat_AC1 =
    map(row -> map(adjoint ∘ Geometry.Covariant1Vector, row), ᶜᶠmat)
const ᶜᶠmat_C12 = map(
    (row1, row2) -> map(Geometry.Covariant12Vector, row1, row2),
    ᶜᶠmat2,
    ᶜᶠmat3,
)
const ᶠᶜmat_AC1 =
    map(row -> map(adjoint ∘ Geometry.Covariant1Vector, row), ᶠᶜmat)
const ᶠᶜmat_C12 = map(
    (row1, row2) -> map(Geometry.Covariant12Vector, row1, row2),
    ᶠᶜmat2,
    ᶠᶜmat3,
)

const ᶜᶠmat_AC1_num =
    map((row1, row2) -> map(tuple, row1, row2), ᶜᶠmat_AC1, ᶜᶠmat)
const ᶜᶠmat_num_C12 =
    map((row1, row2) -> map(tuple, row1, row2), ᶜᶠmat, ᶜᶠmat_C12)
const ᶠᶜmat_C12_AC1 =
    map((row1, row2) -> map(tuple, row1, row2), ᶠᶜmat_C12, ᶠᶜmat_AC1)

const ᶜvec_NT = @. nested_type(ᶜvec, ᶜvec, ᶜvec)
const ᶜᶠmat_NT =
    map((rows...) -> map(nested_type, rows...), ᶜᶠmat, ᶜᶠmat2, ᶜᶠmat3)
const ᶠᶜmat_NT =
    map((rows...) -> map(nested_type, rows...), ᶠᶜmat, ᶠᶜmat2, ᶠᶜmat3)
