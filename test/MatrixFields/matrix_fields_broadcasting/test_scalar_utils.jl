import BandedMatrices: band
import LinearAlgebra: I, mul!

include(joinpath("..", "matrix_field_test_utils.jl"))

const FT = Float64
const center_space, face_space = test_spaces(FT)

seed!(1) # ensures reproducibility
const ᶜvec = random_field(FT, center_space)
const ᶠvec = random_field(FT, face_space)
const ᶜᶜmat = random_field(DiagonalMatrixRow{FT}, center_space)
const ᶜᶠmat = random_field(BidiagonalMatrixRow{FT}, center_space)
const ᶠᶠmat = random_field(TridiagonalMatrixRow{FT}, face_space)
const ᶠᶜmat = random_field(QuaddiagonalMatrixRow{FT}, face_space)
