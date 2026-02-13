import BandedMatrices: band
import LinearAlgebra: I, mul!

include(joinpath("..", "matrix_field_test_utils.jl"))

if !(@isdefined(unit_test_field_broadcast))
    const FT = Float32
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
    using ClimaCore: Geometry, Operators, MatrixFields
import ClimaCore

# Alternatively, we could use Vec₁₂₃, Vec³, etc., if that is more readable.
const C1 = Geometry.Covariant1Vector
const C2 = Geometry.Covariant2Vector
const C12 = Geometry.Covariant12Vector
const C3 = Geometry.Covariant3Vector
const C123 = Geometry.Covariant123Vector
const CT1 = Geometry.Contravariant1Vector
const CT2 = Geometry.Contravariant2Vector
const CT12 = Geometry.Contravariant12Vector
const CT3 = Geometry.Contravariant3Vector
const CT123 = Geometry.Contravariant123Vector
const UVW = Geometry.UVWVector

const divₕ = Operators.Divergence()
const wdivₕ = Operators.WeakDivergence()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()

const ᶜinterp = Operators.InterpolateF2C()
const ᶜdivᵥ = Operators.DivergenceF2C()
const ᶜgradᵥ = Operators.GradientF2C()

# Tracers do not have advective fluxes through the top and bottom cell faces.
const ᶜadvdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(CT3(0)),
    top = Operators.SetValue(CT3(0)),
)

# Subsidence has extrapolated tendency at the top, and has no flux at the bottom.
# TODO: This is not accurate and causes some issues at the domain top.
const ᶜsubdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(CT3(0)),
    top = Operators.Extrapolate(),
)

# Precipitation has no flux at the top, but it has free outflow at the bottom.
const ᶜprecipdivᵥ = Operators.DivergenceF2C(top = Operators.SetValue(CT3(0)))

const ᶠright_bias = Operators.RightBiasedC2F() # for free outflow in ᶜprecipdivᵥ
const ᶜleft_bias = Operators.LeftBiasedF2C()
const ᶜright_bias = Operators.RightBiasedF2C()

const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶠwinterp = Operators.WeightedInterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)


const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(0)),
    top = Operators.SetGradient(C3(0)),
)
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(CT12(0, 0)),
    top = Operators.SetCurl(CT12(0, 0)),
)
const upwind_biased_grad = Operators.UpwindBiasedGradient()
const ᶠupwind1 = Operators.UpwindBiasedProductC2F()
const ᶠupwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.ThirdOrderOneSided(),
    top = Operators.ThirdOrderOneSided(),
)

const ᶜinterp_matrix = MatrixFields.operator_matrix(ᶜinterp)
const ᶜleft_bias_matrix = MatrixFields.operator_matrix(ᶜleft_bias)
const ᶜright_bias_matrix = MatrixFields.operator_matrix(ᶜright_bias)
const ᶜdivᵥ_matrix = MatrixFields.operator_matrix(ᶜdivᵥ)
const ᶜadvdivᵥ_matrix = MatrixFields.operator_matrix(ᶜadvdivᵥ)
const ᶜprecipdivᵥ_matrix = MatrixFields.operator_matrix(ᶜprecipdivᵥ)
const ᶠright_bias_matrix = MatrixFields.operator_matrix(ᶠright_bias)
const ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)
const ᶠwinterp_matrix = MatrixFields.operator_matrix(ᶠwinterp)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)
const ᶠupwind1_matrix = MatrixFields.operator_matrix(ᶠupwind1)
const ᶠupwind3_matrix = MatrixFields.operator_matrix(ᶠupwind3)
∂ᶠρχ_dif_flux_∂ᶜχ = fill(zero(MatrixFields.BidiagonalMatrixRow{C3{Float32}}), face_space)

# Helper functions to extract components of vectors
u_component(u::Geometry.LocalVector) = u.u
v_component(u::Geometry.LocalVector) = u.v
w_component(u::Geometry.LocalVector) = u.w
a = fill(zero(CT3{Float32}), face_space)
 c = fill(1.0f0, center_space)
 f = fill(1.0f0, face_space)
 c1 = fill(1.0f0, center_space)
 c2 =fill(1.0f0, center_space)
 f2 = similar(ᶠlg, BidiagonalMatrixRow{C3{FT}})

 const ᶠlin_vanleer = Operators.LinVanLeerC2F(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
        constraint = Operators.MonotoneLocalExtrema(), # (Mono5)
    )
#     import LazyBroadcast: lazy
#     import Thermodynamics as TD
#     import ClimaAtmos.Parameters as CAP
#     import ClimaAtmos as CA
#  params = CA.ClimaAtmosParameters(Float32)
#  thermo_params = params.thermodynamics_params;
#  ᶜts = similar(c1, TD.PhaseEquil{Float32})
#  parent(ᶜts) .= 1.0f0
#  l1 = @. lazy(c1 / c2)
#  l2 = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, l1))
#  e2 = @. TD.total_specific_enthalpy(thermo_params, ᶜts, l1)
#  g = @. ᶠlin_vanleer(a, e2, 1.0f0)
#   g = @. ᶠlin_vanleer(a, l2, 1.0f0)
# @. ᶠlin_vanleer(a, b, 0.1f0)

# @.  ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(ᶠinterp(b))
#   @.  -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠinterp(b) / 1.0f0)
end

function unit_test_field_broadcast(
    result,
    bc;
    ref_set_result!,
    allowed_max_eps_error = 10,
)
    result_copy = copy(result)
    set_result!(result, bc)
    # Test that set_result! sets the same value as get_result.
    @test result == result_copy

    ref_result = similar(result)
    ref_set_result!(ref_result)
    max_error = mapreduce(
        (a, b) -> (abs(a - b)),
        max,
        parent(result),
        parent(ref_result),
    )
    max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))

    # Test that set_result! is performant and correct when compared
    # against ref_set_result!.
    @test max_eps_error <= allowed_max_eps_error
    return nothing
end

function opt_test_field_broadcast(result, bc; ref_set_result!)
    time = @benchmark set_result!(result, bc)
    ref_result = similar(result)
    ref_time = @benchmark ref_set_result!(ref_result)
    print_time_comparison(; time, ref_time)

    # Test get_result and set_result! for type instabilities, and test
    # set_result! for allocations. Ignore the type instabilities in CUDA and
    # the allocations they incur.
    @test_opt ignored_modules = cuda_frames materialize(bc)
    @test_opt ignored_modules = cuda_frames set_result!(result, bc)
    using_cuda || @test (@allocated set_result!(result, bc)) == 0

    # Test ref_set_result! for type instabilities and allocations to
    # ensure that the performance comparison is fair.
    @test_opt ignored_modules = cuda_frames ref_set_result!(ref_result)
    using_cuda || @test (@allocated ref_set_result!(ref_result)) == 0
    return nothing
end
