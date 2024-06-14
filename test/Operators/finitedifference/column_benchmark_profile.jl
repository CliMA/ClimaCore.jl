#= This script is helpful for targeting specific kernels =#
include("benchmark_column_utils.jl")

function apply_kernel!(cfield, ffield)
    # op_DivergenceF2C!(cfield, ffield)
    # op_CurlC2F!(cfield, ffield)
    # bcs = (; inner = set_upwind_biased_3_bcs(cfield), outer = set_value_contra3_bcs(cfield))
    # op_divUpwind3rdOrderBiasedProductC2F!(cfield, ffield, bcs)
    # op_2mul_1add!(xarr, yarr, D, U)
    # bcs = (; inner = (), outer = set_value_divgrad_uₕ_bcs(cfield))
    bcs = (; inner = (), outer = set_value_divgrad_uₕ_maybe_field_bcs(cfield))
    op_divgrad_uₕ!(cfield, ffield, bcs)
end

function apply_kernel_loop!(cfield, ffield)
    for _ in 1:10
        apply_kernel!(cfield, ffield) # compile
    end
end

cspace = TU.ColumnCenterFiniteDifferenceSpace(Float64; zelem = 1000)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)
cfield = fill(field_vars(Float64), cspace)
ffield = fill(field_vars(Float64), fspace)
apply_kernel_loop!(cfield, ffield) # compile

import Profile
@info "collect profile"
Profile.clear()
prof = Profile.@profile apply_kernel_loop!(cfield, ffield)
results = Profile.fetch()
Profile.clear()

import ProfileCanvas
ProfileCanvas.html_file(joinpath("flame.html"), results)

nothing
