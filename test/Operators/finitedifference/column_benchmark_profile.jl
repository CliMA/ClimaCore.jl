#= This script is helpful for targeting specific kernels =#
include("column_benchmark_utils.jl")

function apply_kernel!(cfield, ffield, D, U, xarr, yarr)
    # op_DivergenceF2C!(cfield, ffield)
    op_CurlC2F!(cfield, ffield)
    # bcs = (; inner = set_upwind_biased_3_bcs(cfield), outer = set_value_contra3_bcs(cfield))
    # op_divUpwind3rdOrderBiasedProductC2F!(cfield, ffield, bcs)
    # op_2mul_1add!(xarr, yarr, D, U)
    # bcs = (; inner = (), outer = set_value_divgrad_uₕ_bcs(cfield))
    # op_divgrad_uₕ!(cfield, ffield, bcs)
end

function apply_kernel_loop!(cfield, ffield, D, U, xarr, yarr)
    for _ in 1:100000
        apply_kernel!(cfield, ffield, D, U, xarr, yarr) # compile
    end
end

import Profile
import PProf

(; cfield, ffield, vars_contig) = get_fields(1000, Float64)
(; L, D, U, xarr, yarr) = vars_contig
apply_kernel_loop!(cfield, ffield, D, U, xarr, yarr) # compile

Profile.clear_malloc_data()
prof = Profile.@profile begin
    apply_kernel_loop!(cfield, ffield, D, U, xarr, yarr)
end
PProf.pprof()

# http://localhost:57599/ui/flamegraph?tf

nothing
