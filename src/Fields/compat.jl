

# for compatibility with OrdinaryDiffEq
# Based on ApproxFun definitions
#  https://github.com/SciML/RecursiveArrayTools.jl/blob/6e779acb321560c75e27739a89ae553cd0f332f1/src/init.jl#L8-L12
# and
#  https://discourse.julialang.org/t/shallow-water-equations-with-differentialequations-jl/2691/16?u=simonbyrne

# This one is annoying
#  https://github.com/SciML/OrdinaryDiffEq.jl/blob/181dcf265351ed3c02437c89a8d2af3f6967fa85/src/initdt.jl#L80
Base.any(f, field::Field) = any(f, parent(field))

import RecursiveArrayTools

RecursiveArrayTools.recursive_unitless_eltype(field::Field) = typeof(field)
RecursiveArrayTools.recursive_unitless_bottom_eltype(field::Field) =
    RecursiveArrayTools.recursive_unitless_bottom_eltype(parent(field))
RecursiveArrayTools.recursive_bottom_eltype(field::Field) =
    RecursiveArrayTools.recursive_bottom_eltype(parent(field))

function RecursiveArrayTools.recursivecopy!(dest::F, src::F) where {F <: Field}
    copy!(parent(dest), parent(src))
    return dest
end

RecursiveArrayTools.recursivecopy(field::Field) = copy(field)
# avoid call to deepcopy
RecursiveArrayTools.recursivecopy(a::AbstractArray{<:Field}) =
    map(RecursiveArrayTools.recursivecopy, a)

function init_diffeq()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        function OrdinaryDiffEq.calculate_residuals!(
            out::Fields.Field,
            ũ::Fields.Field,
            u₀::Fields.Field,
            u₁::Fields.Field,
            α,
            ρ,
            internalnorm,
            t,
        )
            OrdinaryDiffEq.calculate_residuals!(
                parent(out),
                parent(ũ),
                parent(u₀),
                parent(u₁),
                α,
                ρ,
                internalnorm,
                t,
            )
        end
    end
end
