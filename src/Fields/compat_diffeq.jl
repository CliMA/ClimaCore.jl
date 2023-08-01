import DiffEqBase, RecursiveArrayTools, Static

# for compatibility with OrdinaryDiffEq
# Based on ApproxFun definitions
#  https://github.com/SciML/RecursiveArrayTools.jl/blob/6e779acb321560c75e27739a89ae553cd0f332f1/src/init.jl#L8-L12
# and
#  https://discourse.julialang.org/t/shallow-water-equations-with-differentialequations-jl/2691/16?u=simonbyrne

# This one is annoying
#  https://github.com/SciML/OrdinaryDiffEq.jl/blob/181dcf265351ed3c02437c89a8d2af3f6967fa85/src/initdt.jl#L80
Base.any(f, field::Field) = any(f, parent(field))

Base.muladd(x, field::Field) = muladd.(x, field)

Base.similar(field::F, ::Type{F}) where {F <: Field} = similar(field)

Base.vec(field::Field) = vec(parent(field))

