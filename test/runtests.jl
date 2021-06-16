using Base: operator_associativity

include("data1d.jl")
include("data.jl")
include("grid.jl")
include("quadrature.jl")
include("spaces.jl")
include("field.jl")
include("operators.jl")
include("fdspaces.jl")
include("fielddiffeq.jl")
#include("diffusion.jl")

if "CUDA" in ARGS
    include("gpu/cuda.jl")
    include("gpu/data.jl")
end
