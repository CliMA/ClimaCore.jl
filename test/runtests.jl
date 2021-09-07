using Base: operator_associativity

include("recursive.jl")
include("data1d.jl")
include("data2d.jl")
include("data1dx.jl")
include("data2dx.jl")
include("axistensors.jl")
include("grid.jl")
include("quadrature.jl")
include("spaces.jl")
include("field.jl")
include("spectraloperators.jl")
include("fdspaces.jl")
include("fielddiffeq.jl")
include("hybrid.jl")
#include("diffusion2d.jl")

if "CUDA" in ARGS
    include("gpu/cuda.jl")
    include("gpu/data.jl")
end
