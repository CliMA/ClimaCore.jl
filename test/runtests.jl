include("diffusion.jl")

include("data.jl")
include("grid.jl")
include("quadrature.jl")
include("mesh.jl")
include("field.jl")
include("operators.jl")

if "CUDA" in ARGS
    include("gpu/cuda.jl")
    include("gpu/data.jl")
end
