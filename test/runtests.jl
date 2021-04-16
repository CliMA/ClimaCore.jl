include("data.jl")
include("grid.jl")
include("quadrature.jl")
include("mesh.jl")
include("field.jl")

if "CUDA" in ARGS
    include("cuda.jl")
    include("gpu/data.jl")
end
