include("data.jl")

if "CUDA" in ARGS
    include("cuda.jl")
    include("gpu/data.jl")
end
