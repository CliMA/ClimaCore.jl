include("data.jl")

if "CUDA" in ARGS
    include("cuda.jl")
end
