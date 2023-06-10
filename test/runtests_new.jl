using Test
using SafeTestsets
using Base: operator_associativity
import YAML

include("yaml_helper.jl")
include("buildkite_yaml_struct.jl")

cc_dir = joinpath(@__DIR__, "..")
yaml_file = joinpath(cc_dir, ".buildkite", "pipeline.yml")
unit_tests = jobs_from_yaml(yaml_file)
bkjs = BuildkiteJob.(unit_tests)

# Avoid recursive inclusion:
filter!(j -> any(x -> !occursin("test/runtests.jl", x), get_files(j)), bkjs)

tabulate_jobs(bkjs; verbose = false)

# for j in bkjs
#     test_job(j)
# end
