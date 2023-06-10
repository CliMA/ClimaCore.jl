function jobs_from_yaml(yaml_file; filter_name = nothing)
    data = YAML.load_file(yaml_file)
    steps = filter(x -> x isa Dict && haskey(x, "steps"), data["steps"])
    i_unit_tests = findall(steps) do step
        occursin("Unit: ", step["group"])
    end
    @assert i_unit_tests ≠ nothing "Unit tests not found"
    unit_tests = collect(Iterators.flatten(map(i_unit_tests) do t
        steps[t]["steps"]
    end))
    return unit_tests
end
