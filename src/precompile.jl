using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    include(
        joinpath("..", "test", "TestUtilities", "TestUtilities.jl"),
    )
    import .TestUtilities as TU
    @compile_workload begin
        TU.all_spaces(Float64)
        TU.all_spaces(Float32)
    end
end
