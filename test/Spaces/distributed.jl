using MPI

function runmpi(file; ntasks = 1)
    MPI.mpiexec() do cmd
        Base.run(
            `$cmd -n $ntasks $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $file`;
            wait = true,
        )
        true
    end
end

if !Sys.iswindows()
    runmpi(joinpath(@__DIR__, "distributed", "ddss2.jl"), ntasks = 2)
    runmpi(joinpath(@__DIR__, "distributed", "ddss3.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "distributed", "ddss4.jl"), ntasks = 4)
    runmpi(joinpath(@__DIR__, "distributed", "gather4.jl"), ntasks = 4)
end
