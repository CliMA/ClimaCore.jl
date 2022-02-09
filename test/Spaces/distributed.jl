using MPI

function runmpi(file; ntasks = 1)
    # Some MPI runtimes complain if more resources are requested
    # than available.
    if MPI.MPI_LIBRARY == MPI.OpenMPI
        oversubscribe = `--oversubscribe`
    else
        oversubscribe = ``
    end
    MPI.mpiexec() do cmd
        Base.run(
            `$cmd $oversubscribe -np $ntasks $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $file`;
            wait = true,
        )
        true
    end
end

if !Sys.iswindows()
    runmpi(joinpath(@__DIR__, "distributed", "ddss2.jl"), ntasks = 2)
    #runmpi(joinpath(@__DIR__, "distributed", "ddss3.jl"), ntasks = 3)
    #runmpi(joinpath(@__DIR__, "distributed", "ddss4.jl"), ntasks = 4)
end
