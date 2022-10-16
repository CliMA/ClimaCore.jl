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

runmpi(joinpath(@__DIR__, "dtopo4.jl"), ntasks = 4)
