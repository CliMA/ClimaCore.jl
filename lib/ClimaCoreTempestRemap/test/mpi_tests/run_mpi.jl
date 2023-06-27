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
    runmpi(joinpath(@__DIR__, "distr_regrid_example.jl"), ntasks = 2)
end
