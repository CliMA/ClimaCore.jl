#!/bin/bash
module purge
module load julia/1.8.1 openmpi/4.1.1 hdf5/1.12.1-ompi411

export CLIMACORE_DISTRIBUTED="MPI"
export JULIA_HDF5_PATH=""
export GKSwstype=100 # if plotting

julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -e 'using Pkg; Pkg.build("MPI"); Pkg.build("HDF5")'
julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project -e 'include("test/mpi_tests/run_mpi.jl")'
