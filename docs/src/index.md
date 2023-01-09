# ClimaCore.jl

ClimaCore.jl constitutes the dynamical core (dycore) of the atmosphere and land
models for [CliMA](https://clima.caltech.edu/)'s Earth System Model (ESM). ClimaCore.jl provides flexible
and composable discretization tools to solve the governing equations of the ESM
component models. In fact, ClimaCore.jl's high-level application programming
interface (API) facilitates modularity and composition of differential operators
and the definition of flexible discretizations. This, in turn, is coupled
with low-level APIs that support different data layouts, specialized implementations,
and flexible models for threading, to better face high-performance optimization,
data storage, and scalability challenges on modern HPC architectures.
