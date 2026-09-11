# Write and read checkpoints

`ClimaCore.InputOutput` writes fields and field vectors, together with the
spaces they live on, to HDF5 files, and reads them back into fields on
reconstructed spaces. A file written by one run restarts another.

## Prerequisites

HDF5.jl is a dependency of ClimaCore. Distributed writing and reading need an
HDF5 library built with MPI support; see the
[HDF5.jl documentation](https://juliaio.github.io/HDF5.jl/stable/#Parallel-HDF5).

## Steps

 1. Write a state. The `do`-block form takes the `ClimaComms` context and
    closes the file; the domain, mesh, topology, and space are written once and
    referenced by every field on them.

    ```@example checkpoints
    import ClimaComms
    ClimaComms.@import_required_backends
    using ClimaCore.CommonSpaces
    import ClimaCore: Fields, Spaces, InputOutput
    context = ClimaComms.context()
    space = ExtrudedCubedSphereSpace(;
        radius = 6.371e6, h_elem = 4, n_quad_points = 4,
        z_elem = 5, z_min = 0.0, z_max = 30e3, staggering = CellCenter(),
    )
    z = Fields.coordinate_field(space).z
    c = map(zᵢ -> (; ρ = exp(-zᵢ / 8e3), θ = 300 + zᵢ / 100), z)
    Y = Fields.FieldVector(; c)
    filename = tempname() * ".hdf5"
    InputOutput.HDF5Writer(filename, context) do writer
        InputOutput.write!(writer, Y, "Y")
    end
    filesize(filename) > 0
    ```

 2. Read it back. The space is rebuilt from the file, so the reading process
    needs only the file.

    ```@example checkpoints
    Y_restart = InputOutput.HDF5Reader(filename, context) do reader
        InputOutput.read_field(reader, "Y")
    end
    parent(Y_restart.c.ρ) == parent(Y.c.ρ)
    ```

 3. Read one component of a `FieldVector` with its slash path (`"Y/c"` for
    component `c` of `Y`), and inspect what a file holds through the reader's
    caches:

    ```@example checkpoints
    InputOutput.HDF5Reader(filename, context) do reader
        c = InputOutput.read_field(reader, "Y/c")
        (propertynames(c), collect(keys(reader.space_cache)))
    end
    ```

 4. Append to an existing file with `overwrite = false`, and write several
    fields under distinct names; fields are written each time, so the same field
    can be written under different names at different times.

 5. In a distributed run, the context is an `MPICommsContext`, and every rank
    writes its elements and reads them back onto a topology distributed the
    same way.

## What is stored

The file records the float type, the mesh and its stretching, the topology
and element order, the quadrature, the spectral-element discretization
(`Grids.CG()` or `Grids.DG()`; files written before this attribute existed
read back as continuous), the staggering, and the hypsography. A field read
back has the same data layout as the one written, so `parent(restart) == parent(original)` holds bit for bit on the same device.

## Restarting a run

A restart reads the state and rebuilds the model's caches from it; the space
comes from the file. ClimaAtmos's
[restart page](https://clima.github.io/ClimaAtmos.jl/stable/restarts/)
shows how a model wraps this.
