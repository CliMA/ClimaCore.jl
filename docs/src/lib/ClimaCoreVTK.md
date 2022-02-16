# ClimaCoreVTK.jl

```@meta
CurrentModule = ClimaCoreVTK
```

ClimaCoreVTK.jl provides functionality for writing ClimaCore fields to VTK files, using the [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl) package.

# Interface

```@docs
writevtk
writepvd
```

# Internal functions

```@docs
vtk_grid
vtk_cells_lagrange
vtk_cells_linear
vtk_grid_space
vtk_cell_space
addfield!
```