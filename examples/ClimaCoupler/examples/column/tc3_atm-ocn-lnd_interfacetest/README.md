# Atmos-Ocean-Soil Columns

## Notes
- extracting states in Oceananigans
```
    ocean_Nz = ocean_sim.model.grid.Nz
    @inbounds begin
        ocean_surface_u = ocean_sim.model.velocities.u[1, 1, ocean_Nz]
        ocean_surface_v = ocean_sim.model.velocities.v[1, 1, ocean_Nz]
        ocean_surface_T = ocean_sim.model.tracers.T[1, 1, ocean_Nz]
    end
```
-