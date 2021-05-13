#####
##### Hybrid mesh
#####

struct Hybrid3DMesh{M2D, MCOL}
    mesh2D::M2D
    mesh_col::MCOL
end
