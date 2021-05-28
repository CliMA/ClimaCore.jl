#####
##### Hybrid mesh
#####

struct Hybrid3DSpace{M2D, MCOL}
    space_horizontal::M2D
    space_column::MCOL
end
