

using ClimateMachineCore


function tendency!(dY, Y, h_s, t)
    
    # hu = Y.h .* Y.u
    # dY.v = - divergance(hu)
    # divergence!(dY.v, hu)

    Nh = Topologies.nlocalelems(Y)
    space = Fields.space(dY.u)
    for h in 1:Nh
        h_slab = slab(Y.h, h)
        u_slab = slab(Y.u, h)
        
        hu_slab = h_slab .* u_slab
        
        dY_v_slab = slab(dY.v, h)  
        dh_slab .= Operators.slab_weak_divergence(hu_slab)
        
        slab_divergence!(slab(dY.v, h_slab), 
    
        # E = @. (Y.h + h_s) * grav + norm(Y.u)
        E_slab = (h_slab .+ h_s) .* grav .+ norm.(u_slab) 
        curl_u = curl(u_slab)
        loc_J = slab(space.local_geometry, h)
        @. slab(dY.u, h) += loc_J * (u_slab × (f + curlu))

    end

    # dY.v = - divergance(hu)
    # divergence!(dY.v, hu)
   
    # E = @. (Y.h + h_s) * grav + norm(Y.u)
   
    # dY.u = -gradient(E) 
    curlu = curl(u)
    J =  
    @. dY.u += J * (Y.u × (f + curlu))
    gradient!(E)
end


# FieldStyle => per node
# StencilStyle => per stencil window
# CompositeStencilStyle => FieldStyle on top of a StencilStyle

# VolumeStyle per slab
#    CompositeVolumeStyle => 


# @. -divergence(h * u) => 