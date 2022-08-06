module Operators

import LinearAlgebra

using StaticArrays

import ..enable_threading, ..slab, ..slab_args, ..column, ..column_args
import ..DataLayouts: DataLayouts, Data2D, DataSlab2D
import ..Geometry: Geometry, Covariant12Vector, Contravariant12Vector, ⊗
import ..Spaces: Spaces, Quadratures, AbstractSpace
import ..Topologies
import ..Meshes
import ..Fields: Fields, Field

using ..RecursiveApply

include("spectralelement.jl")
include("numericalflux.jl")
include("finitedifference.jl")
include("stencilcoefs.jl")
include("operator2stencil.jl")
include("pointwisestencil.jl")
include("remapping.jl")

if VERSION >= v"1.7.0"
    if hasfield(Method, :recursion_relation)
        dont_limit = (args...) -> true
        for m in methods(slab)
            m.recursion_relation = dont_limit
        end
        for m in methods(slab_args)
            m.recursion_relation = dont_limit
        end
        for m in methods(_apply_slab)
            m.recursion_relation = dont_limit
        end
        for m in methods(_apply_slab_args)
            m.recursion_relation = dont_limit
        end
    end
end

end # module
