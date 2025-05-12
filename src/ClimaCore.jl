module ClimaCore

using PkgVersion
const VERSION = PkgVersion.@Version
import ClimaComms

include("DebugOnly/DebugOnly.jl")
include("interface.jl")
include("devices.jl")
include("Utilities/Utilities.jl")
include("RecursiveApply/RecursiveApply.jl")
include("DataLayouts/DataLayouts.jl")
include("Geometry/Geometry.jl")
include("Domains/Domains.jl")
include("Meshes/Meshes.jl")
include("Topologies/Topologies.jl")
include("Quadratures/Quadratures.jl")
include("Grids/Grids.jl")
include("Spaces/Spaces.jl")
include("Fields/Fields.jl")
include("Operators/Operators.jl")
include("MatrixFields/MatrixFields.jl")
include("Hypsography/Hypsography.jl")
include("Limiters/Limiters.jl")
include("InputOutput/InputOutput.jl")
include("Remapping/Remapping.jl")
include("CommonGrids/CommonGrids.jl")
include("CommonSpaces/CommonSpaces.jl")

include("deprecated.jl")
include("to_device.jl")

# For complex nested types (ex. wrapped SMatrix / broadcast expressions) we hit
# a recursion limit and de-optimize We know the recursion will terminate due to
# the fact that bitstype fields cannot be self referential so there are no
# cycles in these methods (bounded tree) TODO: enforce inference termination
# some other way

if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(Operators.column)
        m.recursion_relation = dont_limit
    end
    for m in methods(Operators.column_args)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.multiply_matrix_at_index)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.unique_and_non_overlapping_values)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.union_values)
        m.recursion_relation = dont_limit
    end
    for m in methods(Operators.reconstruct_placeholder_broadcasted)
        m.recursion_relation = dont_limit
    end
    for m in methods(Operators._reconstruct_placeholder_broadcasted)
        m.recursion_relation = dont_limit
    end
	for m in methods(Operators.get_node)
	    m.recursion_relation = dont_limit
	end
    for m in methods(MatrixFields.has_field)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.get_field)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.broadcasted_has_field)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.broadcasted_get_field)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.wrapped_prop_names)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.filtered_child_names)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.subtree_at_name)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.is_valid_name)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.get_subtree_at_name)
        m.recursion_relation = dont_limit
    end
    for m in methods(MatrixFields.concrete_field_vector_within_subtree)
        m.recursion_relation = dont_limit
    end
    for m in methods(DataLayouts.get_struct_linear)
        m.recursion_relation = dont_limit
    end
    for m in methods(DataLayouts.set_struct_linear!)
        m.recursion_relation = dont_limit
    end
    for m in methods(DataLayouts.get_struct)
        m.recursion_relation = dont_limit
    end
    for m in methods(DataLayouts.set_struct!)
        m.recursion_relation = dont_limit
    end
    for m in methods(Operators.call_bc_f)
        m.recursion_relation = dont_limit
    end
    for m in methods(Operators.getidx)
        m.recursion_relation = dont_limit
    end
    for m in methods(Operators.stencil_interior)
        m.recursion_relation = dont_limit
    end
    for m in methods(Operators.stencil_left_boundary)
        m.recursion_relation = dont_limit
    end
    for m in methods(Operators.stencil_right_boundary)
        m.recursion_relation = dont_limit
    end
end

end # module
