module KrylovExt

import ClimaCore: DataLayouts, Fields
import Krylov

function Krylov.ktypeof(x::Fields.FieldVector)
    array_type_unknown_N = typeof(parent(Fields.representative_field(x)))
    array_type_variable_N = DataLayouts.parent_array_type(array_type_unknown_N)
    return typeintersect(array_type_variable_N, AbstractVector) # Set N = 1.
end

Krylov.kcopy!(::Integer, y::AbstractVector, x::Fields.FieldVector) = y .= x

end
