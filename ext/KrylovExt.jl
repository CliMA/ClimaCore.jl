module KrylovExt

import ClimaComms
import ClimaCore: Fields
import Krylov

Krylov.ktypeof(x::Fields.FieldVector) =
    ClimaComms.array_type(x){eltype(parent(x)), 1}

end
