module KrylovExt

import ClimaComms
import ClimaCore: Fields
import Krylov

Krylov.ktypeof(x::Fields.FieldVector) = typeof(x)

end
