#=
`Krylov.ktypeof` is defined here, if `Krylov`
is loaded to avoid the need for type piracy.

```julia
using Krylov # to load this file
using ClimaCore
```

TODO: Use package extensions when we upgrade to Julia 1.9.
=#

import ClimaComms
import .Krylov
Krylov.ktypeof(x::Fields.FieldVector) =
    ClimaComms.array_type(x){eltype(parent(x)), 1}
