"""
    ClimaCorePlots

Deprecated compatibility shim: the Plots recipes now live in the
`ClimaCoreRecipesBaseExt` extension of ClimaCore, which loads automatically when
both `ClimaCore` and `Plots` (which has `RecipesBase` as a dependency) are
loaded. Loading this package triggers the extension; it provides nothing else.
"""
module ClimaCorePlots

import Plots, ClimaCore

end # module
