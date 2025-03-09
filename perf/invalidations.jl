# From: https://timholy.github.io/SnoopCompile.jl/dev/tutorials/invalidations/#Tutorial-on-@snoop_invalidations
ENV["TEST_NAME"] = "sphere/held_suarez_rhoe"
ENV["FLOAT_TYPE"] = "Float64"
using SnoopCompileCore
invalidations = @snoop_invalidations begin
    include(joinpath(dirname(@__DIR__), "examples", "hybrid", "driver.jl"))
    nothing
end;

import SnoopCompile
import PrettyTables # load report_invalidations
SnoopCompile.report_invalidations(;
    invalidations,
    process_filename = x -> last(split(x, "packages/")),
)
