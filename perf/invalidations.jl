# From: https://timholy.github.io/SnoopCompile.jl/stable/snoopr/
ENV["TEST_NAME"] = "sphere/held_suarez_rhoe"
ENV["FLOAT_TYPE"] = "Float64"
using SnoopCompileCore
invalidations = @snoopr begin
    include(joinpath(dirname(@__DIR__), "examples", "hybrid", "driver.jl"))
    nothing
end;

import ReportMetrics
ReportMetrics.report_invalidations(;
    job_name = "invalidations",
    invalidations,
    process_filename = x -> last(split(x, "packages/")),
)
