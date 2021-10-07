push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Documenter, ClimaCore

format = Documenter.HTML(
    prettyurls = !isempty(get(ENV, "CI", "")),
    collapselevel = 1,
)

using Literate

#Literate.markdown("tutorials/tutorial.jl", "src")

makedocs(
    sitename = "ClimaCore.jl",
    strict = false,
    format = format,
    checkdocs = :exports,
    clean = true,
    doctest = true,
    modules = [ClimaCore],
    pages = Any[
        "Home" => "index.md",
        "API" => "api.md",
        "Operators" => "operators.md",
        "Tutorial" => "tutorial.md",
    ],
)

deploydocs(
    repo = "github.com/CliMA/ClimaCore.jl.git",
    target = "build",
    push_preview = true,
    devbranch = "main",
    forcepush = true,
)
