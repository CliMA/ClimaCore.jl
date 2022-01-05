using Documenter, DocumenterCitations, Literate
using ClimaCore, ClimaCoreVTK, ClimaCoreMakie, ClimaCorePlots

format = Documenter.HTML(
    prettyurls = !isempty(get(ENV, "CI", "")),
    collapselevel = 1,
)

if !@isdefined(TUTORIALS)
    TUTORIALS = ["introduction"]
end

rm(joinpath(@__DIR__, "src", "tutorials"), force = true, recursive = true)
function preprocess_markdown(input)
    line1, rest = split(input, '\n', limit = 2)
    string(
        line1,
        "\n# *This tutorial is available as a [Jupyter notebook](@__NAME__.ipynb).*\n",
        rest,
    )
end
for tutorial in TUTORIALS
    Literate.markdown(
        joinpath(@__DIR__, "tutorials", tutorial * ".jl"),
        joinpath(@__DIR__, "src", "tutorials");
        preprocess = preprocess_markdown,
    )
    Literate.notebook(
        joinpath(@__DIR__, "tutorials", tutorial * ".jl"),
        joinpath(@__DIR__, "src", "tutorials");
        execute = false,
    )
end

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"))

withenv("GKSwstype" => "100") do
    makedocs(
        bib,
        sitename = "ClimaCore.jl",
        strict = false,
        format = format,
        checkdocs = :exports,
        clean = true,
        doctest = true,
        modules = [ClimaCore, ClimaCoreVTK, ClimaCorePlots, ClimaCoreMakie],
        pages = Any[
            "Home" => "index.md",
            "API" => "api.md",
            "Operators" => "operators.md",
            "Tutorials" => [
                joinpath("tutorials", tutorial * ".md") for
                tutorial in TUTORIALS
            ],
            "Libraries" => [joinpath("lib", "ClimaCoreVTK.md")],
            "references.md",
        ],
    )
end

deploydocs(
    repo = "github.com/CliMA/ClimaCore.jl.git",
    target = "build",
    push_preview = true,
    devbranch = "main",
    forcepush = true,
)
