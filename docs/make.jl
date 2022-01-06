import Documenter, DocumenterCitations, Literate
import ClimaCore, ClimaCoreVTK, ClimaCoreMakie, ClimaCorePlots

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

withenv("GKSwstype" => "nul") do

    bib =
        DocumenterCitations.CitationBibliography(joinpath(@__DIR__, "refs.bib"))

    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
    )

    Documenter.makedocs(
        bib,
        sitename = "ClimaCore.jl",
        strict = [:example_block],
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

Documenter.deploydocs(
    repo = "github.com/CliMA/ClimaCore.jl.git",
    target = "build",
    push_preview = true,
    devbranch = "main",
    forcepush = true,
)
