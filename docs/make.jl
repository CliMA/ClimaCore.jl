import Documenter, DocumenterCitations, Literate
import ClimaCore,
    ClimaCoreVTK, ClimaCoreMakie, ClimaCorePlots, ClimaCoreTempestRemap

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

    mathengine = Documenter.MathJax(
        Dict(
            :TeX => Dict(
                :equationNumbers => Dict(:autoNumber => "AMS"),
                :Macros => Dict(),
            ),
        ),
    )

    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        mathengine = mathengine,
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
        modules = [
            ClimaCore,
            ClimaCoreVTK,
            ClimaCorePlots,
            ClimaCoreMakie,
            ClimaCoreTempestRemap,
        ],
        pages = Any[
            "Home" => "index.md",
            "Introduction" => "intro.md",
            "API" => "api.md",
            "Mathematical Framework" => "math_framework.md",
            "Operators" => "operators.md",
            "Developer docs" => ["Performance tips" => "performance_tips.md"],
            "Tutorials" => [
                joinpath("tutorials", tutorial * ".md") for
                tutorial in TUTORIALS
            ],
            "Examples" => "examples.md",
            "AxisTensor conversions" => "axis_tensor_conversions.md",
            "Libraries" => [
                joinpath("lib", "ClimaCoreVTK.md"),
                joinpath("lib", "ClimaCoreTempestRemap.md"),
            ],
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
