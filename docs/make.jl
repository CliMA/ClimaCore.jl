import Documenter, DocumenterCitations, Literate
import ClimaCore,
    ClimaCoreVTK,
    ClimaCoreMakie,
    ClimaCorePlots,
    ClimaCoreTempestRemap,
    ClimaCoreSpectra

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
        size_threshold = 300_000, # default is 200_000
        size_threshold_warn = 200_000, # default is 100_000
    )

    Documenter.makedocs(;
        plugins = [bib],
        sitename = "ClimaCore.jl",
        format = format,
        checkdocs = :exports,
        clean = true,
        doctest = true,
        modules = [
            ClimaCore,
            ClimaCoreVTK,
            ClimaCoreSpectra,
            ClimaCorePlots,
            ClimaCoreMakie,
            ClimaCoreTempestRemap,
        ],
        pages = Any[
            "Home" => "index.md",
            "Introduction" => "intro.md",
            "Mathematical Framework" => "math_framework.md",
            "Installation and How-to Guides" => "installation_instructions.md",
            "Geometry" => "geometry.md",
            "Operators" => "operators.md",
            "Remapping" => "remapping.md",
            "MatrixFields" => "matrix_fields.md",
            "API" => "api.md",
            "Developer docs" => [
                "Performance tips" => "performance_tips.md"
                "Shared memory design" => "shmem_design.md"
            ],
            "Tutorials" => [
                joinpath("tutorials", tutorial * ".md") for
                tutorial in TUTORIALS
            ],
            "Examples" => "examples.md",
            "Masks" => "masks.md",
            "Debugging" => "debugging.md",
            "Libraries" => [
                joinpath("lib", "ClimaCorePlots.md"),
                joinpath("lib", "ClimaCoreMakie.md"),
                joinpath("lib", "ClimaCoreVTK.md"),
                joinpath("lib", "ClimaCoreTempestRemap.md"),
                joinpath("lib", "ClimaCoreSpectra.md"),
            ],
            "Contributing guide" => "Contributing.md",
            "Code of Conduct" => "code_of_conduct.md",
            "Frequently asked questions" => "faq.md",
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
