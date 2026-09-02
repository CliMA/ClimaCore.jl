import Documenter, DocumenterCitations, DocumenterInterLinks, Literate
import DocInventories
import ClimaCore, ClimaCoreTempestRemap, ClimaCoreSpectra
using CairoMakie  # loads ClimaCoreMakieExt so Visualize is documented

if !@isdefined(TUTORIALS)
    TUTORIALS = ["introduction", "cg_dg_switch"]
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

# Every docstring `@ref` should point at a symbol some page renders; otherwise
# the broken link stays latent until that docstring is added to a page and
# fails an unrelated pull request. The existing backlog (several hundred
# references, mostly to internals) is reported, not fatal, until the reference
# pages cover it; STRICT_DOCSTRING_REFS=1 lists the offenders and fails. See
# check_docstring_refs.jl.
include(joinpath(@__DIR__, "check_docstring_refs.jl"))
check_docstring_refs(
    ClimaCore,
    joinpath(@__DIR__, "src");
    strict = !isempty(get(ENV, "STRICT_DOCSTRING_REFS", "")),
)

# The default inventory-download timeout (1 s) fails on slow networks.
inventory(url) = DocInventories.Inventory(url; timeout = 30.0, retries = 5)
links = DocumenterInterLinks.InterLinks(
    "Julia" => inventory("https://docs.julialang.org/en/v1/objects.inv"),
    "ClimaComms" =>
        inventory("https://clima.github.io/ClimaComms.jl/stable/objects.inv"),
    "ClimaTimeSteppers" => inventory(
        "https://clima.github.io/ClimaTimeSteppers.jl/stable/objects.inv",
    ),
    "ClimaAtmos" =>
        inventory("https://clima.github.io/ClimaAtmos.jl/stable/objects.inv"),
)

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

    # External links are checked on every build but only fail it when
    # LINKCHECK_STRICT is set: external sites are transiently unreachable, and
    # that should not block unrelated pull requests.
    warnonly = Symbol[:cross_references]
    isempty(get(ENV, "LINKCHECK_STRICT", "")) && push!(warnonly, :linkcheck)

    Documenter.makedocs(;
        plugins = [bib, links],
        sitename = "ClimaCore.jl",
        format = format,
        checkdocs = :exports,
        linkcheck = true,
        warnonly = warnonly,
        clean = true,
        doctest = true,
        modules = [
            ClimaCore,
            ClimaCore.Remapping,
            ClimaCoreSpectra,
            Base.get_extension(ClimaCore, :ClimaCoreMakieExt),
            ClimaCoreTempestRemap,
        ],
        pages = Any[
            "Home" => "index.md",
            "Getting started" => [
                "Install ClimaCore" => "howto/install.md",
                "Concepts and design" => "getting_started/concepts.md",
                "Tutorial: Introduction" => "tutorials/introduction.md",
                "Tutorial: CG and DG with one tendency" => "tutorials/cg_dg_switch.md",
            ],
            "How-to guides" => [
                "Run the examples" => "howto/run_examples.md",
                "Remap and interpolate" => "howto/remapping.md",
                "Mask horizontal points" => "howto/masks.md",
                "Debug NaNs and broadcasts" => "howto/debugging.md",
                "Move data between CPU and GPU" => "howto/to_device.md",
            ],
            "Explanation" => [
                "Mathematical framework" => "explanation/math_framework.md",
                "Spectral elements: CG and DG" => "explanation/discretizations.md",
                "Example gallery" => "explanation/examples.md",
            ],
            "Reference" => [
                "API overview" => "reference/index.md",
                "Domains" => "reference/domains.md",
                "Meshes" => "reference/meshes.md",
                "Topologies" => "reference/topologies.md",
                "Geometry" => "reference/geometry.md",
                "Quadratures" => "reference/quadratures.md",
                "Grids" => "reference/grids.md",
                "CommonGrids" => "reference/common_grids.md",
                "Spaces" => "reference/spaces.md",
                "CommonSpaces" => "reference/common_spaces.md",
                "Fields" => "reference/fields.md",
                "DataLayouts" => "reference/datalayouts.md",
                "Operators" => "reference/operators.md",
                "DSS" => "reference/dss.md",
                "Limiters" => "reference/limiters.md",
                "Hypsography" => "reference/hypsography.md",
                "MatrixFields" => "reference/matrix_fields.md",
                "Remapping" => "reference/remapping.md",
                "Visualize" => "reference/visualize.md",
                "Input/Output" => "reference/input_output.md",
                "Devices" => "reference/devices.md",
                "Utilities" => "reference/utilities.md",
                "DebugOnly" => "reference/debug_only.md",
                "Companion packages" => [
                    "ClimaCoreTempestRemap" => "lib/ClimaCoreTempestRemap.md",
                    "ClimaCoreSpectra" => "lib/ClimaCoreSpectra.md",
                ],
            ],
            "Developer" => [
                "Contributing" => "Contributing.md",
                "Code of conduct" => "code_of_conduct.md",
            ],
            "References" => "references.md",
        ],
    )
end

Documenter.deploydocs(
    repo = "github.com/CliMA/ClimaCore.jl.git",
    target = "build",
    push_preview = all(
        !isempty,
        (get(ENV, "GITHUB_TOKEN", ""), get(ENV, "DOCUMENTER_KEY", "")),
    ),
    devbranch = "main",
    forcepush = true,
)
