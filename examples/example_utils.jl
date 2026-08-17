# Helpers shared by the example scripts. Each example `include`s this file
# (it defines methods only; see test/README.md on `utils_` files).

"""
    linkfig(figpath, alt = "")

Print the escape sequence that makes Buildkite show the uploaded artifact at
`figpath` inline in the build log. Outside CI this does nothing.
"""
function linkfig(figpath, alt = "")
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end
