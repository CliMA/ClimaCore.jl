module Visualize

export fieldcontourf,
    fieldcontourf!, fieldheatmap, fieldheatmap!, fieldline, fieldline!

function fieldcontourf end
function fieldcontourf! end
function fieldheatmap end
function fieldheatmap! end
function fieldline end
function fieldline! end

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
        if exc.f in (
            fieldcontourf,
            fieldcontourf!,
            fieldheatmap,
            fieldheatmap!,
            fieldline,
            fieldline!,
        )
            print(
                io,
                "\nImport one of the Makie backends (CairoMakie, GLMakie, WGLMakie, etc.) to enable `$(exc.f)`.",
            )
        end
    end
end

end
