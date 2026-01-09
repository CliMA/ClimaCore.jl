"""
This file contains stub functions to be used in conservative remapping.
The actual implementations are in the `ClimaCoreConservativeRegriddingExt`
extension.
"""

function get_element_vertices end

function integrate_each_element end

function get_value_per_element! end

function set_value_per_element! end

# Maps required package to the functions provided by that extension
extension_fns = [
    :ConservativeRegridding => [
        :get_element_vertices,
        :integrate_each_element,
        :get_value_per_element!,
        :set_value_per_element!,
    ],
]

"""
    is_pkg_loaded(pkg::Symbol)

Check if `pkg` is loaded or not.
"""
function is_pkg_loaded(pkg::Symbol)
    return any(k -> Symbol(k.name) == pkg, keys(Base.loaded_modules))
end

function __init__()
    # Register error hint if a package is not loaded
    if isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(
            MethodError,
        ) do io, exc, _argtypes, _kwargs
            for (pkg, fns) in extension_fns
                if Symbol(exc.f) in fns && !is_pkg_loaded(pkg)
                    print(io, "\nImport $pkg to enable `$(exc.f)`.";)
                end
            end
        end
    end
end
