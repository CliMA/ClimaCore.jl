# Temporary diagnostic: device-side profile of flagship AMIP coupling steps, with the device
# time of column reductions, accumulations, and band matrix solves classified by their full
# kernel names. Run with the same arguments as experiments/AMIP/run_simulation.jl.
import CUDA
include(abspath(joinpath(ENV["COUPLER_PATH"], "experiments", "AMIP", "code_loading.jl")))
config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)
@info "Compiling step!..."
ClimaCoupler.step!(cs)
ClimaCoupler.step!(cs)
n_steps = 20
@info "Profiling $n_steps steps..."
res = CUDA.@profile external = false begin
    for _ in 1:n_steps
        ClimaCoupler.step!(cs)
    end
end
# Kernel names can be megabytes long, so the table is cropped to a fixed width.
show(IOContext(stdout, :limit => true, :displaysize => (100000, 600)), res)
println()

# Kernels are grouped by their full names, which the cropped table cannot distinguish.
# Integral kernels (only on main) are checked first, since they contain reductions.
const CLASSES = (
    ("integral", r"column_integral"),
    ("solve", r"tridiag|single_field_solve|single_column_solve|solve_columns|band_matrix_solve"),
    ("reduce", r"column_reduce|single_column_reduce"),
    ("accumulate", r"column_accumulate|single_column_accumulate"),
)
kernel_class(name) =
    something(findfirst(((_, regex),) -> occursin(regex, name), CLASSES), 0)
short_name(name) =
    length(name) <= 400 ? name :
    string(first(name, 150), " … ", last(name, 250))
try
    device = res.device
    names = collect(String, string.(device.name))
    durations = Float64.(device.stop .- device.start)
    totals = Dict{String, Tuple{Int, Float64}}()
    for (name, duration) in zip(names, durations)
        (count, total) = get(totals, name, (0, 0.0))
        totals[name] = (count + 1, total + duration)
    end
    println(
        "PROFILE_TOTAL steps=", n_steps, " kernels=", length(totals),
        " launches_per_step=", length(names) / n_steps,
        " device_ms_per_step=", round(sum(durations) * 1e3 / n_steps; digits = 3),
    )
    for (class_index, (class, _)) in enumerate(CLASSES)
        members = sort!(
            [(name, value) for (name, value) in totals if kernel_class(name) == class_index];
            by = x -> -x[2][2],
        )
        println(
            "PROFILE_CLASS class=", class, " kernels=", length(members),
            " launches_per_step=", sum(x -> x[2][1], members; init = 0) / n_steps,
            " device_ms_per_step=",
            round(sum(x -> x[2][2], members; init = 0.0) * 1e3 / n_steps; digits = 4),
        )
        for (name, (count, total)) in members
            println(
                "PROFILE_KERNEL class=", class, " hash=", string(hash(name); base = 16),
                " launches_per_step=", count / n_steps,
                " device_ms_per_step=", round(total * 1e3 / n_steps; digits = 4),
                " mean_us=", round(total * 1e6 / count; digits = 2),
                " name=", replace(short_name(name), r"\s+" => " "),
            )
        end
    end
catch error
    println("PROFILE_CLASS error ", first(sprint(showerror, error), 300))
    println("device columns: ", propertynames(res.device))
end
