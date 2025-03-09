#=
julia --project=.buildkite
using Revise; include("test/Operators/finitedifference/benchmark_fd_ops_shared_memory.jl")
=#
include("utils_fd_ops_shared_memory.jl")
using BenchmarkTools

#! format: off
function bench_kernels!(fields)
    (; f, ρ, ϕ) = fields
    (; ᶜout1, ᶜout2, ᶜout3, ᶜout4, ᶜout5, ᶜout6, ᶜout7, ᶜout8) = fields
    device = ClimaComms.device(f)
    FT = Spaces.undertype(axes(ϕ))
    div_bcs = Operators.DivergenceF2C(;
        bottom = Operators.SetValue(Geometry.Covariant3Vector(FT(10))),
        top = Operators.SetValue(Geometry.Covariant3Vector(FT(10))),
    )
    div = Operators.DivergenceF2C()
    ᶠwinterp = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    println("ᶜout1: ", @benchmark ClimaComms.@cuda_sync $device begin
        @. $ᶜout1 = $div(Geometry.WVector($f))
    end)
    @. ᶜout2 = 0
    println("ᶜout2: ", @benchmark ClimaComms.@cuda_sync $device begin
        @. $ᶜout2 += $div(Geometry.WVector($f) * 2)
    end)
    println("ᶜout3: ", @benchmark ClimaComms.@cuda_sync $device begin
        @. $ᶜout3 = $div(Geometry.WVector($ᶠwinterp($ϕ, $ρ)))
    end)

    println("ᶜout4: ", @benchmark ClimaComms.@cuda_sync $device begin
        @. $ᶜout4 = $div_bcs(Geometry.WVector($f))
    end)
    @. ᶜout5 = 0
    println("ᶜout5: ", @benchmark ClimaComms.@cuda_sync $device begin
        @. $ᶜout5 += $div_bcs(Geometry.WVector($f) * 2)
    end)
    println("ᶜout6: ", @benchmark ClimaComms.@cuda_sync $device begin
        @. $ᶜout6 = $div_bcs(Geometry.WVector($ᶠwinterp($ϕ, $ρ)))
    end)

    # from the wild
    Ic2f = Operators.InterpolateC2F(; top = Operators.Extrapolate())
    divf2c = Operators.DivergenceF2C(; bottom = Operators.SetValue(Geometry.Covariant3Vector(FT(10))))
    # only upward component of divergence
    println("ᶜout7: ", @benchmark ClimaComms.@cuda_sync $device begin
        @. $ᶜout7 = $divf2c(Geometry.WVector($Ic2f($ϕ))) # works
    end)
    println("ᶜout8: ", @benchmark ClimaComms.@cuda_sync $device begin
        @. $ᶜout8 = $divf2c($Ic2f(Geometry.WVector($ϕ)))
    end)
    return nothing
end;

#! format: on

@info "GPU benchmark results (Float64) z_elem = 10, h_elem = 30:"
ᶜspace =
    get_space_extruded(ClimaComms.device(), Float64; z_elem = 10, h_elem = 30);
ᶠspace = Spaces.face_space(ᶜspace);
fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...);
bench_kernels!(fields)
Utilities.Cache.clean_cache!();
GC.gc(true);

@info "GPU benchmark results (Float64) z_elem = 63, h_elem = 30:"
ᶜspace =
    get_space_extruded(ClimaComms.device(), Float64; z_elem = 63, h_elem = 30);
ᶠspace = Spaces.face_space(ᶜspace);
fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...);
bench_kernels!(fields)
Utilities.Cache.clean_cache!();
GC.gc(true);

@info "GPU benchmark results (Float64) z_elem = 10, h_elem = 100:"
ᶜspace =
    get_space_extruded(ClimaComms.device(), Float64; z_elem = 10, h_elem = 100);
ᶠspace = Spaces.face_space(ᶜspace);
fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...);
bench_kernels!(fields)
Utilities.Cache.clean_cache!();
GC.gc(true);

@info "GPU benchmark results (Float32) z_elem = 10, h_elem = 30:"
ᶜspace =
    get_space_extruded(ClimaComms.device(), Float32; z_elem = 10, h_elem = 30);
ᶠspace = Spaces.face_space(ᶜspace);
fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...);
bench_kernels!(fields)
Utilities.Cache.clean_cache!();
GC.gc(true);

@info "GPU benchmark results (Float32) z_elem = 63, h_elem = 30:"
ᶜspace =
    get_space_extruded(ClimaComms.device(), Float32; z_elem = 63, h_elem = 30);
ᶠspace = Spaces.face_space(ᶜspace);
fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...);
bench_kernels!(fields)
Utilities.Cache.clean_cache!();
GC.gc(true);

@info "GPU benchmark results (Float32) z_elem = 10, h_elem = 100:"
ᶜspace =
    get_space_extruded(ClimaComms.device(), Float32; z_elem = 10, h_elem = 100);
ᶠspace = Spaces.face_space(ᶜspace);
fields = (; get_fields(ᶜspace)..., get_fields(ᶠspace)...);
bench_kernels!(fields)
Utilities.Cache.clean_cache!();
GC.gc(true);

nothing
