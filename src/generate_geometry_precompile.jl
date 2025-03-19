#=
julia --project=.buildkite
using Revise; @time include("src/generate_geometry_precompile.jl")

This script takes about 32 minutes to run:
 - unary operations: ~1 min
 - binary operations: ~31 min

This script generates precompile statements for the ClimaCore
Geometry module by iterating over combinations of types
and calling a wide range of methods. If successful, we push
precompile statements to a list.


Last output:
```
julia> using Revise; @time include("src/generate_geometry_precompile.jl")
[ Info: Generating ClimaCore Geometry precompile list
┌ Warning: empty Cartesian3Vector list
└ @ Main ~/Dropbox/Caltech/work/dev/CliMA/ClimaCore.jl/src/generate_geometry_precompile.jl:195
┌ Warning: empty norm_sqr list
└ @ Main ~/Dropbox/Caltech/work/dev/CliMA/ClimaCore.jl/src/generate_geometry_precompile.jl:195
[ Info: Generated 2647 AxisTensor unary operator precompile statements
┌ Warning: empty / list
└ @ Main ~/Dropbox/Caltech/work/dev/CliMA/ClimaCore.jl/src/generate_geometry_precompile.jl:254
┌ Warning: empty cross list
└ @ Main ~/Dropbox/Caltech/work/dev/CliMA/ClimaCore.jl/src/generate_geometry_precompile.jl:254
[ Info: Generated 414 AxisTensor binary operator precompile statements
[ Info: Generated 3061 AxisTensor precompile statements
1888.495652 seconds (1.17 G allocations: 79.010 GiB, 23.42% gc time, 67.43% compilation time: <1% of which was recompilation)
```
=#

#! format: off
using LinearAlgebra
using LinearAlgebra: det, dot
using StaticArrays: SMatrix, SVector, SArray

using ClimaCore.Geometry
using ClimaCore.Geometry: ContravariantAxis,
	Contravariant3Vector,
	Contravariant123Vector,
	Contravariant13Vector,
	Contravariant12Vector,
	CovariantAxis,
	CartesianAxis,
	Covariant3Vector,
	Covariant123Vector,
	Covariant12Vector,
	Covariant1Vector,
	Covariant13Vector,
	Axis2Tensor,
	AxisTensor,
	WVector,
	UVVector,
	UVector,
	LocalAxis,
	LocalGeometry,
	XZPoint,
	XYZPoint,
	LatLongZPoint,
	XYPoint,
	ZPoint,
	LatLongPoint,
	XPoint,
	YPoint,
	LatPoint,
	LongPoint,
	contravariant1,
	contravariant2,
	contravariant3,
	Jcontravariant3

function try_to_compile(f, args, list)
	try
		f(args...)
		s = "precompile($(f), $(typeof.(args)))"
		s = replace(s, "Float64" => "FT")
		if !(s in list)
			push!(list, s)
		end
	catch
	end
end

using Combinatorics: permutations, combinations
combos() = collect(Iterators.flatten(collect.(permutations.(collect(combinations(1:4))[1:end-1]))))

# Quadratic q.r.t. Ilist():
# Ilist() = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)] # too long
Ilist() = [(3,), (1, 2), (1, 2, 3)]

prand(::Type{T}) where {FT, T <: XZPoint{FT}} = T(rand(FT),rand(FT))
prand(::Type{T}) where {FT, T <: XYZPoint{FT}} = T(rand(FT),rand(FT),rand(FT))
prand(::Type{T}) where {FT, T <: LatLongZPoint{FT}} = T(rand(FT),rand(FT),rand(FT))
prand(::Type{T}) where {FT, T <: XYPoint{FT}} = T(rand(FT),rand(FT))
prand(::Type{T}) where {FT, T <: ZPoint{FT}} = T(rand(FT))
prand(::Type{T}) where {FT, T <: LatLongPoint{FT}} = T(rand(FT),rand(FT))
prand(::Type{T}) where {FT, T <: XPoint{FT}} = T(rand(FT))

get_∂x∂ξ(::Type{FT}, I, ::Type{S}) where {FT, S} = rand(Axis2Tensor{FT, Tuple{LocalAxis{I}, CovariantAxis{I}}, S})

get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{2, 2, FT, 4}, C <: XZPoint{FT}       , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(prand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{3, 3, FT, 9}, C <: XYZPoint{FT}      , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(prand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{3, 3, FT, 9}, C <: LatLongZPoint{FT} , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(prand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{2, 2, FT, 4}, C <: XYPoint{FT}       , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(prand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{1, 1, FT, 1}, C <: ZPoint{FT}        , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(prand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{2, 2, FT, 4}, C <: LatLongPoint{FT}  , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(prand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))
get_lg_instance(::Type{T}) where {FT, I, S <: SMatrix{1, 1, FT, 1}, C <: XPoint{FT}        , T <: LocalGeometry{I, C, FT, S}} = LocalGeometry(prand(C), rand(FT), rand(FT), get_∂x∂ξ(FT, I, S))

@info "Generating ClimaCore Geometry precompile list"

FT = Float64
list = String[]
ulists = Dict()
ulists["Jcontravariant3"] = String[]
ulists["contravariant1"] = String[]
ulists["contravariant2"] = String[]
ulists["contravariant3"] = String[]
ulists["norm_sqr"] = String[]
ulists["CovariantVector"] = String[]
ulists["ContravariantVector"] = String[]
ulists["LocalVector"] = String[]
ulists["Contravariant123Vector"] = String[]
ulists["WVector"] = String[]
ulists["Cartesian3Vector"] = String[]
ulists["Contravariant3Vector"] = String[]
ulists["Covariant3Vector"] = String[]
# unary operators
try_to_compile(Geometry.Covariant123Vector, (FT(0),FT(0),FT(0)), list)
try_to_compile(Geometry.Contravariant123Vector, (FT(0),FT(0),FT(0)), list)
for I in Ilist()
    N = length(I)
    cova = Geometry.CovariantAxis{I}
    cona = Geometry.ContravariantAxis{I}
    carta = Geometry.CartesianAxis{I}
    loca = Geometry.LocalAxis{I}
    axs = (cova(),cona(),carta(),loca())
    for (SLG, C) in (
		(SMatrix{2, 2, FT, 4}, XZPoint{FT}),
		(SMatrix{3, 3, FT, 9}, XYZPoint{FT}),
		(SMatrix{3, 3, FT, 9}, LatLongZPoint{FT}),
		(SMatrix{2, 2, FT, 4}, XYPoint{FT}),
		(SMatrix{1, 1, FT, 1}, ZPoint{FT}),
		(SMatrix{2, 2, FT, 4}, LatLongPoint{FT}),
		(SMatrix{1, 1, FT, 1}, XPoint{FT}),
		)
        for c in combos()
        	ax = Tuple(map(i->axs[i],c))
        	nd = length(ax)
        	S = SArray{Tuple{(nd .* ones(Int, nd))...}, FT, nd}
    		at = AxisTensor{FT, length(ax), typeof(ax), S}
        	for ILG in Ilist()
		        LGT = LocalGeometry{ILG, C, FT, SLG}
		        lg = get_lg_instance(LGT)
		        try_to_compile(Geometry.Jcontravariant3, (zero(at),lg), ulists["Jcontravariant3"])
		        try_to_compile(Geometry.contravariant1, (zero(at),lg), ulists["contravariant1"])
		        try_to_compile(Geometry.contravariant2, (zero(at),lg), ulists["contravariant2"])
		        try_to_compile(Geometry.contravariant3, (zero(at),lg), ulists["contravariant3"])
		        try_to_compile(LinearAlgebra.norm_sqr, (zero(at),lg), ulists["norm_sqr"])
		        try_to_compile(Geometry.CovariantVector, (zero(at), lg), ulists["CovariantVector"])
		        try_to_compile(Geometry.ContravariantVector, (zero(at), lg), ulists["ContravariantVector"])
		        try_to_compile(Geometry.LocalVector, (zero(at), lg), ulists["LocalVector"])
		        try_to_compile(Geometry.Contravariant123Vector, (zero(at), lg), ulists["Contravariant123Vector"])
		        try_to_compile(Geometry.WVector, (zero(at), lg), ulists["WVector"])
		        try_to_compile(Geometry.Cartesian3Vector, (zero(at), lg), ulists["Cartesian3Vector"])
		        try_to_compile(Geometry.Contravariant3Vector, (zero(at), lg), ulists["Contravariant3Vector"])
		        try_to_compile(Geometry.Covariant3Vector, (zero(at), lg), ulists["Covariant3Vector"])
		    end
    	end
    end
end

for k in keys(ulists)
	isempty(ulists[k]) && @warn "empty $k list"
end
ulist = collect(Iterators.flatten(values(ulists)))
@info "Generated $(length(ulist)) AxisTensor unary operator precompile statements"
@assert ulist isa Vector{String}
@assert length(ulist) ≥ 2647

# binary operators
blists = Dict()
blists["conversion"] = String[]
blists["dot"] = String[]
blists["cross"] = String[]
blists["+"] = String[]
blists["-"] = String[]
blists["*"] = String[]
blists["/"] = String[]
for I in Ilist()
    N = length(I)
    cova = Geometry.CovariantAxis{I}
    cona = Geometry.ContravariantAxis{I}
    carta = Geometry.CartesianAxis{I}
    loca = Geometry.LocalAxis{I}
    axs = (cova(),cona(),carta(),loca())
    for (S, C) in (
		# (SMatrix{2, 2, FT, 4}, XZPoint{FT}),
		(SMatrix{3, 3, FT, 9}, XYZPoint{FT}),
		(SMatrix{3, 3, FT, 9}, LatLongZPoint{FT}),
		# (SMatrix{2, 2, FT, 4}, XYPoint{FT}),
		(SMatrix{1, 1, FT, 1}, ZPoint{FT}),
		(SMatrix{2, 2, FT, 4}, LatLongPoint{FT}),
		# (SMatrix{1, 1, FT, 1}, XPoint{FT}),
		)
    	for c1 in combos()
        	a1 = Tuple(map(i->axs[i],c1))
	        for c2 in combos()
	        	a2 = Tuple(map(i->axs[i],c2))

	        	nd1 = length(a1)
	        	nd2 = length(a2)
	        	S1 = SArray{Tuple{(nd1 .* ones(Int, nd1))...}, FT, nd1}
	        	S2 = SArray{Tuple{(nd2 .* ones(Int, nd2))...}, FT, nd2}

        		x = AxisTensor{FT, length(a1), typeof(a1), S1}
        		y = AxisTensor{FT, length(a2), typeof(a2), S2}

		        LGT = LocalGeometry{I, C, FT, S} # assumes I is diagonal
		        lg = get_lg_instance(LGT)
		        # type conversion
		        try_to_compile(x,  (zero(y), lg), blists["conversion"])
		        try_to_compile(dot, (zero(x), zero(y)), blists["dot"])
		        try_to_compile(LinearAlgebra.cross, (zero(x), zero(y),lg), blists["cross"])
		        try_to_compile(+, (zero(x), zero(y)), blists["+"])
		        try_to_compile(-, (zero(x), zero(y)), blists["-"])
		        try_to_compile(*, (zero(x), zero(y)), blists["*"])
		        try_to_compile(/, (zero(x), zero(y)), blists["/"])
			end
    	end
    end
end
for k in keys(blists)
	isempty(blists[k]) && @warn "empty $k list"
end
blist = collect(Iterators.flatten(values(blists)))
@info "Generated $(length(blist)) AxisTensor binary operator precompile statements"
@assert blist isa Vector{String}
@assert length(blist) ≥ 414

list = vcat(ulist, blist)

@assert length(list) ≥ 3061

@info "Generated $(length(list)) AxisTensor precompile statements"
# @assert length(list) ≥ 1034
open("src/geometry_precompile_list.jl", "w") do io
	println(io, "#=")
	println(io, "This file was automatically generated by generate_geometry_precompile.jl")
	println(io, "Editing it directly is not advised")
	println(io, "=#")
	println(io, "#! format: off")
	for e in list
		println(io, e)
	end
	println(io, "#! format: on")
end
#! format: on
