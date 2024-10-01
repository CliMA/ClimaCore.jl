#=
julia --project=.buildkite
ENV["CLIMACOMMS_DEVICE"]="CUDA";
using Revise; include("test/MatrixFields/multiple_field_solve_reproducer_1.jl")

# TODO: simplify this reproducer
=#
using Test
using StaticArrays
import LinearAlgebra: I
import ClimaCore
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore.Utilities: PlusHalf, half
import ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name, TridiagonalMatrixRow, DiagonalMatrixRow
import ClimaCore:
    DataLayouts,
    Fields,
    Domains,
    Topologies,
    Meshes,
    Operators,
    Spaces,
    Geometry,
    Quadratures

function toy_sphere(::Type{FT}) where {FT}
    context = ClimaComms.context()
    helem = 101
    npoly = 1
    hdomain = Domains.SphereDomain(FT(1e7))
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    quad = Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)
    vdomain = ClimaCore.Domains.IntervalDomain(
        Geometry.ZPoint(FT(-50)),
        Geometry.ZPoint(FT(0));
        boundary_names = (:bottom, :top),
    )
    dz_tuple = FT.((10.0, 0.05))
    vmesh = ClimaCore.Meshes.IntervalMesh(
        vdomain,
        ClimaCore.Meshes.GeneralizedExponentialStretching{FT}(
            dz_tuple[1],
            dz_tuple[2],
        );
        nelems = 15,
        reverse_mode = true,
    )
    vtopology = Topologies.IntervalTopology(context, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    return (center_space, face_space, hspace)
end

(cspace, fspace, hspace) = toy_sphere(Float64);

FT = Float64

vector = Fields.FieldVector(;
    soilco2 = (; C = Fields.Field(FT, cspace),),
    soil = (;
        ϑ_l = Fields.Field(FT, cspace),
        θ_i = Fields.Field(FT, cspace),
        ρe_int = Fields.Field(FT, cspace),
    ),
    canopy = (;
        hydraulics = (; ϑ_l = Fields.Field(Tuple{FT}, hspace)),
        energy = (; T = Fields.Field(FT, hspace)),
    ),
);

b_vec = similar(vector);
x = MatrixFields.field_vector_view(vector);
b = MatrixFields.field_vector_view(b_vec);

implicit_vars = (@name(soil.ϑ_l), @name(soil.ρe_int), @name(canopy.energy.T));
explicit_vars =
    (@name(soilco2.C), @name(soil.θ_i), @name(canopy.hydraulics.ϑ_l));
get_jac_type(
    space::Union{
        Spaces.FiniteDifferenceSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    FT,
) = MatrixFields.TridiagonalMatrixRow{FT};
get_jac_type(
    space::Union{Spaces.PointSpace, Spaces.SpectralElementSpace2D},
    FT,
) = MatrixFields.DiagonalMatrixRow{FT};
get_j_field(space, FT) = zeros(get_jac_type(space, FT), space)

implicit_blocks = MatrixFields.unrolled_map(
    var ->
        (var, var) =>
            get_j_field(axes(MatrixFields.get_field(vector, var)), FT),
    implicit_vars,
);
explicit_blocks =
    MatrixFields.unrolled_map(var -> (var, var) => FT(-1) * I, explicit_vars);

A_mf = MatrixFields.FieldMatrix(implicit_blocks..., explicit_blocks...);
A = MatrixFields.replace_name_tree(A_mf, keys(b).name_tree);
names = MatrixFields.matrix_row_keys(keys(A))

alg = MatrixFields.BlockDiagonalSolve();
solver = MatrixFields.FieldMatrixSolver(alg, A_mf, vector);
(; cache) = solver;

Nnames = length(names);
sscache = Operators.strip_space(cache);
ssx = Operators.strip_space(x);
ssA = Operators.strip_space(A);
ssb = Operators.strip_space(b);
caches = map(name -> sscache[name], names);
xs = map(name -> ssx[name], names);
As = map(name -> ssA[name, name], names);
bs = map(name -> ssb[name], names);
x1 = first(xs);
us = DataLayouts.UniversalSize(Fields.field_values(x1));
device = ClimaComms.device(x[first(names)]);
args = (device, caches, xs, As, bs, x1, us, Val(Nnames));

# Incorrectly throws
@test_throws CUDA.InvalidIRError begin
    CCCE = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt) # (reproducer requires CUDA)
    CUDA.@cuda CCCE.multiple_field_solve_kernel!(args...)
end
