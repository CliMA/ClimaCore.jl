# TODO: improve this filename
include("utils_cuda.jl")

@testset "Finite difference GradientF2C CUDA" begin
    device = ClimaComms.device()
    gpu_context = ClimaComms.SingletonCommsContext(device)
    println("running test on $device device")

    # Define hv GPU space
    hv_center_space_gpu, hv_face_space_gpu = hvspace_3D_sphere(gpu_context)

    coords = Fields.coordinate_field(hv_face_space_gpu)
    z = coords.z

    gradc = Operators.GradientF2C()

    @test parent(Geometry.WVector.(gradc.(z))) ≈
          parent(Geometry.WVector.(ones(hv_center_space_gpu)))


    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.WeakCurl()


    ccoords = Fields.coordinate_field(hv_center_space_gpu)

    cuₕ = Geometry.Covariant12Vector.(.-ccoords.lat, ccoords.long)
    duₕ = @. hwgrad(hdiv(cuₕ)) - Geometry.Covariant12Vector(
        hwcurl(Geometry.Covariant3Vector(hcurl(cuₕ))),
    )

    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )

    cρ = ones(hv_center_space_gpu)

    vdivf2c.(Ic2f.(cρ .* cuₕ))
end
