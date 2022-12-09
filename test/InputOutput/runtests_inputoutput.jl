using SafeTestsets

@safetestset "spectralelement2d" begin
    @time include("spectralelement2d.jl")
end
@safetestset "hybrid2dbox" begin
    @time include("hybrid2dbox.jl")
end
@safetestset "hybrid2dbox_topography" begin
    @time include("hybrid2dbox_topography.jl")
end
@safetestset "hybrid2dbox_stretched" begin
    @time include("hybrid2dbox_stretched.jl")
end
@safetestset "hybrid3dbox" begin
    @time include("hybrid3dbox.jl")
end
@safetestset "hybrid3dcubedsphere" begin
    @time include("hybrid3dcubedsphere.jl")
end
@safetestset "hybrid3dcubedsphere_topography" begin
    @time include("hybrid3dcubedsphere_topography.jl")
end
