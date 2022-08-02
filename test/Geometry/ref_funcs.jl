
@inline ref_contravariant1(u::AxisVector, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant1Axis(), u, local_geometry)[1]
@inline ref_contravariant2(u::AxisVector, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant2Axis(), u, local_geometry)[1]
@inline ref_contravariant3(u::AxisVector, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant3Axis(), u, local_geometry)[1]

@inline ref_contravariant1(u::Axis2Tensor, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant1Axis(), u, local_geometry)[1, :]
@inline ref_contravariant2(u::Axis2Tensor, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant2Axis(), u, local_geometry)[1, :]
@inline ref_contravariant3(u::Axis2Tensor, local_geometry::LocalGeometry) =
    @inbounds ref_project(Contravariant3Axis(), u, local_geometry)[1, :]

@inline ref_covariant1(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₁
@inline ref_covariant2(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₂
@inline ref_covariant3(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₃
