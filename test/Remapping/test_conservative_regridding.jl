using ClimaCore:
    CommonSpaces, Remapping, Fields, Spaces, RecursiveApply, Meshes, Quadratures
using ConservativeRegridding



# TODO figure out if 0-area elements is expected with odd h_elem

space1 = CommonSpaces.CubedSphereSpace(;
    radius = 10,
    n_quad_points = 3,
    h_elem = 8,
)
space2 = CommonSpaces.CubedSphereSpace(;
    radius = 10,
    n_quad_points = 4,
    h_elem = 6,
)

vertices1 = Remapping.get_element_vertices(space1)
vertices2 = Remapping.get_element_vertices(space2)

# Pass in destination vertices first, source vertices second
# TODO open issue in CR.jl about ordering of inputs source/dest
regridder_1_to_2 = ConservativeRegridding.Regridder(vertices2, vertices1)
regridder_2_to_1 = ConservativeRegridding.Regridder(vertices1, vertices2)

# Define a field on the first space, to use as our source field
field1 = Fields.coordinate_field(space1).lat
ones_field1 = Fields.ones(space1)

# Check that integrating over each element and summing gives the same result as integrating over the whole domain
@assert isapprox(sum(Remapping.integrate_each_element(field1)), sum(field1), atol = 1e-12)
# Check that integrating 1 over each element and summing gives the same result as integrating 1 over the whole domain
@assert sum(Remapping.integrate_each_element(ones_field1)) ≈ sum(ones_field1)

# Get one value per element in the field, equal to the average of the values at nodes of the element
value_per_element1 = zeros(Float64, Meshes.nelements(space1.grid.topology.mesh))
Remapping.get_value_per_element!(value_per_element1, field1, ones_field1)

# Allocate a vector with length equal to the number of elements in the target space
value_per_element2 = zeros(Float64, Meshes.nelements(space2.grid.topology.mesh))
ConservativeRegridding.regrid!(value_per_element2, regridder_1_to_2, value_per_element1)

# Now that we have our regridded vector, put it onto a field on the second space
field2 = Fields.zeros(space2)
Remapping.set_value_per_element!(field2, value_per_element2)
field1_one_value_per_element = Fields.zeros(space1)
Remapping.set_value_per_element!(field1_one_value_per_element, value_per_element1)

# # Plot the fields
# using ClimaCoreMakie
# using GLMakie
# fig = ClimaCoreMakie.fieldheatmap(field1)
# save("field1.png", fig)
# fig = ClimaCoreMakie.fieldheatmap(field1_one_value_per_element)
# save("field1_one_value_per_element.png", fig)
# fig = ClimaCoreMakie.fieldheatmap(field2)
# save("field2.png", fig)

# Check the conservation error
abs_error = abs(sum(field1) - sum(field2))
@assert abs_error < 1e-12
abs_error_one_value_per_element = abs(sum(field1_one_value_per_element) - sum(field2))
@assert abs_error_one_value_per_element < 2e-12
