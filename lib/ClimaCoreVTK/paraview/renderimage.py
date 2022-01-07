#> pvpython renderimage.py input.vtk output.png

import sys, os

print(sys.argv)
if len(sys.argv) != 3:
    raise Exception('must provide a valid input vtk path and output render png path')

from paraview.simple import *

#open data file
data = OpenDataFile(sys.argv[1])

# get active view
view = GetActiveViewOrCreate('RenderView')
Show(data, view)

# reset view to fit data
view.ResetCamera()
view.OrientationAxesVisibility = 0

#position camera
# XZ view by default
camera = GetActiveCamera()
camera.Pitch(90)
view.CameraViewUp = [0, 0, 1]

# This is computed by reseting the camera to the data
# bounds = data.GetDataInformation().DataInformation.GetBounds()
# centerXPos = 0.5 * sum(bounds[0:1])
# centerYPos = 0.5 * sum(bounds[2:3])
# centerZPos = 0.5 * sum(bounds[4:5])]
# view.CameraFocalPoint = [centerXPos, centerYPos, centerZPos]
# view.CameraPosition = [centerXPos, centerYPos, 0]

# show the grid
axesGrid = view.AxesGrid
axesGrid.Visibility = 1

#draw the object
Show()

#set the background color
view.Background = [0,0,0]  #black
#set image size
view.ViewSize = [600, 600] #[width, height]

#set representation to wireframe
dp = GetDisplayProperties()
dp.Representation = "Wireframe"

Render()

#save screenshot
WriteImage(sys.argv[2])