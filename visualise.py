import skimage
import pyvista as pv

img = skimage.io.imread('dk.tiff')

# Choose which subsection to view
img = img[:200, :200, :200, 0]

# 0 = Air-filled pore space
# 255 = Solid soil matrix
val = 255

points = []
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		for k in range(img.shape[2]):
			if img[i,j,k] == val:
				points.append([i,j,k])

point_cloud = pv.PolyData(points)
point_cloud.plot(eye_dome_lighting=True, point_size=10, render_points_as_spheres=True)