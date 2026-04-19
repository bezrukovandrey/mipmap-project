import numpy as np

width = 1280  
height = 720

import time

start = time.time()


from graphicPipeline import GraphicPipeline
from graphicPipeline import generateMipmaps

pipeline = GraphicPipeline(width,height)


from camera import Camera
from projection import Projection


position = np.array([0.0, 1.2, 2.5])
lookAt = np.array([0.0, 0.0, -1.0])
up = np.array([0.0, 1.0, 0.0])
right = np.array([1.0, 0.0, 0.0])

cam = Camera(position, lookAt, up, right)

nearPlane = 0.1
farPlane = 20.0
fov = 1.2
aspectRatio = width/height

proj = Projection(nearPlane ,farPlane,fov, aspectRatio) 


lightPosition = np.array([0.0, 8.0, 2.0])

from readply import readply

vertices, triangles = readply('flat_quad.ply')


# load and show an image with Pillow
from PIL import Image
from numpy import asarray
# Open the image form working directory
image = asarray(Image.open('checkerboard.png').convert('RGB'))

mipmapLookupMethod = 'trilinear' # nearest / trilinear / anisotropic
texelMethod = 'bilinear' # nearest / bilinear 
mimapGenerationMethod = 'box' # nearest / box / gaussian
mipmaps = generateMipmaps(image, mimapGenerationMethod)

data = dict([
  ('viewMatrix',cam.getMatrix()),
  ('projMatrix',proj.getMatrix()),
  ('cameraPosition',position),
  ('lightPosition',lightPosition),
  ('texture', image),
  ('mipmaps', mipmaps),
  ('mipmapLookupMethod', mipmapLookupMethod),
  ('texelMethod', texelMethod)
])


start = time.time()

pipeline.draw(vertices, triangles, data)

end = time.time()
print(end - start)

import matplotlib.pyplot as plt
imgplot = plt.imshow(pipeline.image)
plt.show()


