import numpy as np
from exportToPly import write_ply_file

pointPerRing = 8
nbRing = 3
vertices = []
for i in range (nbRing) :
    for j in range (pointPerRing) :
        x= np.cos(2 * np.pi * j / pointPerRing)
        y= np.sin(2 * np.pi * j / pointPerRing)
        z= i * 1.0
        vertices.append([x,y,z])
vertices = np.array(vertices)

triangles = []
for i in range (nbRing -1) :
    for j in range (pointPerRing) :
        current = i * pointPerRing + j
        next = current + pointPerRing
        nextPoint = i * pointPerRing + (j + 1) % pointPerRing
        nextPointNext = nextPoint + pointPerRing

        triangles.append([current, next, nextPoint])
        triangles.append([next, nextPointNext, nextPoint])


triangles = np.array(triangles, dtype=int)
write_ply_file(vertices,triangles, 'cylinder.ply' )