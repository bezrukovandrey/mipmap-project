import numpy as np
from exportToPly import write_ply_file
vertices = np.array([
[0.0,0.0,0.0], #0
[1.0,0.0,0.0], #1
[0.0,1.0,0.0], #2
[1.0,1.0,0.0], #3
[0.0,0.0,1.0], #4
[1.0,0.0,1.0], #5
[0.0,1.0,1.0], #6
[1.0,1.0,1.0], #7
])



triangles = np.array([
[0,4,1],
[1,4,5],
[1,5,3],
[3,5,7],
[4,6,5],
[5,6,7],
[0,1,2],
[2,1,3],
[0,2,4],
[4,2,6],
[2,3,6],
[6,3,7]

], dtype=int)

write_ply_file(vertices,triangles, 'cube.ply' )

