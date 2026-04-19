import numpy as np

# mipmap generation
def generateMipmaps(texture, method):
    h,w = texture.shape[0], texture.shape[1]
    nb_levels = int(np.floor(np.log2(max(h,w)))) + 1
    mipmaps = [texture.astype(np.float32)]
    for level in range (1, nb_levels) :
        prev = mipmaps[level - 1]
        prev_h, prev_w = prev.shape[0], prev.shape[1]
        next_h, next_w = max(1, prev_h//2), max(1, prev_w//2)

        next = np.zeros((next_h, next_w, 3), dtype=np.float32)

        if method == 'nearest' :
            for i in range(next_h) :
                for j in range(next_w) :
                    next[i][j] = prev[i*2][j*2]

        elif method == 'box' :
            for i in range(next_h) :
                for j in range(next_w) :
                    next[i][j] = np.mean(prev[2*i:2*i+2, 2*j:2*j+2], axis=(0,1))

        elif method == 'gaussian' :
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
            padded_prev = np.pad(prev, ((1,1),(1,1),(0,0)), mode='reflect')
            for i in range(next_h) :
                for j in range(next_w) :
                    region = padded_prev[2*i : 2*i+3, 2*j : 2*j+3]
                    next[i][j] = np.sum(region * kernel[:,:,None], axis=(0,1))

        else:
            raise ValueError("This method for mipmap generation doesn't exist")

        mipmaps.append(next)
    return mipmaps


# sample one level
def sample(texture, u, v) : 

    u = int(u * texture.shape[0])
    v = int((1-v) * texture.shape[1])
    return texture[v,u] / 255.0

def sampleBilinear(texture, u, v) : 

    x = u * (texture.shape[0]-1)
    y = (1-v) * (texture.shape[1]-1)

    x0 = int(x)
    x1 = x0 + 1
    x1 = min(x1, texture.shape[0]-1)
    y0 = int(y)
    y1 = y0 + 1
    y1 = min(y1, texture.shape[1]-1)
   
    tx = x - x0
    ty = y - y0

    c00 = texture[x0,y0] / 255.0
    c10 = texture[x1,y0] / 255.0
    c01 = texture[x0,y1] / 255.0
    c11 = texture[x1,y1] / 255.0

    c0 = (1-tx)*c00 + tx*c10
    c1 = (1-tx)*c01 + tx*c11

    c = (1-ty)*c0 + ty*c1

    return c

# sample with mipmaps
def sampleMipmaps(mipmaps, u, v, lod, mipmap_lookup_method, texel_method, fragment_properties=None):

    if mipmap_lookup_method == 'nearest' :
        # Clamp lod to valid range to handle infinity/NaN
        lod = np.clip(lod, 0, len(mipmaps)-1)
        lod = int(round(lod))

        if texel_method == 'nearest' :
            return sample(mipmaps[lod], u, v)
            
        elif texel_method == 'bilinear' :
            return sampleBilinear(mipmaps[lod], u, v)
        
        else:
            raise ValueError("Only nearest and bilinear methods for texel sampling are supported")

    elif mipmap_lookup_method == 'trilinear' :
        lod0 = int(np.floor(lod))
        lod1 = int(np.ceil(lod))
        lod0 = min(lod0, len(mipmaps)-1)
        lod1 = min(lod1, len(mipmaps)-1)

        c0 = sampleBilinear(mipmaps[lod0], u, v)
        c1 = sampleBilinear(mipmaps[lod1], u, v)

        t = lod - lod0
        return (1-t)*c0 + t*c1
    elif mipmap_lookup_method == 'anisotropic' :
        tex_h, tex_w, _ = mipmaps[0].shape

        du_dx = fragment_properties['du_dx'] * tex_w
        dv_dx = fragment_properties['dv_dx'] * tex_h
        du_dy = fragment_properties['du_dy'] * tex_w
        dv_dy = fragment_properties['dv_dy'] * tex_h

        #dervivative vectors
        dx = np.array([du_dx, dv_dx])
        dy = np.array([du_dy, dv_dy])

        len_x = np.linalg.norm(dx)
        len_y = np.linalg.norm(dy)

        len_x = max(len_x, 1e-8) # euclidean norms of derivative vectors
        len_y = max(len_y, 1e-8)

        max_aniso = 8.0  #how many samples we want to process along the major axis at most (could be another nt, like 16)
        if len_x > len_y:
            major = dx
            minor_len = len_y
            ratio = len_x / len_y
        else:
            major = dy
            minor_len = len_x
            ratio = len_y / len_x

        samples = int(min(max_aniso, max(1.0, ratio)))

        major_dir = major / np.linalg.norm(major) #normalizing the vector

        lod = np.log2(max(minor_len, 1e-8))  # computing lod based on minor axis
        lod = max(lod, 0.0)

        # accumulating the final colour
        color = np.zeros(3)

        for i in range(samples):
            # distribute samples along major axis
            t = (i + 0.5) / samples - 0.5
            u_offset = u + t * major_dir[0] / tex_w
            v_offset = v + t * major_dir[1] / tex_h

            c = sampleMipmaps(
                mipmaps,
                u_offset,
                v_offset,
                lod,
                'trilinear',
                'bilinear'
            )
            color += c

        return color / samples
    
    else:
        raise ValueError("This method for mipmap sampling doesn't exist")


class Fragment:
    def __init__(self, x : int, y : int, depth : float, interpolated_data, properties):
        self.x = x
        self.y = y
        self.depth = depth
        self.interpolated_data = interpolated_data
        self.properties = properties
        self.output = []

def edgeSide(p, v0, v1) : 
    return (p[0]-v0[0])*(v1[1]-v0[1]) - (p[1]-v0[1])*(v1[0]-v0[0])

class GraphicPipeline:
    def __init__ (self, width, height):
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3))
        self.depthBuffer = np.ones((height, width))


    def VertexShader(self, vertex, data) :
        outputVertex = np.zeros((14))

        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        w = 1.0

        vec = np.array([[x],[y],[z],[w]])

        vec = np.matmul(data['projMatrix'],np.matmul(data['viewMatrix'],vec))

        outputVertex[0] = vec[0]/vec[3]
        outputVertex[1] = vec[1]/vec[3]
        outputVertex[2] = vec[2]/vec[3]

        outputVertex[3] = vertex[3]
        outputVertex[4] = vertex[4]
        outputVertex[5] = vertex[5]

        outputVertex[6] = data['cameraPosition'][0] - vertex[0]
        outputVertex[7] = data['cameraPosition'][1] - vertex[1]
        outputVertex[8] = data['cameraPosition'][2] - vertex[2]

        outputVertex[9] = data['lightPosition'][0] - vertex[0]
        outputVertex[10] = data['lightPosition'][1] - vertex[1]
        outputVertex[11] = data['lightPosition'][2] - vertex[2]

        outputVertex[12] = vertex[6]
        outputVertex[13] = vertex[7]

        return outputVertex


    def Rasterizer(self, v0, v1, v2) :
        fragments = []

        #culling back face
        area = edgeSide(v0,v1,v2)
        if area < 0 :
            return fragments
        
        
        #AABBox computation
        #compute vertex coordinates in screen space
        v0_image = np.array([0,0])
        v0_image[0] = (v0[0]+1.0)/2.0 * self.width 
        v0_image[1] = ((v0[1]+1.0)/2.0) * self.height 

        v1_image = np.array([0,0])
        v1_image[0] = (v1[0]+1.0)/2.0 * self.width 
        v1_image[1] = ((v1[1]+1.0)/2.0) * self.height 

        v2_image = np.array([0,0])
        v2_image[0] = (v2[0]+1.0)/2.0 * self.width 
        v2_image[1] = (v2[1]+1.0)/2.0 * self.height 


        # screen coordinates of the triangle
        x0, y0 = v0_image[0], v0_image[1]
        x1, y1 = v1_image[0], v1_image[1]
        x2, y2 = v2_image[0], v2_image[1]

        # texture coordinates of the triangle
        u0_tex, v0_tex = v0[12:14]
        u1_tex, v1_tex = v1[12:14]
        u2_tex, v2_tex = v2[12:14]

        # derrivatives computation
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

        # skip degenerate triangles (zero area in screen space)
        if abs(det) < 1e-8:
            return fragments

        du_dx = ((u1_tex - u0_tex) * (y2 - y0) - (u2_tex - u0_tex) * (y1 - y0)) / det
        du_dy = (-(u1_tex - u0_tex) * (x2 - x0) + (u2_tex - u0_tex) * (x1 - x0)) / det

        dv_dx = ((v1_tex - v0_tex) * (y2 - y0) - (v2_tex - v0_tex) * (y1 - y0)) / det
        dv_dy = (-(v1_tex - v0_tex) * (x2 - x0) + (v2_tex - v0_tex) * (x1 - x0)) / det

        fragment_properties = dict([
            ('du_dx', du_dx),
            ('du_dy', du_dy),
            ('dv_dx', dv_dx),
            ('dv_dy', dv_dy)
        ])


        #compute the two point forming the AABBox
        A = np.min(np.array([v0_image,v1_image,v2_image]), axis = 0)
        B = np.max(np.array([v0_image,v1_image,v2_image]), axis = 0)

        #cliping the bounding box with the borders of the image
        max_image = np.array([self.width-1,self.height-1])
        min_image = np.array([0.0,0.0])

        A  = np.max(np.array([A,min_image]),axis = 0)
        B  = np.min(np.array([B,max_image]),axis = 0)
        
        #cast bounding box to int
        A = A.astype(int)
        B = B.astype(int)
        #Compensate rounding of int cast
        B = B + 1

        #for each pixel in the bounding box
        for j in range(A[1], B[1]) : 
           for i in range(A[0], B[0]) :
                x = (i+0.5)/self.width * 2.0 - 1 
                y = (j+0.5)/self.height * 2.0 - 1

                p = np.array([x,y])
                
                area0 = edgeSide(p,v0,v1)
                area1 = edgeSide(p,v1,v2)
                area2 = edgeSide(p,v2,v0)

                #test if p is inside the triangle
                if (area0 >= 0 and area1 >= 0 and area2 >= 0) : 
                    
                    #Computing 2d barricentric coordinates
                    lambda0 = area1/area
                    lambda1 = area2/area
                    lambda2 = area0/area
                    
                    #one_over_z = lambda0 * 1/v0[2] + lambda1 * 1/v1[2] + lambda2 * 1/v2[2]
                    #z = 1/one_over_z
                    
                    z = lambda0 * v0[2] + lambda1 * v1[2] + lambda2 * v2[2]

                    p = np.array([x,y,z])
                    
                    
                    l = v0.shape[0]
                    #interpolating
                    interpolated_data = v0[3:l] * lambda0 + v1[3:l] * lambda1 + v2[3:l] * lambda2
                    
                    #Emiting Fragment
                    fragments.append(Fragment(i,j,z, interpolated_data, fragment_properties))

        return fragments
    
    def fragmentShader(self,fragment,data):
        N = fragment.interpolated_data[0:3]
        N = N/np.linalg.norm(N)
        V = fragment.interpolated_data[3:6]
        V = V/np.linalg.norm(V)
        L = fragment.interpolated_data[6:9]
        L = L/np.linalg.norm(L)

        R = 2 * np.dot(L,N) * N  -L

        

        ambient = 1.0
        diffuse = max(np.dot(N,L),0)
        specular = np.power(max(np.dot(R,V),0.0),64) 

        ka = 0.1
        kd = 0.9
        ks = 0.3
        phong = ka * ambient + kd * diffuse + ks * specular
        phong = np.ceil(phong*4 +1 )/6.0


        # find LOD
        tex_h, tex_w, _ = data['texture'].shape

        rho_x = np.sqrt((fragment.properties['du_dx'] * tex_w)**2 + (fragment.properties['dv_dx'] * tex_h)**2)
        rho_y = np.sqrt((fragment.properties['du_dy'] * tex_w)**2 + (fragment.properties['dv_dy'] * tex_h)**2)

        rho = max(rho_x, rho_y)
        lod = np.log2(max(rho, 1e-8))
        lod = max(lod, 0.0)

        texture = sampleMipmaps(
            data['mipmaps'],
            fragment.interpolated_data[9], # u
            fragment.interpolated_data[10], # v
            lod,
            data['mipmapLookupMethod'],
            data['texelMethod'],
            fragment.properties #anisostropic filters needs partial derivatives for the fragment
        )

        # old
        # texture = sample(data['texture'], fragment.interpolated_data[9], fragment.interpolated_data[10])

        color = np.array([phong,phong,phong]) * texture

        fragment.output = color

    def draw(self, vertices, triangles, data):
        #Calling vertex shader
        self.newVertices = np.zeros((vertices.shape[0], 14))

        for i in range(vertices.shape[0]) :
            self.newVertices[i] = self.VertexShader(vertices[i],data)
        
        fragments = []
        #Calling Rasterizer
        for i in triangles :
            fragments.extend(self.Rasterizer(self.newVertices[i[0]], self.newVertices[i[1]], self.newVertices[i[2]]))
        
        for f in fragments:
            self.fragmentShader(f,data)
            #depth test
            if self.depthBuffer[f.y][f.x] > f.depth : 
                self.depthBuffer[f.y][f.x] = f.depth
                
                self.image[f.y][f.x] = f.output
                