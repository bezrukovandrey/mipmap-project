import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from graphicPipeline import generateMipmaps

def saveMipMapStack(mipmaps):
    max_w = max(m.shape[1] for m in mipmaps)

    rows = []
    for mip in mipmaps:
        h, w, _ = mip.shape
        pad = max_w - w

        padded = np.pad( mip,((0,0),(0,pad),(0,0)),mode='constant',constant_values=255)

        rows.append(padded)

    pyramid = np.vstack(rows)

    plt.figure(figsize=(5,10), facecolor='white')
    plt.title("Mipmap Pyramid (Flat Stack)")
    plt.imshow(pyramid.astype(np.uint8))
    plt.axis('off')

    plt.savefig("mipmap_flat_stack.png", dpi=300, facecolor='white')


def mip_scale(i, n):
    t = i / (n - 1 + 1e-8)
    scale = (1.0 - t) ** 3
    return max(scale, 0.02)


def saveMipMap3DPyramid(mipmaps):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    n = len(mipmaps)

    for i, mip in enumerate(mipmaps):
        h, w, _ = mip.shape

        scale = mip_scale(i, n)

        x = np.linspace(-0.5, 0.5, w) * scale
        y = np.linspace(-0.5, 0.5, h) * scale
        X, Y = np.meshgrid(x, y)

        Z = np.full((h, w), i)

        facecolors = mip.astype(np.float32) / 255.0

        ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            shade=False
        )

    ax.set_title("Mipmap Pyramid (3D Visualization)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Level Of Detail (LOD)")

    ax.view_init(elev=20, azim=-35)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    plt.savefig("mipmap_3d_pyramid.png", dpi=300, facecolor='white')
    plt.close(fig)

def main():
    image_path = "brickwall.png"
    img = Image.open(image_path).convert("RGB")
    texture = np.array(img)

    mipmaps = generateMipmaps(texture, method='box')

    saveMipMapStack(mipmaps)
    saveMipMap3DPyramid(mipmaps)


if __name__ == "__main__":
    main()