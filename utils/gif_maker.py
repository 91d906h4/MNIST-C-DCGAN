import os
import imageio

# Set directory of images.
dir = "../result/2024-04-17-3/"
output = "./output.gif"

# Get images and sort with filename.
images = os.listdir(dir)
images = sorted(images, key=lambda x: int(x.split(".")[0]))

# Read and append images into buffer.
buffer = []
for image in images:
    buffer.append(imageio.imread(dir + image))

# Output gif file.
imageio.mimsave(
    uri=output,
    ims=buffer,
    format="gif",
    duration=500,
    loop=0,
)