from __future__ import division
import glob
from PIL import Image

target = './img_vid/frame*.jpg'
output = './output/output.gif'

images = []
basewidth = 640

# Create images list
for filename in glob.glob(target):
    img = Image.open(filename)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)

    images.append(img)


images[0].save(output, format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)
