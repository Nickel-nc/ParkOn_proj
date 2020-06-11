import glob
import cv2

IMGS_SIZE = (1920, 1080)
target = './img_vid/frame*.jpg'
output = './output/NN_out_test_sample.mp4'

img_array = []
for filename in glob.glob(target):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(output, -1, 15, IMGS_SIZE)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
