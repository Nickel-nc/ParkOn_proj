import plotly.express as px
import imageio

target = 'image2.jpg'

img = imageio.imread(target)
fig = px.imshow(img)
fig.show()
