import plotly.express as px
import imageio
img = imageio.imread('image.jpg')
fig = px.imshow(img)
fig.show()
