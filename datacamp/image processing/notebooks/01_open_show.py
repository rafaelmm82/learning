import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, color


# present image using pyplot/imshow function
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()


coffee_img = data.coffee()
type(coffee_img)
print(coffee_img.shape)

coins_img = data.coins()

# Show image with io from scikit-image
io.imshow(coffee_img)
io.show()

# Show image from adapted pyplot imshow
show_image(coffee_img)
show_image(coins_img)


rocket = data.rocket()
gray_rocket = color.rgb2gray(rocket)

show_image(rocket)
show_image(gray_rocket)

