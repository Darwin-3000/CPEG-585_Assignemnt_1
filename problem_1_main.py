'''
Created on Jan 11, 2023
@author: ahmed-notebook
'''

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def main():
    img = Image.open('./907_img_.png').convert('L')

    width, height = img.size
    npimage = np.array(img)

    # Brightness & contrast parameters
    alpha = 1.5   # contrast
    beta = 40     # brightness

    # Create empty image
    enhanced = np.zeros((height, width), dtype=np.uint8)

    # Apply g(x,y) = alpha*f(x,y) + beta
    for row in range(height):
        for col in range(width):
            pixel = alpha * npimage[row, col] + beta

            # Manual clamping
            if pixel < 0:
                pixel = 0
            elif pixel > 255:
                pixel = 255

            enhanced[row, col] = int(pixel)

    # Save enhanced image
    Image.fromarray(enhanced).save('./907_img_enhanced.png')

    # Display image
    plt.imshow(enhanced, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
