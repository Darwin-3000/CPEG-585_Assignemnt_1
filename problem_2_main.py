
'''
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def gamma_image(img, gamma):
    h = img.shape[0]
    w = img.shape[1]

    gamma_image_mat = np.zeros((h, w), dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            # Normalize pixel to [0,1]
            normalized = img[row, col] / 255.0

            # Apply gamma correction
            pixel = (normalized ** gamma) * 255

            # Clamp values
            if pixel < 0:
                pixel = 0
            elif pixel > 255:
                pixel = 255

            gamma_image_mat[row, col] = int(pixel)

    return gamma_image_mat


def process_image(image_filename, gamma):
    img = Image.open(image_filename).convert('L')
    npimg = np.array(img)

    g_img = gamma_image(npimg, gamma)

    # Show images
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(npimg, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(g_img, cmap='gray')
    plt.title(f'Gamma Corrected (γ={gamma})')
    plt.axis('off')

    plt.show()

    # Plot histograms
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.hist(npimg.flatten(), bins=256, range=(0,255))
    plt.title('Original Histogram')

    plt.subplot(1,2,2)
    plt.hist(g_img.flatten(), bins=256, range=(0,255))
    plt.title('Gamma Corrected Histogram')

    plt.show()

    return g_img


def main():
    # Image 1
    process_image('./907_img_.png', gamma=0.6)

    # Image 2
    process_image('./Unequalized_Hawkes_Bay_NZ.jpg', gamma=1.8)


if __name__ == '__main__':
    main()
'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Function: Apply gamma correction
# --------------------------
def gamma_image(img, gamma):
    h, w = img.shape
    gamma_img = np.zeros((h, w), dtype=np.uint8)  # output image

    for row in range(h):
        for col in range(w):
            normalized = img[row, col] / 255.0          # normalize 0-1
            corrected = (normalized ** gamma) * 255     # apply gamma formula

            # clamp to 0-255
            if corrected < 0:
                corrected = 0
            elif corrected > 255:
                corrected = 255

            gamma_img[row, col] = int(corrected)
    return gamma_img

# --------------------------
# Function: Display image and histograms
# --------------------------
def show_image_with_histogram(original, gamma_img, title='Image'):
    fig, axes = plt.subplots(2, 2, figsize=(10,6))
    
    # Original image
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # Gamma corrected image
    axes[0,1].imshow(gamma_img, cmap='gray')
    axes[0,1].set_title('Gamma Corrected')
    axes[0,1].axis('off')
    
    # Histogram of original
    axes[1,0].hist(original.flatten(), bins=256, range=(0,255))
    axes[1,0].set_title('Original Histogram')
    
    # Histogram of gamma corrected
    axes[1,1].hist(gamma_img.flatten(), bins=256, range=(0,255))
    axes[1,1].set_title('Gamma Histogram')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()  # only one plt.show() per figure

# --------------------------
# Main program
# --------------------------
def main():
    # Image 1
    img1 = np.array(Image.open('./907_img_.png').convert('L'))
    gamma1 = 0.6
    img1_gamma = gamma_image(img1, gamma1)
    show_image_with_histogram(img1, img1_gamma, title='Image 1')

    # Image 2
    img2 = np.array(Image.open('./Unequalized_Hawkes_Bay_NZ.jpg').convert('L'))
    gamma2 = 1.8
    img2_gamma = gamma_image(img2, gamma2)
    show_image_with_histogram(img2, img2_gamma, title='Image 2')

# --------------------------
# Run the program
# --------------------------
if __name__ == '__main__':
    main()
