import sys
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    
    # try:
    #     height, width, color = img.shape
    #     m, n = kernel.shape
    #     result = np.zeros((img.shape))

    #     if((m%2 == 0 and n%2 == 0) and (m <= height and n <= width)):
    #         for k in range(color):
    #             for i in range(m):
    #                 for j in range(n):
    #                     if((i - m/2)>=0 and (j - n/2)>=0):
    #                         img[i][j][k] = 1

    try:

        # img_height, img_width = img.shape[:2]
        # kernel_height, kernel_width = kernel.shape

        # if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        #     raise ValueError("Kernel dimensions should be odd.")

        # pad_height = kernel_height // 2
        # pad_width = kernel_width // 2

        # #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = CAN WE USE np.pad???
        # # Handling RGB and grayscale images
        # if len(img.shape) == 3:
        #     padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        #     padded_img_height = height + 2 * pad_height
        #     padded_img_width = width + 2 * pad_width
        #     # Initialize a new array filled with zeros for padding
        #     padded_img = np.zeros((padded_img_height, padded_img_width, color_channels))
        #     # Copy the original image into the center of the padded image
        #     padded_img[pad_height:pad_height + height, pad_width:pad_width + width, :] = img
        #     color_channels = img.shape[2]
        # else:
        #     padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        #     color_channels = 1

        # result = np.zeros_like(img)

        # for y in range(img_height):
        #     for x in range(img_width):
        #         for channel in range(color_channels):
        #             sum = 0
        #             for ky in range(kernel_height):
        #                 for kx in range(kernel_width):
        #                     # Image coordinates with padding
        #                     iy = y + ky
        #                     ix = x + kx

        #                     # Only apply the kernel to the overlapping region
        #                     if 0 <= iy - pad_height < img_height and 0 <= ix - pad_width < img_width:
        #                         if color_channels == 1:
        #                             sum += kernel[ky, kx] * padded_img[iy, ix]
        #                         else:
        #                             sum += kernel[ky, kx] * padded_img[iy, ix, channel]

        #             if color_channels == 1:
        #                 result[y, x] = sum
        #             else:
        #                 result[y, x, channel] = sum

        # return result

        img_height, img_width = img.shape[:2]
        kernel_height, kernel_width = kernel.shape
        pad_height, pad_width = kernel_height // 2, kernel_width // 2

        # Handling both grayscale and RGB images
        if len(img.shape) == 3:
            depth = img.shape[2]
        else:
            depth = 1

        # Initialize the output image with zeros
        output = np.zeros_like(img)

        # Perform cross-correlation
        for y in range(img_height):
            for x in range(img_width):
                for d in range(depth):  # For each color channel
                    sum = 0
                    for ky in range(kernel_height):
                        for kx in range(kernel_width):
                            # Calculate the position of the current kernel element over the image
                            iy = y + ky - pad_height
                            ix = x + kx - pad_width
                            # Check if the kernel position is inside the image boundaries
                            if 0 <= iy < img_height and 0 <= ix < img_width:
                                if depth == 1:
                                    sum += kernel[ky, kx] * img[iy, ix]
                                else:
                                    sum += kernel[ky, kx] * img[iy, ix, d]
                    if depth == 1:
                        output[y, x] = sum
                    else:
                        output[y, x, d] = sum

        return output
         
    except:
        raise Exception("TODO cross_correlation_2d in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN

    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = CAN WE USE np.flip???

    try:
        # Flip the kernel manually: reverse rows and then reverse columns.
        flipped_kernel = kernel[::-1, ::-1]

        return cross_correlation_2d(img, flipped_kernel)

    except:
        raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    ''' 
    # TODO-BLOCK-BEGIN

    try:
        import math

        # def calculateGaussianFunction(x, y):
        #     exponent = (-1)*(x*x + y*y)/(2*sigma*sigma)
        #     numerator = math.pow(math.e, exponent)
        #     denominator = 2*math.pi*sigma*sigma
        #     return numerator/denominator
        
        kernel = np.zeros((height, width))
        center_y = height // 2
        center_x = width // 2

        gaussian_constant = 1 / (2 * np.pi * sigma ** 2)

        # Compute Gaussian values for each cell in the kernel
        for x in range(width):
            for y in range(height):
                dx = x - center_x
                dy = y - center_y
                kernel[y, x] = gaussian_constant * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

        # Normalize the kernel to avoid reducing the brightness of the image
        kernel /= np.sum(kernel)

        return kernel

    except:
        raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    try:
        if len(img.shape) == 3:
            color_channels = 3
        else:
            color_channels = 1
        
        kernel = gaussian_blur_kernel_2d(sigma, size, size)
        
        return convolve_2d(img, kernel)

    except:
        raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    try:
        smoothed_img = low_pass(img, sigma, size)
        return img - smoothed_img

    except:
        raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

