from PIL import Image 
import numpy as np 
import math

# Numpy Reference : https://numpy.org/doc/stable/reference/routines.html

# Returns a box filter of size n x n
def boxfilter(n):
    # assert : Raises error if condition is not true
    assert n%2 == 1, 'n should be odd number'

    # numpy.full(shape, fill_value, dtype=None, order='C', *, like=None)
    return np.full((n, n), 1.0 / (n*n))

# Returns 1-dimensional Gaussian filter for a given value of sigma
def gauss1d(sigma):
    xlen = math.ceil(sigma * 6)
    # if xlen is even, make it to odd
    if (xlen%2 == 0):
        xlen += 1

    x = np.arange(-(xlen//2), xlen//2+1)
    # Can also be : x = np.array(range(-(xlen//2), xlen//2+1))

    x = np.exp(-x*x/(2.0*sigma*sigma))
     # Another answer
    # x = list(map(lambda x:math.exp(-x*x/(2.0*sigma*sigma)), x))

    total = np.sum(x)
    # Normalize
    x /= total

    return x

# Returns 2-dimensional Gaussian filter for a given value of sigma
def gauss2d(sigma):
    x = np.outer(gauss1d(sigma),gauss1d(sigma))

    total = np.sum(x)

    # Normalize
    x /= total
    
    return x

# Returns convolved array
# The name ‘filter’ already exists in python, so it was changed to ‘fil’
def convolve2d(array, fil):
    zlen = int((len(fil)-1)/2)

    # save the size of initial array and make return array with same size
    n, m = np.shape(array)
    result = np.ones((n,m))

    # get the length of filter array
    flen = len(fil)
    # Padding an array : https://numpy.org/doc/stable/reference/routines.padding.html
    # numpy.pad(array, pad_width, mode='constant', **kwargs)
    # if before = after = pad width -> pad all axes,
    # so (zlen, zlen), (zlen, zlen) means to pad all axis.
    array = np.pad(array, ((zlen, zlen), (zlen, zlen)), mode = 'constant')

    # Convolution : Reverse the filter array and multiply it to the image
    fil = np.flip(fil)

    for i in range(n):
        for j in range(m):
            # Convolution : Move the filter around the image, get sum of that area
            result[i][j] = np.sum( array[i:i+flen, j:j+flen] * fil )

    return result



# Applies Gaussian convolution to a 2D array for a given value of sigma.
def gaussconvolve2d(array, sigma):
    return convolve2d(array, gauss2d(sigma))
    
# Color Image = [x][y][channel] -> 3 dimensional array
def convolveColor(array, fil):
    zlen = int((len(fil)-1)/2)

    # Save the size of original array
    n, m = np.shape(array)[0], np.shape(array)[1]
    # Make 3-dimensional result array, change type to float32
    result = np.ones((n,m,3))
    result = result.astype('float32')

    flen = len(fil)
    # numpy.pad(array, pad_width, mode='constant', **kwargs)
    # No padding to RGB colors, last tuple of pad_width tuple is (0,0)
    array = np.pad(array, ((zlen, zlen), (zlen, zlen),(0,0)), mode = 'constant')

    # Convolution : Reverse the filter array and multiply it to the image
    fil = np.flip(fil)

    # Calculate R, G, B value
    for i in range(n):
        for j in range(m):
            # Convolution : Move the filter around the image, get sum of that area
            result[i][j][0] = np.sum( array[i:i+flen, j:j+flen, 0] * fil )
            result[i][j][1] = np.sum( array[i:i+flen, j:j+flen, 1] * fil )
            result[i][j][2] = np.sum( array[i:i+flen, j:j+flen, 2] * fil )

    return result

def gaussconvolveColor(array, sigma):
    return convolveColor(array, gauss2d(sigma))

# Returns high frequency array
def highFrequency(array, sigma):
    # Original picture - low frequency version
    array = array - gaussconvolveColor(array, sigma)
    return array


def main():
    # BoxFilter
    print(boxfilter(3))
    print(boxfilter(4))
    print(boxfilter(7))

    # Gauss1d
    print("gauss1d(0.3): ", gauss1d(0.3), end="\n\n")
    print("gauss1d(0.5): ", gauss1d(0.5), end="\n\n")
    print("gauss1d(1): ", gauss1d(1), end="\n\n")
    print("gauss1d(2): ", gauss1d(2), end="\n\n")

    # Gauss2d
    print("gauss2d(0.5): ", gauss2d(0.5), end="\n\n")
    print("gauss2d(1): ", gauss2d(1), end="\n\n")

    # # Gaussconvolve2d
    im = Image.open('2b_dog.bmp')
    # Convert to Greyscale Image
    im_grey = im.convert('L')
    im_array = np.asarray(im_grey)
    im_array = im_array.astype('float32') # Change to float32 type
    im2_array = gaussconvolve2d(im_array, 3) # Apply function
    im2_array = im2_array.astype('uint8') # Return the array to uint8 type
    im2 = Image.fromarray(im2_array)
    im.show()
    im2.show()

    # Gaussconvolve2d_color / Low Frequency
    # fileB = Image.open('0b_marilyn.bmp')
    # fileB = Image.open('1b_motorcycle.bmp')
    fileB = Image.open('2b_dog.bmp')
    # fileB = Image.open('3b_tower.bmp')
    fileB_array = np.asarray(fileB)
    fileB_array = fileB_array.astype('float32')

    blurfileB_array32 = gaussconvolveColor(fileB_array, 3)
    blurfileB_array = blurfileB_array32.astype('uint8')
    blurfileB = Image.fromarray(blurfileB_array)
    fileB.show()
    blurfileB.show()

    # # Gaussconvolve2d_color / High Frequency
    # fileA = Image.open('0a_einstein.bmp')
    # fileA = Image.open('1a_bicycle.bmp')
    fileA = Image.open('2a_cat.bmp')
    # fileA = Image.open('3a_eiffel.bmp')
    fileA_array = np.asarray(fileA)
    fileA_array = fileA_array.astype('float32')
    # Add 128 to remove zero-mean with negative values
    sharpfileA_array32 = highFrequency(fileA_array, 3) + 128
    sharpfileA_array = sharpfileA_array32.astype('uint8')
    sharpfileA = Image.fromarray(sharpfileA_array)
    fileA.show()
    sharpfileA.show()

    # Hybrid Image
    # Use originally computed high frequency array
    hybridImage_array = blurfileB_array32 + (sharpfileA_array32-128)
    
    # Remove Dangling values
    hybridImage_array[hybridImage_array>255] = 255
    hybridImage_array[hybridImage_array<0] = 0

    hybridImage_array = hybridImage_array.astype('uint8')

    hybridImage = Image.fromarray(hybridImage_array)
    hybridImage.show()
    

if __name__ == "__main__":
	main()