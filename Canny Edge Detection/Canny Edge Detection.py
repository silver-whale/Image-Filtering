from PIL import Image
import math
import numpy as np

# Numpy Reference : https://numpy.org/doc/stable/reference/routines.html

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
def convolve2d(array,fil):
    zlen = int((len(fil)-1)/2)

    # save the size of initial array and make return array with same size
    n, m = np.shape(array)

    result = np.zeros((n,m))
    result.astype('float32')

    # get the length of filter array
    flen = len(fil)
    # Padding an array : https://numpy.org/doc/stable/reference/routines.padding.html
    # numpy.pad(array, pad_width, mode='constant', **kwargs)
    # if before = after = pad width -> pad all axes,
    # so (zlen, zlen), (zlen, zlen) means to pad all axis.
    array = np.pad(array, ((zlen, zlen), (zlen, zlen)), mode='constant')

    # Convolution : Reverse the filter array and multiply it to the image
    fil = fil.T
    for i in range(n):
        for j in range(m):
            # Convolution : Move the filter around the image, get sum of that area
            result[i][j] = np.sum( np.multiply(array[i:i+flen, j:j+flen], fil) )
            
    result.astype('float32')
    
    return result

# Applies Gaussian convolution to a 2D array for a given value of sigma.
def gaussconvolve2d(array,sigma):
    return convolve2d(array, gauss2d(sigma))

# Returns gradient magnitude and direction of input img.
def sobel_filters(img):

    # Make sobel filters
    x_filter = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]        
    ], dtype = 'float32')

    y_filter = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]    
    ], dtype='float32')
    
    # convolve2d function includes array reversing (flip)
    # Get X and Y intensity arrays
    x_intensity = convolve2d(img, x_filter)
    y_intensity = convolve2d(img, y_filter)

    # Calculate Gradient
    G = np.hypot(x_intensity, y_intensity)
    # Remove values over 255
    G *= 255.0 / G.max()

    theta = np.arctan2(y_intensity, x_intensity)
    
    # Show picture
    # G = G.astype('uint8') # Return the array to uint8 type
    # G_img = Image.fromarray(G)
    # G_img.show()

    return (G, theta)

# Performs non-maximum suppression.
def non_max_suppression(G, theta):
    H, W = np.shape(G)
    # Convert radian to angle
    angle = np.rad2deg(theta)

    # result array
    res = np.zeros((H, W), dtype='uint8')

    # Divide 360 by 45 -> 8 sectors, the center value should be (0, 45, 90, 135)
    # So, the range becomes +/-[0-22.5, 0+22.5], +/-[45-22.5, 45+22.5], +/-[90-22.5, 90+22.5], +/-[135-22.5, 135+22.5]
    for i in range(1, H-1):
        for j in range(1, W-1):
            # +/-[0-22.5, 0+22.5]
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
                b = G[i+1, j]
                c = G[i-1, j]
            # +/-[45-22.5, 45+22.5]
            elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
                b = G[i+1, j-1]
                c = G[i-1, j+1]    
            # +/-[90-22.5, 90+22.5]
            elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
                b = G[i, j+1]
                c = G[i, j-1]
            # +/-[135-22.5, 135+22.5]
            elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
                b = G[i+1, j+1]
                c = G[i-1, j-1]   
 
            # Compare the middle value with interpolated values
            # if the middle is the largest, maintain it
            # else, set the value to 0
            if (G[i,j] >= b) and (G[i,j] >= c):
                res[i,j] = G[i,j]
            else:
                res[i,j] = 0

    # Show picture
    res = res.astype('uint8')
    # res_img = Image.fromarray(res)
    # res_img.show()

    return res

# Sort Edges
def double_thresholding(img):
    diff = np.max(img) - np.min(img)
    t_high = np.min(img) + diff * 0.15
    t_low = np.min(img) + diff * 0.03

    H, W = img.shape

    res = np.zeros((H,W))

    # np.where(condition, [x, y, ]/)
    # return elements chosen from x or y depending on condition
    n_x, n_y = np.where(img < t_low)
    # Normal and operator does not work inside where function, so we should use np.logical_and
    w_x, w_y = np.where(np.logical_and((img < t_high),(img >= t_low)))
    s_x, s_y = np.where(img >= t_high)

    res[n_x, n_y] = 0
    res[w_x, w_y] = 80
    res[s_x, s_y] = 255

    # Show picture
    # res = res.astype('uint8') # Return the array to uint8 type
    # res_img = Image.fromarray(res)
    # res_img.show()

    return res

# Find weak edges connected to strong edges and link them.
# Iterate over each pixel in strong_edges and perform depth first
# search across the connected pixels in weak_edges to link them.
# Here we consider a pixel (a, b) is connected to a pixel (c, d)
# if (a, b) is one of the eight neighboring pixels of (c, d)
def hysteresis(img):

    H, W = img.shape

    # Directions to move
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]
    # Save visited nodes
    vis = [[False for i in range(W)] for j in range(H)]

    stack = []
    stack.append([0,0])

    while (len(stack) > 0):
        current = stack[len(stack) - 1]
        stack.remove(stack[len(stack)-1])
        row = current[0]
        col = current[1]

        if (row < 0 or col < 0 or row >= H or col >= W):
            continue

        if (vis[row][col]): continue
        
        vis[row][col] = True
        
        # If current pixel is weak edge and the strong edge is nearby
        if(img[row, col] == 80):
            # Can also use 'in' function
            if img[row+1, col] == 255 or img[row, col+1] == 255 or img[row+1, col+1] == 255 or \
            img[row-1, col] == 255 or img[row, col-1] == 255 or img[row-1, col-1] == 255 or \
                img[row+1, col-1] == 255 or img[row-1, col+1] == 255:
                img[row, col] = 255
            else:
                img[row, col] = 0         

        # Move in four directions
        for i in range(4):
            nx = row + dx[i]
            ny = col + dy[i]
            stack.append([nx, ny])               

    # Show picture
    res = img
    res = res.astype('uint8')
    res_img = Image.fromarray(res)
    res_img.show()

    return res

def main():
    im = Image.open('iguana.bmp')
    im_grey = im.convert('L')
    im_array = np.asarray(im_grey)
    im_array = im_array.astype('float32')
    
    img_array = gaussconvolve2d(im_array, 1.6)
    img = Image.fromarray(img_array)
    
    # Original picture
    # im.show()

    # Greyscaled Picture
    # img.show()

    # Sobel Filter
    G, theta = sobel_filters(img_array)

    # Non max suppression
    nonMax = non_max_suppression(G, theta)

    # Double thresholding
    doubleThresh = double_thresholding(nonMax)

    # Hysteresis, Final result
    hyst = hysteresis(doubleThresh)


if __name__ == "__main__":
	main()

