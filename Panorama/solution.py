import numpy as np
import cv2
import math
import random

def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START

    # Return Value, index of match points
    matched_pairs = []
    
    # Compare image1's point with each of image2's points.
    for i in range(len(descriptors1)):
        # This list contains len(descriptors2) elements.
        angles = []
        # Calculate each angles
        for j in range(len(descriptors2)):
            angle = math.acos(np.dot(descriptors1[i], descriptors2[j]))
            angles.append(angle)
        # Sort the angles, and append the smallest one which satisfies the condition
        # Condition : Best angle / Second angle <= threshold.
        sAngles = sorted(angles)
        if sAngles[0] / sAngles[1] <= threshold:
            second = angles.index(sAngles[0])
            matched_pairs.append((i, second))

    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    # Add 1 to every row's end -> add [[1], [1], [1], ...] to column(axis = 1)
    hpoints = np.append(xy_points, [[1] for i in range(xy_points.shape[0])], axis = 1)

    # Returning array
    xy_points_out = np.zeros((xy_points.shape[0], 2))

    # For every point
    for i in range(hpoints.shape[0]):
        # Calculate H multiply homogenous point
        # Be aware of the sequence, h should be in front
        value = np.dot(h, hpoints[i])

        # If value is 0, make it to 1e-10
        if value[2] == 0:
            value[2] = 1e-10

        # Make the last coordinate to 1
        value[0] /= value[2]
        value[1] /= value[2]

        # Except last coordinate
        xy_points_out[i] = value[0:2]

    # END
    return xy_points_out


def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START

    # Keep track of maximum count value
    max_count = -1

    # First we will set h with 3x3 zero-num array
    h = np.array([[0, 0, 0] for i in range(3)])

    for i in range(num_iter):
        # count of points which satisfies tol value
        count = 0

        # Select 4 random number
        randNum = np.random.randint(low = 0, high = len(xy_ref), size = 4)

        # Make (4, 2) array with selected value from xy_src and xy_ref
        src = np.array([[xy_src[randNum[m]][0], xy_src[randNum[m]][1]] for m in range(4)])
        ref = np.array([[xy_ref[randNum[m]][0], xy_ref[randNum[m]][1]] for m in range(4)])
        
        # calculate h
        temp_h, _ = cv2.findHomography(src, ref)

        # Project all points from src to ref, using calculated h matrix
        proj = KeypointProjection(xy_src, temp_h)

        # calculate all points' Euclidian distance
        for j in range(proj.shape[0]):
            # If moved point and source's point's distance is within range, increase count
            if(math.sqrt((xy_ref[j][0] - proj[j][0])**2 + (xy_ref[j][1] - proj[j][1])**2) <= tol):
                count += 1

        # if the value is the maximum, change h to calculated one
        if (max_count <= count):
            max_count = count
            h = temp_h

    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)

    return h
