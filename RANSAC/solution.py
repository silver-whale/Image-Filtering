import numpy as np
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START

    # Return value, I can also return temp_set[0][0]
    largest_set = []

    # Save all 10 round results
    temp_set = []

    # Select 10 random values
    randNum = np.random.randint(low = 0, high = len(matched_pairs), size = 10)

    for i in range(10):
        # Save the points which satisfies the conditon
        points = []
        # number of points
        count = 1
        key1 = matched_pairs[randNum[i]][0]
        key2 = matched_pairs[randNum[i]][1]
        # saved the random point so that the size of consistency set is at minimum 1
        points.append((key1, key2))

        # Calculate Scale difference
        scaleDiff = keypoints2[key2][2] / keypoints1[key1][2]
        # Calculate Orientation difference
        orientationDiff = keypoints1[key1][3] - keypoints2[key2][3]


        # Calculate every matches
        for match in matched_pairs:
            c1 = match[0]
            c2 = match[1]
            
            # Calculate orienctation and scale difference
            scaleDiff2 = keypoints2[c2][2] / keypoints1[c1][2]
            orientationDiff2 = keypoints1[c1][3] - keypoints2[c2][3]

            # if orientation is in the range
            if (abs(orientationDiff2 - orientationDiff)<=math.radians(orient_agreement)):
                # And scale is in the range
                if(scaleDiff*(1-scale_agreement) <= scaleDiff2 and scaleDiff*(1+scale_agreement) >= scaleDiff2):
                    # append point and increase count
                    points.append(match)
                    count += 1
        # save points(indices of point) and number of points to sort       
        temp_set.append((points, count))

    # sort by number of points and get the result
    temp_set.sort(key=lambda x : x[1], reverse=True)

    largest_set = temp_set[0][0]

    ## END
    assert isinstance(largest_set, list)
    return largest_set


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


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
