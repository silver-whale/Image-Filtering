import numpy as np
import matplotlib.pyplot as plt

img1 = plt.imread('./data/warrior_a.jpg')
img2 = plt.imread('./data/warrior_b.jpg')

i1 = plt.imread('./data/graffiti_a.jpg')
i2 = plt.imread('./data/graffiti_b.jpg')

cor1 = np.load("./data/warrior_a.npy")
cor2 = np.load("./data/warrior_b.npy")

c1 = np.load("./data/graffiti_a.npy")
c2 = np.load("./data/graffiti_b.npy")

def compute_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
        
    F = None
    
    # x1, x2 is already Transposed(x = (u,v,1).Transpose)

    # make A matrix, size = Nx9
    # uu', vu', u', uv', vv', v', u, v, 1
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [ x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]

    # numpy.SVD returns u, s, and vh
    # U: nxn, Sigma: nx9, vh: 9x9
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

    # In SVD, Orthogonal Matrix's last row contains Eigenvectors of A(t)A -> We use [-1] row of this matrix

    # Compute F
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    v = vh.T
    F = v[:, -1].reshape(3,3)
    # Last row contains Eigenvectors of A(t)A

    # Compute F'
    u, s, vh = np.linalg.svd(F, full_matrices=True)

    # Last row in Sigma should be zero -> rank(F) = 2
    s[2] = 0

    # Use diagonal values
    s = np.diag(s)
    F = np.dot(u, np.dot(s, vh))
    
    return F


def compute_norm_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = T1 @ x1
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = T2.T @ F @ T1
    
    return F


def compute_epipoles(F):
    e1 = None
    e2 = None
    
    # Use SVD to compute e1, e2 (1x3 -> change the last value to 1)
    u, s, vh = np.linalg.svd(F)
    # Last row = Eigenvector
    e1 = vh[-1]
    # Change last value to 1
    e1 = e1 / e1[-1]
    
    u, s, vh = np.linalg.svd(F.T)
    e2 = vh[-1]
    e2 = e2 / e2[-1]

    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))

    # Y value starts from top, sequence is different from x value
    axes[0].set_xlim (0,img1.shape[1])
    axes[0].set_ylim (img1.shape[0],0)
    axes[1].set_xlim (0,img1.shape[1])
    axes[1].set_ylim (img1.shape[0],0)
    
    axes[0].imshow(img1)
    axes[1].imshow(img2)

    # For every points
    for i in range(cor1.shape[1]):
        # Mark the point
        axes[0].scatter(cor1[0][i], cor1[1][i], marker = 'o')
        axes[1].scatter(cor2[0][i], cor2[1][i], marker = 'o')

        # Draw a line(e2-cor1, e1-cor2)
        x1 = np.array([e2[0], cor1[0][i]])
        y1 = np.array([e2[1], cor1[1][i]])

        x2 = np.array([e1[0], cor2[0][i]])
        y2 = np.array([e1[1], cor2[1][i]])

        # Calculate the gradient and intercept
        s1, d1 = np.polyfit(x1, y1, 1)
        s2, d2 = np.polyfit(x2, y2, 1)

        # Draw linear function
        x = np.linspace(0, img1.shape[1])
        axes[0].plot(x, s1*x + d1)
        axes[1].plot(x, s2*x + d2)

    plt.show()

    return

draw_epipolar_lines(img1, img2, cor1, cor2)
draw_epipolar_lines(i1, i2, c1, c2)
