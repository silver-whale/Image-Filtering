import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

from sklearn.inspection import DecisionBoundaryDisplay

# 1. Create Datasets
# random_state = 20
X1, y1 = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=20)
plt.subplot(1,3,1)
plt.scatter(X1[:,0], X1[:,1], marker='o', c=y1)
plt.title("1-1. random_state=20")

# random_state = 30
X2, y2 = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=30)
plt.subplot(1,3,2)
plt.scatter(X2[:,0], X2[:,1], marker='o', c=y2)
plt.title("1-2. random_state=30")

# random_state = 40
X3, y3 = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=40)
plt.subplot(1,3,3)
plt.scatter(X3[:,0], X3[:,1], marker='o', c=y3)
plt.title("1-3. random_state=40")

plt.show()

# 2. Train SVM

rList = [20, 30, 40]
Clist = [10, 1, 0.1]
Xlist = [X1, X2, X3]
ylist = [y1, y2, y3]

for i in range(1, 4):
    for j in range(1, 4):
        plt.subplot(3, 3, (i-1)*3 + j)
        clf = SVC(kernel='linear', C=Clist[j-1]).fit(Xlist[i-1],ylist[i-1])
        plt.scatter(Xlist[i-1][:,0], Xlist[i-1][:,1], marker='o', c=ylist[i-1])
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(clf, Xlist[i-1], plot_method="contour", colors="k", levels = [-1, 0, 1], linestyles=["--","-","--"], ax=ax)
        ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],s=100, linewidth=1, facecolors="none", edgecolors="k")
        plt.title("Random_state:{}, Misclassification:{}".format(rList[i-1], Clist[j-1]))
plt.show()