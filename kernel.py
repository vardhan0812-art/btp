import numpy as np
import cv2

# Read the original image
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array where each row is an RGB pixel
features = image_rgb.reshape((-1, 3))


import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm

def kernel_k_means(X, n_clusters, max_iter=100, tol=1e-4, gamma=1.0, subsample_size=50000):
    n_samples = X.shape[0]

    # Step 1: Subsample if dataset is too large
    if n_samples > subsample_size:
        idx = np.random.choice(n_samples, subsample_size, replace=False)
        X_sub = X[idx]
        n_samples = subsample_size
    else:
        X_sub = X

    # Initialize cluster assignments randomly
    cluster_assignments = np.random.randint(0, n_clusters, n_samples)

    # Compute the kernel matrix
    K = rbf_kernel(X_sub, X_sub, gamma=gamma)

    for iteration in tqdm(range(max_iter), desc="Kernel K-Means Iteration"):
        distances = np.zeros((n_samples, n_clusters))

        for j in range(n_clusters):
            members = cluster_assignments == j
            if np.sum(members) == 0:
                continue

            K_j = K[members][:, members]
            K_sum_j = np.sum(K_j) / (np.sum(members) ** 2)
            distances[:, j] = np.diag(K) - 2 * np.sum(K[:, members], axis=1) / np.sum(members) + K_sum_j

        # Assign clusters
        new_assignments = np.argmin(distances, axis=1)

        # Check for convergence
        if np.all(cluster_assignments == new_assignments):
            break

        cluster_assignments = new_assignments

    return cluster_assignments, idx  # Return subsample indices for possible reconstruction

# Example usage
n_clusters = 5  # Choose number of clusters
gamma = 0.5     # Parameter for RBF kernel
subsample_size = 50000  # Adjust subsample size based on available memory

# features is your dataset with shape (5038848, 3)
labels, subsample_indices = kernel_k_means(features, n_clusters, gamma=gamma, subsample_size=subsample_size)

print("Cluster assignments for the subsampled dataset:", labels)
