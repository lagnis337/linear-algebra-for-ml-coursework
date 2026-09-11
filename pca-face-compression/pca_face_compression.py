"""PCA face-image compression (UCSD Extended Studies, CSE X412.87, Week 7).

Fits PCA on the 400 Olivetti faces, reconstructs them from K principal
components, and reports the reconstruction error for image #125 at K=10 and
K=100 as the assignment required. Figures are written to figures/.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


# Step 1: Load the Olivetti faces dataset
data = fetch_olivetti_faces()
images = data.images
n_samples, h, w = images.shape


# Step 2: Preprocess the data by centering it around the mean
X = images.reshape(n_samples, -1)
mean_face = np.mean(X, axis=0)
X_centered = X - mean_face

# Step 3: Perform PCA
n_components = 100  # Number of principal components to keep
pca = PCA(n_components=n_components, svd_solver="full")
X_pca = pca.fit_transform(X_centered)


###### Student Modified code here

# Step 4: Reconstruct the images using a subset of principal components
# pca.inverse_transform maps the PCA scores back from PC-space (n_components dims)
# to the original 4096-dim pixel space. Because PCA was fit on already-centered
# data, that result is itself centered, so the mean face has to be added back on.
X_reconstructed = pca.inverse_transform(X_pca) + mean_face

###### Student Modified code end


# Reconstruction error (MSE) for image #125, at K=10 and K=100 components,
# as requested in the assignment README.
image_index = 125

def reconstruct_with_k(k):
    pca_k = PCA(n_components=k, svd_solver="full")
    X_pca_k = pca_k.fit_transform(X_centered)
    return pca_k.inverse_transform(X_pca_k) + mean_face

X_recon_10 = reconstruct_with_k(10)
X_recon_100 = reconstruct_with_k(100)

mse_10 = np.mean((X[image_index] - X_recon_10[image_index]) ** 2)
mse_100 = np.mean((X[image_index] - X_recon_100[image_index]) ** 2)

print(f"Mean square error, image #{image_index}, K=10  PCs: {mse_10:.6f}")
print(f"Mean square error, image #{image_index}, K=100 PCs: {mse_100:.6f}")


# Step 5: Visualize the results
n_rows = 4
n_cols = 4
plt.figure(figsize=(10, 8))
for i in range(n_rows * n_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    if i < n_samples:
        plt.imshow(np.hstack((X[i].reshape(h, w), X_reconstructed[i].reshape(h, w))),
                   cmap='gray')
        plt.title("Original vs Reconstructed")
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIG / "reconstruction_grid.png", dpi=120)


# ---------------------------------------------------------------------------
# Bonus (not required by the assignment): MSE as a function of K, for image_index
# ---------------------------------------------------------------------------
ks = list(range(1, 200, 5))
mse_curve = []
for k in ks:
    X_recon_k = reconstruct_with_k(k)
    mse_curve.append(np.mean((X[image_index] - X_recon_k[image_index]) ** 2))

plt.figure(figsize=(6, 4))
plt.plot(ks, mse_curve)
plt.xlabel("Number of principal components (K)")
plt.ylabel(f"MSE (image #{image_index})")
plt.title("Reconstruction MSE vs K")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "mse_vs_k.png", dpi=120)


# Side-by-side detail: original vs K=10 vs K=100 for the graded image
fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
for ax, img, title in zip(
    axes,
    [X[image_index], X_recon_10[image_index], X_recon_100[image_index]],
    ["Original", f"K=10  (MSE {mse_10:.4f})", f"K=100  (MSE {mse_100:.4f})"],
):
    ax.imshow(img.reshape(h, w), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(FIG / "detail_image125.png", dpi=120)
