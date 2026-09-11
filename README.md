# Linear Algebra for Machine Learning — coursework

Graded assignments from **CSE X412.87 Linear Algebra for Machine Learning**,
UC San Diego Division of Extended Studies, Summer 2026. Final grade **A+**
(3 quarter units; 100% on all quizzes and the final exam).

The course covered matrix algebra, linear systems, vector spaces, matrix
decompositions (LU, QR, eigendecomposition, SVD), PCA, and gradient descent for
linear models, in NumPy and Octave. The two coding assignments are below, each
with the submitted code, the data, the figures, and a README explaining the
mathematics and the results.

| Assignment | Grade | What it exercises |
|---|---|---|
| [PCA face-image compression](pca-face-compression/) | 10/10 | Centering, PCA via SVD, projection and reconstruction, reconstruction error vs. number of components |
| [Linear regression: gradient descent vs. the normal equation](linear-regression-gradient-descent/) | 12/10 (bonus work) | Closed-form least squares, deriving the MSE gradient, batch gradient descent, feature standardisation, learning-rate sensitivity |

## Reproducing

```bash
pip install numpy matplotlib scikit-learn
python pca-face-compression/pca_face_compression.py           # ~30 s, downloads Olivetti faces via scikit-learn
python linear-regression-gradient-descent/linear_regression.py # < 1 s
```

The regression assignment was submitted in Octave (`*.m` files, included);
`linear_regression.py` is a line-for-line Python port that reproduces the
submitted numbers exactly and regenerates every figure.
