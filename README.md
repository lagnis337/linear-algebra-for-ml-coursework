# Linear Algebra for Machine Learning: coursework

Graded assignments from **CSE X412.87 Linear Algebra for Machine Learning**,
UC San Diego Division of Extended Studies, Summer 2026. The final grade was
**A+** (3 quarter units, with 100% in all the quizzes and in the final examination).

The course included matrix algebra, linear systems, vector spaces, matrix
decompositions (LU, QR, eigendecomposition, and SVD), PCA, and gradient descent
for linear models. The tools were NumPy and Octave. The two coding assignments
are below. Each one contains the code that I submitted, the data, the figures,
and a README that explains the mathematics and the results.

| Assignment | Grade | What it uses |
|---|---|---|
| [PCA face-image compression](pca-face-compression/) | 10/10 | Centring of the data, PCA with the SVD, projection and reconstruction, and the reconstruction error against the number of components |
| [Linear regression: gradient descent against the normal equation](linear-regression-gradient-descent/) | 12/10 (bonus work) | The closed-form least-squares solution, the gradient of the mean squared error, batch gradient descent, feature standardisation, and the effect of the learning rate |

## Reproduce the results

```bash
pip install numpy matplotlib scikit-learn
python pca-face-compression/pca_face_compression.py            # ~30 s, scikit-learn downloads the Olivetti faces
python linear-regression-gradient-descent/linear_regression.py # < 1 s
```

I submitted the regression assignment in Octave, and the `*.m` files are in this
repository. The file `linear_regression.py` is a Python version of the same
work. It gives the same numbers as the submitted files and makes all the figures
again.
