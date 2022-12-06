import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize

# Generate random values from a normal distribution
X = np.random.randn(3, 3)

# Compute A = X.T * X
A = X.T @ X

A[A< 0] = 0

print(A)


# Test eig at different maximum iteration values
for max_iter in [1,5,25,100]:
    try:
        eigenvalues, eigenvectors = eigsh(A, maxiter=max_iter, which='SM', k=1)
        print(f"max_iter = {max_iter}: eigenvector = {eigenvectors}")
    except:
        print('failed for this iteration')
        # The currently converged eigenvalues and eigenvectors can be found as eigenvalues and eigenvectors attributes of the exception object.

# currently we can get flipped eigs
# Note that eigenvectors are not unique. Multiplying by any constant gives another valid eigenvector.
# Just need to multiply by 



# ALTERNATIVE APPROACH

# Collect the values during the iteration of minimize
values = []

def collect_values(x):
    values.append(x)

# Solve the eigenvalue problem using minimize
res = minimize(lambda x: x.dot(A.dot(x)), np.ones(3), method="L-BFGS-B",
    options={"maxiter": 10}, callback=collect_values)

print(res.x)
    

# One potential issue with using minimize to solve the eigenvalue problem is that the objective function passed to minimize must be a scalar function, but the eigenvalue problem involves finding a vector x such that x.T * A * x is minimized, where A is a matrix. This means that the objective function passed to minimize must compute the scalar value x.T * A * x from the input vector x. Depending on the input data and the implementation of the objective function, this may introduce numerical errors or instability.
# To improve the numerical stability of using minimize to solve the eigenvalue problem, you can use the bounds parameter of minimize to constrain the values of x to be within a specified range. This will prevent x from taking on extreme values that may cause numerical errors or instability in the objective function.



# May be more numerically unstable, 
# # Solve the eigenvalue problem using minimize
# res = minimize(lambda x: x.dot(A.dot(x)), np.ones(3), method="L-BFGS-B",
#                bounds=(-1, 1), options={"maxiter": 10})

# print(res.x)
