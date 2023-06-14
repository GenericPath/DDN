import numpy as np
from scipy.optimize import minimize

# Define the real symmetric matrices A and B
A = np.array([[3, -1, 0], [-1, 2, -1], [0, -1, 3]])
B = np.eye(3)  # Identity matrix in this example

# Compute the eigendecomposition of matrix A
eigenvalues, eigenvectors = np.linalg.eigh(A)

# Find the index of the smallest eigenvalue (second smallest since Python is 0-indexed)
index = np.argsort(eigenvalues)[1]

# Extract the smallest eigenvector
v = eigenvectors[:, index]

# Define the optimization problem
def objective(x):
    return x @ A @ x

def constraint1(x):
    return x @ B @ x - 1

def constraint2(x):
    return np.linalg.norm(x) - 1

def constraint3(x):
    return x @ v

# Define the bounds for the variables
bounds = [(None, None), (None, None), (None, None)]  # No bounds on variables

# Initial guess for the eigenvector
x0 = np.ones_like(v)

# Define the constraints
constraints = [{'type': 'eq', 'fun': constraint1},
               {'type': 'eq', 'fun': constraint2},
               {'type': 'eq', 'fun': constraint3}]

# Solve the optimization problem
result = minimize(objective, x0, bounds=bounds, constraints=constraints)

print("minimize sucess:")
print(result.success)
# Extract the second smallest eigenvector
x = result.x

# Print the second smallest eigenvector
print("Second smallest eigenvector:")
print(x)
