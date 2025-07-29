import numpy as np
from arch.univariate.distribution import SkewStudent
import inspect

# 1. Instantiate the distribution object
skew_t_dist = SkewStudent()

# 2. Use Python's 'inspect' module to get the exact function signature
ppf_signature = inspect.signature(skew_t_dist.ppf)

# 3. Print the results
print("--- ARCH Library SkewStudent PPF Test ---")
print(f"The required arguments for .ppf() are: {ppf_signature}")

# 4. Attempt to call it with the parameters we've been trying
try:
    # This will fail, but it will confirm the exact nature of the error
    print("\nAttempting to call with 'nu' and 'lam'...")
    skew_t_dist.ppf(np.array([0.01, 0.05]), nu=8.0, lam=0.1)
except TypeError as e:
    print(f"Call failed as expected: {e}")

try:
    # This will also fail, confirming the other error
    print("\nAttempting to call with 'eta' and 'lambda'...")
    skew_t_dist.ppf(np.array([0.01, 0.05]), eta=8.0, lambda_=-0.1)
except TypeError as e:
    print(f"Call failed as expected: {e}")