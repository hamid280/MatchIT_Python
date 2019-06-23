import numpy as np
import pandas as pd


A = np.random.randint(10, size=(3,4))

print(A)

print(A - A[0])


data = pd.DataFrame(A)

print(data)
print()
print()
print()
print(data.values)