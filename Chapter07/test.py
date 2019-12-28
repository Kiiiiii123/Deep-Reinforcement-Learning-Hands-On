
import numpy as np

if __name__ == "__main__":
    a = np.array([0,1,1,0,0])
    a = a.astype(np.bool)
    print(a)
    b = a.copy()
    print(b)
    print(a[b])