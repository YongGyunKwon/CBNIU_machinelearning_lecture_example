import numpy as np

def maxPooling(mat, K, L):
    M,N = mat.shape
    MK = M//K
    NL = N//L
    pmat = mat[:MK*K,:NL*L].reshape(MK,K,NL,L).max(axis=(1,3))
    return pmat

mat = np.array([[20, 200, -5, 23],
                [-13, 134, 119, 100],
                [120, 32, 49, 25],
                [-120, 12, 9 ,23]])
print(maxPooling(mat,2,2))