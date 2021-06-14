import numpy as np

np.random.seed(0)

L = 10
K = 5
M = 3
R = 3
N = 3

A = np.random.uniform(size=L*K*M*R) + np.random.uniform(size=L*K*M*R) * 1.j
A = np.reshape(A, (L,K,M,R))

B = np.random.uniform(size=K*R*N) + np.random.uniform(size=K*R*N) * 1.j
B = np.reshape(B, (K,R,N))

C = np.einsum("lkmr,krn->lkmn", A, B)
print("shape(C):", np.shape(C))

print("A(0,0)B(0,0)=\n",np.matmul(A[0,0,...],B[0,...]))
print("C(0,0)=\n",C[0,0,...])

C = np.einsum("lkmr,krn->kmn", A, B)
print("shape(C):", np.shape(C))

for l in range(L):
    if l == 0:
        C_2 = np.matmul(A[l,0,...],B[0,...])
    else:
        C_2 = C_2 + np.matmul(A[l,0,...],B[0,...])
    
print("C_2(0)=\n", C_2)
print("C(0)=\n", C[0,...])

C = np.einsum("lkmn,kmn->lkmn", A, B)
print("shape(C):", np.shape(C))

print("A(0,0)B(0,0)=\n",np.multiply(A[0,0,...],B[0,...]))
print("C(0,0)=\n",C[0,0,...])
