import numpy as np

np.random.seed(0)

L = 10
M = 3
N = 3

A = np.random.uniform(size=L*M*N) + np.random.uniform(size=L*M*N) * 1.j
A = np.reshape(A, (L,M,N))
detA = np.linalg.det(A)
print("detA(0):", detA[0])
det3A = np.linalg.det(3*A)
print("det3A(0):", det3A[0])
A_inv = np.linalg.inv(A)
AA_inv = np.einsum("lmk,lkn->lmn", A, A_inv)
print("Identical?: \n", AA_inv[0])
A_invA = np.einsum("lmk,lkn->lmn", A_inv, A)
print("Identical?: \n", A_invA[0])
A_inv = np.linalg.pinv(A)
AA_inv = np.einsum("lmk,lkn->lmn", A, A_inv)
print("Identical?: \n", AA_inv[0])
A_invA = np.einsum("lmk,lkn->lmn", A_inv, A)
print("Identical?: \n", A_invA[0])

A = np.matrix([[3.,1.,2.], [2.,3.,1.]])
b = np.array([2., 1., 4.])
print("A = \n", A)
print("b = \n", b)
print("Ab = \n", np.dot(A,b))

a = np.matrix([3.+2.j, 1.-1.j, 2.+2.j])
b = np.array([2.+5j, 1.-1j, 4.+1.j])
print("a^Hb =", np.inner(np.conjugate(a), b))
print("a^Ha =", np.inner(np.conjugate(a), a))

A = np.random.uniform(size=L*M*N) + np.random.uniform(size=L*M*N) * 1.j
A = np.reshape(A, (L,M,N))
b = np.random.uniform(size=L*M) + np.random.uniform(size=L*M) * 1.j
b = np.reshape(b, (L,M))
print("tr(A) = \n", np.trace(A, axis1=-2, axis2=-1))
print("tr(A) = \n", np.einsum("lmm->l", A))
print("b^H A b = \n", np.einsum("lm,lmn,ln->l", np.conjugate(b), A, b))
print("trA bb^H = \n", np.trace(np.einsum("lmn,ln,lk->lmk", A, b, np.conjugate(b)), axis1=-2, axis2=-1))

