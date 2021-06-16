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

B = np.einsum("lmk,lnk->lmn", A, np.conjugate(A))
w, v = np.linalg.eig(A)
A_reconst = np.einsum("lmk,lk,lkn->lmn", v, w, np.linalg.inv(v))
print("A[0]: \n", A[0])
print("A_reconst[0]: \n", A_reconst[0])
w, v = np.linalg.eigh(B)
B_reconst = np.einsum("lmk,lk,lnk->lmn", v, w, np.conjugate(v))
print("B[0]: \n", B[0])
print("B_reconst[0]: \n", B_reconst[0])

def batch_kron(A, B):
    if np.shape(A)[:-2] != np.shape(B)[:-2]:
        print("error")
        return None
    else:
        return(np.reshape(np.einsum("...mn,...ij->...minj", A, B),
                          np.shape(A)[:-2] + (np.shape(A)[-2] * np.shape(B)[-2], 
                          np.shape(A)[-1] * np.shape(B)[-1])))
R = 3
T = 3
A = np.random.uniform(size=L*M*R) + np.random.uniform(size=L*M*R) * 1.j
A = np.reshape(A, (L,M,R))
X = np.random.uniform(size=R*N) + np.random.uniform(size=R*N) * 1.j
X = np.reshape(X, (R,N))
B = np.random.uniform(size=L*N*T) + np.random.uniform(size=L*N*T) * 1.j
B = np.reshape(B, (L,N,T))
D = np.random.uniform(size=L*M*T) + np.random.uniform(size=L*M*T) * 1.j
D = np.reshape(D, (L,M,T))

C = batch_kron(np.transpose(B, (0,2,1)), A)
C_2 = np.kron(np.transpose(B[0,...], (1,0)), A[0,...])
print("error:", np.sum(np.abs(C[0,...] - C_2)))

vecX = np.reshape(np.transpose(X, [1,0]), (N*R))
AXB = np.einsum("lmr,rn,lnt->lmt", A, X, B)
vecAXB = np.reshape(np.transpose(AXB, [0,2,1]), (L, T*M))
CvecX = np.einsum("lmr,r->lm", C, vecX)
print("error =", np.sum(np.abs(vecAXB - CvecX)))

vecD = np.reshape(np.transpose(D, [0,2,1]), (L, T*M))
vecX = np.einsum("mr,r->m", np.linalg.inv(np.sum(C, axis=0)), np.sum(vecD, axis=0))
X = np.transpose(np.reshape(vecX, (N,R)), (1,0))
sum_AXB = np.einsum("lmr,rn,lnt->mt", A, X, B)
sum_D = np.sum(D, axis=0)
print("error =", np.sum(np.abs(sum_AXB - sum_D)))