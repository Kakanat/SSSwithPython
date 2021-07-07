import numpy as np


def phi_laplacian(s_hat):
    norm = np.abs(s_hat)
    phi = s_hat / np.maximum(norm, 1.e-18)
    return(phi)

def contrast_laplacian(s_hat):
    norm = 2. * np.abs(s_hat)
    return(norm)

def execute_natural_gradient_ica(x, W, phi_func=phi_laplacian, contrast_func=contrast_laplacian,
                                 mu=1.0, n_ica_iterations=20, is_use_non_holonomic=True):
    M = np.shape(x)[0]
    cost_buff = []
    for t in range(n_ica_iterations):
        s_hat = np.einsum('kmn,nkt->mkt', W, x)
        G = contrast_func(s_hat)
        cost = np.sum(np.mean(G, axis=-1)) - np.sum(2. * np.log(np.abs(np.linalg.det(W))))
        cost_buff.append(cost)
        phi = phi_func(s_hat)
        phi_s = np.einsum('mkt,nkt->ktmn', phi, np.conjugate(s_hat))
        phi_s = np.mean(phi_s, axis=1)
        I = np.eye(M,M)
        if is_use_non_holonomic == False:
            deltaW = np.einsum('kmi,kin->kmn', I[None,...] - phi_s, W)
        else:
            mask = (np.ones((M,M)) - I)[None,...]
            deltaW = np.einsum('kmi,kin->kmn', np.multiply(mask, - phi_s), W)
        W = W + mu * deltaW
    s_hat = np.einsum('kmn,nkt->mkt', W, x)
    return(W, s_hat, cost_buff)