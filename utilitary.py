import numpy as np


def complex_hermitian_matrix_generation(dim):
    real_values = np.random.uniform(-0.05, 0.05, size=(dim, dim))
    im_values = np.random.uniform(-0.05, 0.05, size=(dim, dim))
    A = np.matrix(real_values + 1j*im_values)
    hermitian_matrix = (1/2)*(A + (A.H)) # le .H conjugue puis transpose
    return hermitian_matrix


def real_hermitian_matrix_generation(dim):
    A = np.matrix(np.random.uniform(-1, 1, size=(dim, dim)))
    symetric_matrix = (1/2)*(A + (A.T))
    return symetric_matrix


def real_GUE(dim):
    A = np.random.randn(dim, dim)
    A_sym = np.triu(A) + np.triu(A, 1).T
    return A_sym

def complex_GUE(dim):
    A = np.matrix(np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))
    A_sym = (1/2)*(A + (A.H))
    return A_sym * (1/2)**(0.5)

def herm_matrix(dim, beta): #beta=1 pour real et 2 pour herm
    B = np.random.randn(dim, dim)
    B_tild = np.random.randn(dim, dim)
    diag = (2/(beta*dim))**0.5 * np.diag(np.diag(B))
    triu = (1/(beta*dim))**0.5 * (np.triu(B,1) + 1j*(beta-1)*np.triu(B_tild,1))
    H = np.zeros((dim,dim)) + diag + triu + triu.T
    return H


if __name__ == '__main__':
    print(herm_matrix(3,2))