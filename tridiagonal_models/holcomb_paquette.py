import numpy as np
from numpy.polynomial import polynomial as P
import sys

sys.path.append('../dyson_brownian_motion/')
from plot_tools import adjust_spines, sc_law, plot_dpp, plot_traj_with_histo, plot_traj_with_histo_sclaw

SEED = 32
np.random.seed(SEED)

def tridiagonal_model_gaussian(dim, beta):
    diag = (1/np.sqrt(dim))*np.diag(np.random.randn(dim))
    u_diag_vect = np.zeros(dim-1)
    for i in range(0,dim-1):
        #u_diag_vect[i] = np.sqrt(np.random.chisquare((dim-(i+1))*beta))
        u_diag_vect[i] = np.linalg.norm( (1/np.sqrt(2*dim))*np.random.randn((dim-(i+1))*beta) )
    u_diag = np.diag(u_diag_vect, 1)
    H_b = diag + u_diag + u_diag.T
    return H_b


class tridiag_dyson:
    def __init__(self, n_traj, n_samples, tf):
        self.n_traj = n_traj
        self.dim = n_traj
        self.n_samples = n_samples
        self.tf = tf
        self.dt = tf/n_samples
        init_mat = tridiagonal_model_gaussian(self.dim, beta=2)
        self.tridiag_matrices = [init_mat]
        self.eigen_values = [sorted(np.linalg.eigvalsh(init_mat), reverse=True)]
        self.brownians = [np.zeros(n_traj)]
        self.p = None
        self.G = None
        self.O = None
        self.generate()

    def compute_polynomials(self, U):
        p_unit = (0, 1)
        p = []
        p.append((1))
        p_k_1_1 = P.polymul(p_unit, p[0])
        p_k_1_2 = P.polymul(U[0, 0], p[0])
        p_new = (1 / U[0, 1]) * P.polysub(p_k_1_1, p_k_1_2)
        p.append(p_new)
        for k in range(2,self.dim):
            #print(k)
            p_k_1_1 = P.polymul(p_unit, p[k-1])
            p_k_1_2 = P.polymul(U[k-1,k-1], p[k-1])
            p_k_2 = P.polymul(U[k-1,k-2], p[k-2])
            p_new = (1/U[k-1,k])*P.polysub( P.polysub(p_k_1_1, p_k_1_2), p_k_2 )
            p.append(p_new)
        self.p = p
        self.p_n = P.polyadd(P.polymul(-U[self.dim-2,self.dim-1], p[self.dim-2]), P.polymul(P.polysub(p_unit, U[self.dim-1,self.dim-1]), p[self.dim-1]))
        return p

    def compute_p_kl(self, k ,l, lbda, eigenvalues):
        p_kl = 0
        if l <= k:
            pass
        else:
            for i, eigenvalue in enumerate(eigenvalues):
                if eigenvalue != lbda :
                    frac = (P.polyval(lbda, self.p[l]) - P.polyval(eigenvalue, self.p[l])) / (lbda-eigenvalue)
                    p_kl += self.frozen_spectral_w[i]**2 * P.polyval(eigenvalue, self.p[k]) * frac
                else:
                    deriv = P.polyval(eigenvalue, P.polyder(self.p[l]))
                    p_kl += self.frozen_spectral_w[i]**2 * P.polyval(eigenvalue, self.p[k]) * deriv
        return p_kl

    def spectral_weight(self, lbda):
        q_i = 1 / np.linalg.norm([P.polyval(lbda, self.p[j]) for j in range(self.dim)])
        return q_i

    def compute_G(self, lbda, U):
        G = np.zeros((self.dim, self.dim))

        G[0, 0] = U[0, 1] * P.polyval(lbda, self.p[0]) * P.polyval(lbda, P.polyder(self.p[1])) \
                + U[0, 1] * P.polyval(lbda, self.p[1]) * P.polyval(lbda, P.polyder(self.p[0]))

        G[self.dim-1, self.dim-1] = - U[self.dim - 2, self.dim - 1] * P.polyval(lbda, self.p[self.dim - 2]) * P.polyval(lbda,P.polyder(self.p[self.dim - 1])) \
                                    + 1*P.polyval(lbda, self.p[self.dim - 1]) * P.polyval(lbda, P.polyder(self.p_n)) \
                                    - U[self.dim - 2, self.dim - 1] * P.polyval(lbda, self.p[self.dim - 1]) * P.polyval(lbda, P.polyder(self.p[self.dim - 2])) \
                                    + 1*P.polyval(lbda, self.p_n) * P.polyval(lbda, P.polyder(self.p[self.dim - 1]))

        for k in range(1, self.dim-1):
            G[k, k] = - U[k, k-1] * P.polyval(lbda, self.p[k - 1]) * P.polyval(lbda, P.polyder(self.p[k])) \
                      + U[k, k+1] * P.polyval(lbda, self.p[k]) * P.polyval(lbda, P.polyder(self.p[k+1])) \
                      - U[k, k-1] * P.polyval(lbda, self.p[k]) * P.polyval(lbda, P.polyder(self.p[k-1])) \
                      + U[k, k+1] * P.polyval(lbda, self.p[k+1]) * P.polyval(lbda, P.polyder(self.p[k]))

        for l in range(0, self.dim-1):
            G[l+1, l] = -U[l, l+1] * (P.polyval(lbda, self.p[l-1]) * P.polyval(lbda, P.polyder(self.p[l-1]))
                                    - P.polyval(lbda, self.p[l]) * P.polyval(lbda, P.polyder(self.p[l])))

        self.G = -(G + np.diag(np.diag(G, -1), +1))
        return self.G

    def compute_G_kl(self, k, l, U, eigenvalues):
        G_kl = np.zeros((self.dim, self.dim))
        if k == l:
            G_kl[k,k] = 1

        elif k == l+1:
            G_kl[k, l] = 1
            G_kl[l, k] = 1

        else:
            sum = 0
            for i, eigenvalue in enumerate(eigenvalues):
                acc = - U[0,1]*P.polyval(eigenvalue, self.p[0])*self.compute_p_kl(k=1, l=l, lbda=eigenvalue, eigenvalues=eigenvalues)\
                      - U[0,1]*P.polyval(eigenvalue, self.p[1])*self.compute_p_kl(k=0, l=l, lbda=eigenvalue, eigenvalues=eigenvalues)
                sum += self.frozen_spectral_w[i]**2 * P.polyval(eigenvalue, self.p[k]) * acc
            G_kl[0,0] = 0.5 * sum


            for r in range(1, self.dim-1):
                sum = 0
                for i, eigenvalue in enumerate(eigenvalues):
                    acc = U[r, r-1] * P.polyval(eigenvalue, self.p[r-1]) * self.compute_p_kl(k=r, l=l, lbda=eigenvalue, eigenvalues=eigenvalues)\
                        - U[r, r+1] * P.polyval(eigenvalue, self.p[r]) * self.compute_p_kl(k=r+1, l=l, lbda=eigenvalue, eigenvalues=eigenvalues)\
                        + U[r, r-1] * P.polyval(eigenvalue, self.p[r]) * self.compute_p_kl(k=r-1, l=l, lbda=eigenvalue, eigenvalues=eigenvalues)\
                        - U[r, r+1] * P.polyval(eigenvalue, self.p[r+1]) * self.compute_p_kl(k=r, l=l, lbda=eigenvalue, eigenvalues=eigenvalues)

                    sum += self.frozen_spectral_w[i]**2 * P.polyval(eigenvalue, self.p[k]) * acc
                G_kl[r, r] = 0.5 * sum

            for u in range(0, self.dim-2):
                sum=0
                for i, eigenvalue in enumerate(eigenvalues):
                    acc = U[r, r+1] * ( P.polyval(eigenvalue, self.p[r]) * self.compute_p_kl(k=r, l=l, lbda=eigenvalue, eigenvalues=eigenvalues)
                                      - P.polyval(eigenvalue, self.p[r+1]) * self.compute_p_kl(k=r+1, l=l, lbda=eigenvalue, eigenvalues=eigenvalues))

                    sum += self.frozen_spectral_w[i]**2 * P.polyval(eigenvalue, self.p[k]) * acc
                G_kl[u, u-1] = 0.5 * sum

        G_kl_tridiag = (G_kl+np.diag(np.diag(G_kl, -1), +1))
        return G_kl_tridiag


    def compute_brownians(self):
        draw_brownians = self.brownians[-1] + np.random.randn(self.n_traj) * (self.dt/self.n_traj)**0.5
        self.brownians.append(draw_brownians)

    def generate(self):
        self.compute_polynomials(self.tridiag_matrices[0])
        self.frozen_spectral_w = [self.spectral_weight(self.eigen_values[0][i]) for i in range(self.n_traj)]

        for sample in range(1, self.n_samples+1):
            self.compute_polynomials(self.tridiag_matrices[sample-1])
            self.compute_brownians()

            d_A = 0
            for i in range(self.n_traj):
                lbda_i = self.eigen_values[sample-1][i]
                G_i = self.compute_G(lbda_i, self.tridiag_matrices[sample-1])

                eigen_values_list = [lbda for lbda in list(self.eigen_values[sample - 1]) if lbda != lbda_i]
                sum_term = sum([(1 / (lbda_i - lbda_k)) for lbda_k in eigen_values_list])

                d_A -= (self.brownians[sample][i] + self.dt*(-lbda_i/2 + sum_term))*(self.frozen_spectral_w[i]**2)*G_i
            #print(d_A)

            d_P = 0
            for i in range(0, self.dim):
                lbda_i = self.eigen_values[sample-1][i]
                for k in range(0, self.dim):
                    for l in range(0, self.dim):
                        add = P.polyval(lbda_i, self.p[k]) * P.polyval(lbda_i, P.polyder(self.p[l])) + P.polyval(lbda_i, P.polyder(self.p[k])) * P.polyval(lbda_i, self.p[l])
                        d_P -= 0.5 * self.frozen_spectral_w[i]**2 * add * self.compute_G_kl(k, l, self.tridiag_matrices[sample-1], self.eigen_values[sample-1]) * self.dt

            draw_A = self.tridiag_matrices[sample-1] + d_A + d_P
            #print(d_P)
            #print(draw_A)
            self.tridiag_matrices.append(draw_A)
            self.eigen_values.append(sorted(np.linalg.eigvalsh(self.tridiag_matrices[sample]), reverse=True))
            print(eigen_values_list)

if __name__ == '__main__':
    test_tridiag = tridiag_dyson(n_traj=6, n_samples=10, tf=0.02)
    plot_traj_with_histo(test_tridiag, './plot/tridiag_dyson_histo.png')
