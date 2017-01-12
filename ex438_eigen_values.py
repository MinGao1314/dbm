import numpy as np

class DPP438_eigen_values:
    def __init__(self, n_traj, n_samples, tf, M):
        self.n_traj = n_traj
        self.n_samples = n_samples
        self.T = tf
        self.dt = tf/n_samples
        self.eigen_values = np.zeros((self.n_samples, self.n_traj))
        self.initialisation(M)
        self.generate(M)

    def initialisation(self, M):
        real_values = np.random.uniform(-0.05, 0.05, size=(self.n_traj, M))
        im_values = np.random.uniform(-0.05, 0.05, size=(self.n_traj, M))
        A = np.matrix(real_values + 1j * im_values)
        V_0 = np.dot(A , A.H)  # le .H conjugue puis transpose
        self.eigen_values[0] = sorted(np.real(np.linalg.eigvals(V_0)), reverse=False)
        return self.eigen_values

    def generate(self,M):
        for sample in range(self.n_samples-1):
            #print(self.eigen_values[sample])
            for i in range(self.n_traj):
                lbda_i = self.eigen_values[sample][i]
                eigen_values_list = [lbda for lbda in list(self.eigen_values[sample]) if lbda != lbda_i]
                #print(lbda_i, "\n", eigen_values_list)
                sum_term = sum ([ (lbda_k + lbda_i)/(lbda_k - lbda_i) for lbda_k in eigen_values_list ])
                print("sumterm", sum_term)
                W = (self.dt) ** (0.5) * np.random.randn()
                self.eigen_values[sample+1][i] = self.eigen_values[sample][i] + \
                                          2* W *(self.eigen_values[sample][i]/self.n_traj)**(0.5) + \
                                          2* ( (M/self.n_traj) + sum_term ) * self.dt
                print("eigen:", self.eigen_values[sample+1][i])

        return self.eigen_values


if __name__ == '__main__':
    test = DPP438_eigen_values(3, 5, 1, 2)
    print(test.eigen_values)